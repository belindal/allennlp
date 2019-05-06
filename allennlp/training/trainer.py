"""
A :class:`~allennlp.training.trainer.Trainer` is responsible for training a
:class:`~allennlp.models.model.Model`.

Typically you might create a configuration file specifying the model and
training parameters and then use :mod:`~allennlp.commands.train`
rather than instantiating a ``Trainer`` yourself.
"""
# pylint: disable=too-many-lines

import logging
import os
import shutil
import time
import re
import datetime
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, Set

import torch
import torch.optim.lr_scheduler
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel.scatter_gather import gather
from tensorboardX import SummaryWriter

from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import dump_metrics, gpu_memory_mb, parse_cuda_device, peak_memory_mb, scatter_kwargs
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.fields import SequenceLabelField
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training import active_learning_coref_utils
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metrics import MentionRecall, ConllCorefScores
from allennlp.training.optimizers import Optimizer

import pdb
import random

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def is_sparse(tensor):
    return tensor.is_sparse


def sparse_clip_norm(parameters, max_norm, norm_type=2) -> float:
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Supports sparse gradients.

    Parameters
    ----------
    parameters : ``(Iterable[torch.Tensor])``
        An iterable of Tensors that will have gradients normalized.
    max_norm : ``float``
        The max norm of the gradients.
    norm_type : ``float``
        The type of the used p-norm. Can be ``'inf'`` for infinity norm.

    Returns
    -------
    Total norm of the parameters (viewed as a single vector).
    """
    # pylint: disable=invalid-name,protected-access
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if is_sparse(p.grad):
                # need to coalesce the repeated indices before finding norm
                grad = p.grad.data.coalesce()
                param_norm = grad._values().norm(norm_type)
            else:
                param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if is_sparse(p.grad):
                p.grad.data._values().mul_(clip_coef)
            else:
                p.grad.data.mul_(clip_coef)
    return total_norm


def move_optimizer_to_cuda(optimizer):
    """
    Move the optimizer state to GPU, if necessary.
    After calling, any parameter specific state in the optimizer
    will be located on the same device as the parameter.
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.is_cuda:
                param_state = optimizer.state[param]
                for k in param_state.keys():
                    if isinstance(param_state[k], torch.Tensor):
                        param_state[k] = param_state[k].cuda(device=param.get_device())


class TensorboardWriter:
    """
    Wraps a pair of ``SummaryWriter`` instances but is a no-op if they're ``None``.
    Allows Tensorboard logging without always checking for Nones first.
    """
    def __init__(self, train_log: SummaryWriter = None, validation_log: SummaryWriter = None) -> None:
        self._train_log = train_log
        self._validation_log = validation_log

    @staticmethod
    def _item(value: Any):
        if hasattr(value, 'item'):
            val = value.item()
        else:
            val = value
        return val

    def add_train_scalar(self, name: str, value: float, global_step: int) -> None:
        # get the scalar
        if self._train_log is not None:
            self._train_log.add_scalar(name, self._item(value), global_step)

    def add_train_histogram(self, name: str, values: torch.Tensor, global_step: int) -> None:
        if self._train_log is not None:
            if isinstance(values, torch.Tensor):
                values_to_write = values.cpu().data.numpy().flatten()
                self._train_log.add_histogram(name, values_to_write, global_step)

    def add_validation_scalar(self, name: str, value: float, global_step: int) -> None:

        if self._validation_log is not None:
            self._validation_log.add_scalar(name, self._item(value), global_step)


def time_to_str(timestamp: int) -> str:
    """
    Convert seconds past Epoch to human readable string.
    """
    datetimestamp = datetime.datetime.fromtimestamp(timestamp)
    return '{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}'.format(
            datetimestamp.year, datetimestamp.month, datetimestamp.day,
            datetimestamp.hour, datetimestamp.minute, datetimestamp.second
    )


def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)


class Trainer(Registrable):
    default_implementation = "default"

    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 held_out_train_dataset: Optional[Iterable[Instance]] = None,
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 held_out_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False,
                 active_learning: Optional[Dict[str, int]] = None) -> None:
        """
        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_iterator : ``DataIterator``, optional (default=None)
            An iterator to use for the validation set.  If ``None``, then
            use the training `iterator`.
        shuffle: ``bool``, optional (default=True)
            Whether to shuffle the instances in the iterator or not.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : ``int``, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : ``int``, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        model_save_interval : ``float``, optional (default=None)
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        cuda_device : ``int``, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
        grad_norm : ``float``, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : ``PytorchLRScheduler``, optional, (default = None)
            A Pytorch learning rate scheduler. The learning rate will be decayed with respect to
            this schedule at the end of each epoch. If you use
            :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`, this will use the ``validation_metric``
            provided to determine if learning has plateaued.  To support updating the learning
            rate on every batch, this can optionally implement ``step_batch(batch_num_total)`` which
            updates the learning rate given the batch number.
        summary_interval: ``int``, optional, (default = 100)
            Number of batches between logging scalars to tensorboard
        histogram_interval : ``int``, optional, (default = ``None``)
            If not None, then log histograms to tensorboard every ``histogram_interval`` batches.
            When this parameter is specified, the following additional logging is enabled:
                * Histograms of model parameters
                * The ratio of parameter update norm to parameter norm
                * Histogram of layer activations
            We log histograms of the parameters returned by
            ``model.get_parameters_for_histogram_tensorboard_logging``.
            The layer activations are logged for any modules in the ``Model`` that have
            the attribute ``should_log_activations`` set to ``True``.  Logging
            histograms requires a number of GPU-CPU copies during training and is typically
            slow, so we recommend logging histograms relatively infrequently.
            Note: only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.
        should_log_parameter_statistics : ``bool``, optional, (default = True)
            Whether to send parameter statistics (mean and standard deviation
            of parameters and gradients) to tensorboard.
        should_log_learning_rate : ``bool``, optional, (default = False)
            Whether to send parameter specific learning rate to tensorboard.
        active_learning : ``Dict[str, int]``, optional, (default = None)
            Settings for active learning, ONLY applies if model is a CorefResolver
        """
        self.model = model
        self.iterator = iterator
        self._held_out_iterator = held_out_iterator
        self._validation_iterator = validation_iterator
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.train_data = train_dataset
        self._held_out_train_data = held_out_train_dataset
        self._validation_data = validation_dataset

        if patience is None:  # no early stopping
            if validation_dataset:
                logger.warning('You provided a validation dataset but patience was set to None, '
                               'meaning that early stopping is disabled')
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError('{} is an invalid value for "patience": it must be a positive integer '
                                     'or None (if you want to disable early stopping)'.format(patience))
        self._patience = patience
        self._num_epochs = num_epochs

        self._serialization_dir = serialization_dir
        self._num_serialized_models_to_keep = num_serialized_models_to_keep
        self._keep_serialized_model_every_num_seconds = keep_serialized_model_every_num_seconds
        self._serialized_paths: List[Any] = []
        self._last_permanent_saved_checkpoint_time = time.time()
        self._model_save_interval = model_save_interval

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._learning_rate_scheduler = learning_rate_scheduler

        increase_or_decrease = validation_metric[0]
        if increase_or_decrease not in ["+", "-"]:
            raise ConfigurationError("Validation metrics must specify whether they should increase "
                                     "or decrease by pre-pending the metric name with a +/-.")
        self._validation_metric = validation_metric[1:]
        self._validation_metric_decreases = increase_or_decrease == "-"

        if not isinstance(cuda_device, int) and not isinstance(cuda_device, list):
            raise ConfigurationError("Expected an int or list for cuda_device, got {}".format(cuda_device))

        if isinstance(cuda_device, list):
            logger.warning(f"Multiple GPU support is experimental not recommended for use. "
                           "In some cases it may lead to incorrect results or undefined behavior.")
            self._multiple_gpu = True
            self._cuda_devices = cuda_device
        else:
            self._multiple_gpu = False
            self._cuda_devices = [cuda_device]

        if self._cuda_devices[0] != -1:
            self.model = self.model.cuda(self._cuda_devices[0])

        self._log_interval = 10  # seconds
        self._summary_interval = summary_interval
        self._histogram_interval = histogram_interval
        self._log_histograms_this_batch = False
        self._should_log_parameter_statistics = should_log_parameter_statistics
        self._should_log_learning_rate = should_log_learning_rate

        # We keep the total batch number as a class variable because it
        # is used inside a closure for the hook which logs activations in
        # ``_enable_activation_logging``.
        self._batch_num_total = 0

        self._last_log = 0.0  # time of last logging

        if serialization_dir is not None:
            train_log = SummaryWriter(os.path.join(serialization_dir, "log", "train"))
            validation_log = SummaryWriter(os.path.join(serialization_dir, "log", "validation"))
            self._tensorboard = TensorboardWriter(train_log, validation_log)
        else:
            self._tensorboard = TensorboardWriter()
        self._warned_tqdm_ignores_underscores = False

        # Whether or not to do active learning
        self._do_active_learning = False
        self.DEBUG_BREAK_FLAG = False
        if active_learning:
            if active_learning['model_type'] != 'coref':
                raise ConfigurationError("Active learning only compatible with coreference model (for now)")
            self._do_active_learning = True
            self._active_learning_epoch_interval = active_learning['epoch_interval']
            self._use_percent_labels = active_learning['use_percent'] is not None and active_learning['use_percent']
            if self._use_percent_labels:
                self._active_learning_percent_labels = active_learning['num_labels']
            else:
                self._active_learning_num_labels = active_learning['num_labels']
            self._sample_from_training = active_learning['simulate_user_inputs']
            self._active_learning_patience = active_learning['patience']
            self._percent_label_experiments = True if 'percent_label_experiments' in active_learning else False
            self._replace_with_next_pos_edge = active_learning['replace_with_next_pos_edge']
            if self._percent_label_experiments:
                self._percent_labels = active_learning['percent_label_experiments']['percent_labels']
                assert(self._percent_labels >= 0 and self._percent_labels <= 1)
            pdb.set_trace()
            self._selector = active_learning['selector']['type'] if 'selector' in active_learning else 'entropy'
            self._selector_clusters = active_learning['selector']['use_clusters'] if 'selector' in active_learning else True
            assert(self._selector == 'random' or self._selector == 'score' or self._selector == 'entropy')
            self._query_type = active_learning['query_type'] if 'query_type' in active_learning else 'discrete'
            assert(self._query_type == 'pairwise' or self._query_type == 'discrete')

    def _enable_gradient_clipping(self) -> None:
        if self._grad_clipping is not None:
            # Pylint is unable to tell that we're in the case that _grad_clipping is not None...
            # pylint: disable=invalid-unary-operand-type
            clip_function = lambda grad: grad.clamp(-self._grad_clipping, self._grad_clipping)
            for parameter in self.model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

    def _enable_activation_logging(self) -> None:
        """
        Log activations to tensorboard
        """
        if self._histogram_interval is not None:
            # To log activation histograms to the forward pass, we register
            # a hook on forward to capture the output tensors.
            # This uses a closure on self._log_histograms_this_batch to
            # determine whether to send the activations to tensorboard,
            # since we don't want them on every call.
            for _, module in self.model.named_modules():
                if not getattr(module, 'should_log_activations', False):
                    # skip it
                    continue

                def hook(module_, inputs, outputs):
                    # pylint: disable=unused-argument,cell-var-from-loop
                    log_prefix = 'activation_histogram/{0}'.format(module_.__class__)
                    if self._log_histograms_this_batch:
                        if isinstance(outputs, torch.Tensor):
                            log_name = log_prefix
                            self._tensorboard.add_train_histogram(log_name,
                                                                  outputs,
                                                                  self._batch_num_total)
                        elif isinstance(outputs, (list, tuple)):
                            for i, output in enumerate(outputs):
                                log_name = "{0}_{1}".format(log_prefix, i)
                                self._tensorboard.add_train_histogram(log_name,
                                                                      output,
                                                                      self._batch_num_total)
                        elif isinstance(outputs, dict):
                            for k, tensor in outputs.items():
                                log_name = "{0}_{1}".format(log_prefix, k)
                                self._tensorboard.add_train_histogram(log_name,
                                                                      tensor,
                                                                      self._batch_num_total)
                        else:
                            # skip it
                            pass

                module.register_forward_hook(hook)

    def rescale_gradients(self) -> Optional[float]:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
        """
        if self._grad_norm:
            parameters_to_clip = [p for p in self.model.parameters()
                                  if p.grad is not None]
            return sparse_clip_norm(parameters_to_clip, self._grad_norm)
        return None

    def _data_parallel(self, batch):
        """
        Do the forward pass using multiple GPUs.  This is a simplification
        of torch.nn.parallel.data_parallel to support the allennlp model
        interface.
        """
        inputs, module_kwargs = scatter_kwargs((), batch, self._cuda_devices, 0)

        used_device_ids = self._cuda_devices[:len(inputs)]
        replicas = replicate(self.model, used_device_ids)
        outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)

        # Only the 'loss' is needed.
        # a (num_gpu, ) tensor with loss on each GPU
        losses = gather([output['loss'].unsqueeze(0) for output in outputs], used_device_ids[0], 0)
        return {'loss': losses.mean()}

    def batch_loss(self, batch: torch.Tensor, for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batch and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = self._data_parallel(batch)
        else:
            batch = util.move_to_device(batch, self._cuda_devices[0])
            output_dict = self.model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None

        return loss

    def _get_metrics(self, total_loss: float, num_batches: int, reset: bool = False) -> Dict[str, float]:
        """
        Gets the metrics but sets ``"loss"`` to
        the total loss divided by the ``num_batches`` so that
        the ``"loss"`` metric is "average loss per batch".
        """
        metrics = self.model.get_metrics(reset=reset)
        metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
        return metrics

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        logger.info(f"Peak CPU memory usage MB: {peak_memory_mb()}")
        for gpu, memory in gpu_memory_mb().items():
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()

        # Get tqdm for the training batches
        train_generator = self.iterator(self.train_data,
                                        num_epochs=1,
                                        shuffle=self.shuffle)
        num_training_batches = self.iterator.get_num_batches(self.train_data)
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        if self._histogram_interval is not None:
            histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator,
                                         total=num_training_batches)
        for batch in train_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self._log_histograms_this_batch = self._histogram_interval is not None and (
                    batch_num_total % self._histogram_interval == 0)

            self.optimizer.zero_grad()

            loss = self.batch_loss(batch, for_training=True)
            try:
                torch.cuda.empty_cache()
                loss.backward()
            except:
                pdb.set_trace()

            train_loss += loss.item()

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using an
            # LRScheduler which doesn't update per batch.
            if self._learning_rate_scheduler and (self._held_out_train_data is None or
                                                  len(self._held_out_train_data) == 0):
                self._learning_rate_scheduler.step_batch(batch_num_total)

            if self._log_histograms_this_batch:
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {name: param.detach().cpu().clone()
                                 for name, param in self.model.named_parameters()}
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                       update_norm / (param_norm + 1e-7),
                                                       batch_num_total)
            else:
                self.optimizer.step()

            # Update the description with the latest metrics
            metrics = self._get_metrics(train_loss, batches_this_epoch)
            description = self._description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if batch_num_total % self._summary_interval == 0:
                if self._should_log_parameter_statistics:
                    self._parameter_and_gradient_statistics_to_tensorboard(batch_num_total, batch_grad_norm)
                if self._should_log_learning_rate:
                    self._learning_rates_to_tensorboard(batch_num_total)
                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"], batch_num_total)
                self._metrics_to_tensorboard(batch_num_total,
                                             {"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._log_histograms_this_batch:
                self._histograms_to_tensorboard(batch_num_total, histogram_parameters)

            # Save model if needed.
            if self._model_save_interval is not None and (
                    time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                        '{0}.{1}'.format(epoch, time_to_str(int(last_save_time))), [], is_best=False
                )

        return self._get_metrics(train_loss, batches_this_epoch, reset=True)

    def _should_stop_early(self, metric_history: List[float], patience: int) -> bool:
        """
        uses patience and the validation metric to determine if training should stop early
        """
        if patience and patience < len(metric_history):
            # Pylint can't figure out that in this branch `self._patience` is an int.
            # pylint: disable=invalid-unary-operand-type

            # Is the best score in the past N epochs worse than or equal the best score overall?
            if self._validation_metric_decreases:
                return min(metric_history[-patience:]) >= min(metric_history[:-patience])
            else:
                return max(metric_history[-patience:]) <= max(metric_history[:-patience])

        return False

    def _parameter_and_gradient_statistics_to_tensorboard(self, # pylint: disable=invalid-name
                                                          epoch: int,
                                                          batch_grad_norm: float) -> None:
        """
        Send the mean and std of all parameters and gradients to tensorboard, as well
        as logging the average gradient norm.
        """
        # Log parameter values to Tensorboard
        for name, param in self.model.named_parameters():
            self._tensorboard.add_train_scalar("parameter_mean/" + name,
                                               param.data.mean(),
                                               epoch)
            self._tensorboard.add_train_scalar("parameter_std/" + name, param.data.std(), epoch)
            if param.grad is not None:
                if is_sparse(param.grad):
                    # pylint: disable=protected-access
                    grad_data = param.grad.data._values()
                else:
                    grad_data = param.grad.data

                # skip empty gradients
                if torch.prod(torch.tensor(grad_data.shape)).item() > 0: # pylint: disable=not-callable
                    self._tensorboard.add_train_scalar("gradient_mean/" + name,
                                                       grad_data.mean(),
                                                       epoch)
                    self._tensorboard.add_train_scalar("gradient_std/" + name,
                                                       grad_data.std(),
                                                       epoch)
                else:
                    # no gradient for a parameter with sparse gradients
                    logger.info("No gradient for %s, skipping tensorboard logging.", name)
        # norm of gradients
        if batch_grad_norm is not None:
            self._tensorboard.add_train_scalar("gradient_norm",
                                               batch_grad_norm,
                                               epoch)

    def _learning_rates_to_tensorboard(self, batch_num_total: int):
        """
        Send current parameter specific learning rates to tensorboard
        """
        # optimizer stores lr info keyed by parameter tensor
        # we want to log with parameter name
        names = {param: name for name, param in self.model.named_parameters()}
        for group in self.optimizer.param_groups:
            if 'lr' not in group:
                continue
            rate = group['lr']
            for param in group['params']:
                # check whether params has requires grad or not
                effective_rate = rate * float(param.requires_grad)
                self._tensorboard.add_train_scalar(
                        "learning_rate/" + names[param],
                        effective_rate,
                        batch_num_total
                )

    def _histograms_to_tensorboard(self, epoch: int, histogram_parameters: Set[str]) -> None:
        """
        Send histograms of parameters to tensorboard.
        """
        for name, param in self.model.named_parameters():
            if name in histogram_parameters:
                self._tensorboard.add_train_histogram("parameter_histogram/" + name,
                                                      param,
                                                      epoch)

    def _metrics_to_tensorboard(self,
                                epoch: int,
                                train_metrics: dict,
                                val_metrics: dict = None) -> None:
        """
        Sends all of the train metrics (and validation metrics, if provided) to tensorboard.
        """
        metric_names = set(train_metrics.keys())
        if val_metrics is not None:
            metric_names.update(val_metrics.keys())
        val_metrics = val_metrics or {}

        for name in metric_names:
            train_metric = train_metrics.get(name)
            if train_metric is not None:
                self._tensorboard.add_train_scalar(name, train_metric, epoch)
            val_metric = val_metrics.get(name)
            if val_metric is not None:
                self._tensorboard.add_validation_scalar(name, val_metric, epoch)

    def _metrics_to_console(self,  # pylint: disable=no-self-use
                            train_metrics: dict,
                            val_metrics: dict = None) -> None:
        """
        Logs all of the train metrics (and validation metrics, if provided) to the console.
        """
        val_metrics = val_metrics or {}
        dual_message_template = "%s |  %8.3f  |  %8.3f"
        no_val_message_template = "%s |  %8.3f  |  %8s"
        no_train_message_template = "%s |  %8s  |  %8.3f"
        header_template = "%s |  %-10s"

        metric_names = set(train_metrics.keys())
        if val_metrics:
            metric_names.update(val_metrics.keys())

        name_length = max([len(x) for x in metric_names])

        logger.info(header_template, "Training".rjust(name_length + 13), "Validation")
        for name in metric_names:
            train_metric = train_metrics.get(name)
            val_metric = val_metrics.get(name)

            if val_metric is not None and train_metric is not None:
                logger.info(dual_message_template, name.ljust(name_length), train_metric, val_metric)
            elif val_metric is not None:
                logger.info(no_train_message_template, name.ljust(name_length), "N/A", val_metric)
            elif train_metric is not None:
                logger.info(no_val_message_template, name.ljust(name_length), train_metric, "N/A")

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self.model.eval()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        val_generator = val_iterator(self._validation_data,
                                     num_epochs=1,
                                     shuffle=False)
        num_validation_batches = val_iterator.get_num_batches(self._validation_data)
        val_generator_tqdm = Tqdm.tqdm(val_generator,
                                       total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch in val_generator_tqdm:

            torch.cuda.empty_cache()
            loss = self.batch_loss(batch, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = self._get_metrics(val_loss, batches_this_epoch)
            description = self._description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        return val_loss, batches_this_epoch

    def _translate_to_indA(self, edges, output_dict, all_spans, translation_reference = None) -> torch.LongTensor:
        """
        :param edges: Tensor (Nx3) of N edges, each of form [instance in batch, indB of proform, indC of antecedent]
        output_dict: holds information for translation
        :param translation_reference: Tensor (BxS) of B batches and S top_spans, whereby number at [b,s] represents index of
            to translate sth *top* span in batch b in *all* spans
        :return: Tensor (Nx3) of N edges, which indices translated to indA. Each edge has form
        [instance in batch, indA of proform, indA of antecedent]
        """
        if len(edges) == 0:
            return edges

        # NOTE in below code-- each span has 3 indices, will refer to them as A,B,C respectively:
        #   indA. index in all spans (i.e. in the training data)
        #   indB. index in output_dict['top_spans']
        #   indC. index in output_dict['predicted_antecedents'] (antecedents only)
        # note output_dict['antecedent_indices'] translates indB <-> indC by:
        # indB = output_dict['antecedent_indices'][instance, proform idx, indC]
        if translation_reference is not None:
            indA_edges = edges.clone()
            indA_edges[:,2] = output_dict['antecedent_indices'][indA_edges[:,0], indA_edges[:,1], indA_edges[:,2]]
            indA_edges[:,1] = translation_reference[indA_edges[:,0], indA_edges[:,1]]
            indA_edges[:,2] = translation_reference[indA_edges[:,0], indA_edges[:,2]]
            return indA_edges

        instances = edges[:, 0]
        ind_proforms = edges[:, 1]
        ind_antecedents = edges[:, 2]
        ind_antecedents = output_dict['antecedent_indices'][instances, ind_proforms, ind_antecedents] #indB

        proform_spans = output_dict['top_spans'][instances, ind_proforms]
        antecedent_spans = output_dict['top_spans'][instances, ind_antecedents]
        chunk_size = 10000
        if len(proform_spans) > chunk_size:
            # too big for cuda, break into chunks
            indA_proforms = torch.empty(instances.size(), dtype=torch.long).cuda(self._cuda_devices[0])
            indA_antecedents = torch.empty(instances.size(), dtype=torch.long).cuda(self._cuda_devices[0])
            i = 0
            while i < len(proform_spans):
                try:
                    instances_chunk = instances[i:i+chunk_size]
                    proform_span_chunk = proform_spans[i:i+chunk_size]
                    antecedent_span_chunk = antecedent_spans[i:i+chunk_size]
                    indA_proforms[i:i+chunk_size] = ((proform_span_chunk.unsqueeze(1) - all_spans[instances_chunk]).abs().sum(-1) == 0).nonzero()[:, 1]
                    indA_antecedents[i:i+chunk_size] = ((antecedent_span_chunk.unsqueeze(1) - all_spans[instances_chunk]).abs().sum(-1) == 0).nonzero()[:, 1]
                    i += chunk_size
                    # should use < 75% of GPU memory
                    assert(torch.cuda.memory_cached(instances.device.index) + torch.cuda.memory_allocated(instances.device.index)
                           < 0.75 * torch.cuda.get_device_properties(instances.device.index).total_memory)
                except:
                    torch.cuda.empty_cache()
                    chunk_size = int(chunk_size / 2)
        else:
            indA_proforms = ((proform_spans.unsqueeze(1) - all_spans[instances]).abs().sum(-1) == 0).nonzero()[:, 1]
            indA_antecedents = ((antecedent_spans.unsqueeze(1) - all_spans[instances]).abs().sum(-1) == 0).nonzero()[:, 1]
        return torch.stack([instances, indA_proforms, indA_antecedents], dim=-1)

    def _get_sorted_masked_edges(self, coreference_mask, output_dict, all_spans, farthest_from_zero=False) -> torch.LongTensor:
        """
        :param coreference_mask: should be a boolean tensor with size equal to output_dict["coreference_scores"]
        :param output_dict: should have a field "coreference_scores"
        :param farthest_from_zero: in the case of non-random sorting,
                                   True if sort by farthest_from_zero (most certain), False if sort by closest to 0 (least certain)
        :return: edges, sorted in farthest_from_zero order, and their corresponding scores
        """
        masked_edge_inds = coreference_mask.nonzero()
        if masked_edge_inds.size(0) == 0:
            # no edges, just return empty arrays (note masked_edge_inds is empty)
            return masked_edge_inds, masked_edge_inds
        # Subtract one here because index 0 is the "no antecedent" class,
        # so this makes the indices line up with actual spans if the prediction
        # is greater than -1.
        masked_edge_inds[:, 2] -= 1
        edge_scores = output_dict['coreference_scores'][coreference_mask]
        if self._selector == 'score':
            # get sorted closest/furthest from 0 scores
            _, ind_max_edge_scores = edge_scores.abs().sort(descending=farthest_from_zero)
        elif self._selector == 'random':
            # using random selector
            ind_max_edge_scores = torch.randperm(len(masked_edge_inds))
        else: # selector is entropy
            pdb.set_trace()
        sorted_edges = self._translate_to_indA(masked_edge_inds[ind_max_edge_scores], output_dict, all_spans)
        return sorted_edges, edge_scores[ind_max_edge_scores]

    def _filter_gold_cluster_edges(self, chosen_edges, span_labels):
        """
        :param chosen_edges: edges chosen to verify
        :param span_labels: from batch['span_labels']
        Filter out edges for which both ends are in the same, or different, gold clusters already
        """
        proform_gold_labels = span_labels[chosen_edges[:, 0], chosen_edges[:, 1]]
        antecedent_gold_labels = span_labels[chosen_edges[:, 0], chosen_edges[:, 2]]
        edges_both_ends_in_gold_clusters_mask = (proform_gold_labels != -1) & (antecedent_gold_labels != -1)
        chosen_edges = chosen_edges[~edges_both_ends_in_gold_clusters_mask]
        return chosen_edges

    def _query_user_labels_pairwise(self, chosen_edges, edge_scores, user_labels, num_labels_to_query, return_all_edges,
                                    use_alt_edges=False, all_candidate_alt_edges=None, num_alts_to_check=0, alt_edge_scores=None):
        """
        :param chosen_edges: should be sorted with most uncertain first, with edges in which both ends in gold clusters filtered out
        :param user_labels: from batch['user_labels']
        :param num_labels_to_query:
        :param return_all_edges: returns all deemed coreferent edges, regardless of num_labels_to_query,
                                 if False, returns only up to num_labels_to_query edges
        :param use_alt_edges: replace non-coreferent positive edges with next most certain option
        :param all_candidate_alt_edges:
        :param num_alts_to_check: # of alternate edges to verify coreference
        :param alt_edge_scores: scores of alternate edges (all_candidate_alt-edges)
        :return:
        """
        total_possible_queries = len(chosen_edges) + len(all_candidate_alt_edges) if all_candidate_alt_edges is not None else len(chosen_edges)
        pos_edges_mask = (edge_scores[:num_labels_to_query] > 0)
        num_labels_to_query = len(chosen_edges[:num_labels_to_query])
        num_alt_edge_queried = 0
        if num_labels_to_query > 0:
            if self._sample_from_training:
                proform_user_labels = user_labels[chosen_edges[:num_labels_to_query][:, 0], chosen_edges[:num_labels_to_query][:, 1]]
                antecedent_user_labels = user_labels[chosen_edges[:num_labels_to_query][:, 0], chosen_edges[:num_labels_to_query][:, 2]]
                coreferent_mask = (proform_user_labels == antecedent_user_labels) & (proform_user_labels != -1)
                # ensure all edges deemed coreferent are assigned positive scores
                edge_scores[:num_labels_to_query][coreferent_mask] = edge_scores[:num_labels_to_query][coreferent_mask].abs()
                non_coreferent_pos_edges = chosen_edges[:num_labels_to_query][~coreferent_mask & pos_edges_mask]
                if len(non_coreferent_pos_edges) > 0 and not use_alt_edges:
                    # set all non-coreferent edges to -1
                    chosen_edges[:num_labels_to_query][~coreferent_mask & pos_edges_mask] = -1
                elif len(non_coreferent_pos_edges) > 0:
                    # use alternate edges
                    assert all_candidate_alt_edges is not None
                    assert alt_edge_scores is not None
                    # replace non-coreferent and (+) edges with alternate edges (same proform, next largest antecedent)
                    allalt_differences = (non_coreferent_pos_edges.unsqueeze(0) - all_candidate_alt_edges.unsqueeze(1)).abs()
                    same_proform_diff_antecedent_mask = (allalt_differences[:,:,1] == 0) & (
                                                         allalt_differences[:,:,1] == 0) & (
                                                         allalt_differences[:,:,2] != 0)
                    # [inds of possible alternates in all_candidate_alt_edges, inds of edges to replace in non_coreferent_pos_edges]
                    possible_alt_inds_to_query = same_proform_diff_antecedent_mask.nonzero()[:num_alts_to_check]
                    num_alt_edge_queried += len(possible_alt_inds_to_query)
                    alternate_pos_edges = -torch.ones(non_coreferent_pos_edges.size(), dtype=torch.long,
                                                      ).cuda(self._cuda_devices[0])
                    chosen_alternate_edge_scores = -torch.ones(non_coreferent_pos_edges.size(0), dtype=torch.float,
                                                               ).cuda(self._cuda_devices[0])
                    if possible_alt_inds_to_query.size(0) > 0:
                        possible_alts_to_query = all_candidate_alt_edges[possible_alt_inds_to_query[:,0]]
                        alt_proforms = user_labels[possible_alts_to_query[:,0], possible_alts_to_query[:,1]]
                        alt_antecedents = user_labels[possible_alts_to_query[:,0], possible_alts_to_query[:,2]]
                        coreferent_alts_mask = (alt_proforms >= 0) & (alt_proforms == alt_antecedents)
                        # flip since for the same proform, want to set to highest-scoring coreferent antecedent, which
                        # means want highest-scoring at the end
                        coreferent_alt_inds = possible_alt_inds_to_query[coreferent_alts_mask].flip(0)
                        if len(coreferent_alt_inds) > 0:
                            
                            alternate_pos_edges[coreferent_alt_inds[:,1]] = all_candidate_alt_edges[coreferent_alt_inds[:,0]]
                            # also set new edge score
                            chosen_alternate_edge_scores[coreferent_alt_inds[:,1]] = alt_edge_scores[coreferent_alt_inds[:,0]]
                    # TODO: not sure why this errors out sometimes... but always fine on retry...
                    try:
                        chosen_edges[:num_labels_to_query][~coreferent_mask & pos_edges_mask] = alternate_pos_edges
                    except:
                        try:
                            chosen_edges[:num_labels_to_query][~coreferent_mask & pos_edges_mask] = alternate_pos_edges
                        except:
                            pdb.set_trace()
                    edge_scores[:num_labels_to_query][~coreferent_mask & pos_edges_mask] = chosen_alternate_edge_scores
                # filter -1s
                filter_deleted_edges_mask = chosen_edges[:, 0] >= 0
                chosen_edges = chosen_edges[filter_deleted_edges_mask]
                edge_scores = edge_scores[filter_deleted_edges_mask]
            else:
                # iterate through chosen edges (note iterating through inds given by ind_min_exist_edge_scores)
                for i, edge in enumerate(chosen_edges[:num_labels_to_query]):
                    ind_instance = edge[0]  # index in batch
                    # TODO: mechanism for printing chosen_proform_span and chosen_antecedent_span to user and getting user input
                    coreferent = True
        # add all edges with positive scores (including unchecked edges that the model predicted)
        if return_all_edges:
            chosen_edges = chosen_edges[edge_scores >= 0]
        else:
            chosen_edges = chosen_edges[:num_labels_to_query][coreferent_mask]
        return chosen_edges, num_labels_to_query + num_alt_edge_queried, total_possible_queries

    def _query_user_labels_discrete(self, chosen_edges, edge_scores, num_labels_to_query, return_all_edges,
                                    output_dict, batch):
        """
        :param chosen_edges: should be sorted with most uncertain first, with edges in which both ends in gold clusters filtered out
        :param user_labels: from batch['user_labels']
        :param num_labels_to_query:
        :return:
        NOTE on implementation: sets scores of checked, coreferent edges to be positive, scores of checked, non-coreferenet edges
        to be negative (if return_all_edges is True)
        """
        user_labels = batch['user_labels']
        span_labels = batch['span_labels']
        # Verify all edges and alt edges
        total_possible_queries = len(chosen_edges)
        first_spans_of_clusters = {}
        num_labels_queried = 0
        i = 0
        if num_labels_to_query > 0:
            # use for loop, as may want to change things depending on which edge we're currently querying, also a few hundred
            # mentions = not too expensive
            while i < len(chosen_edges):
                if num_labels_queried >= num_labels_to_query:
                    break
                proform_span_label = span_labels[chosen_edges[i,0], chosen_edges[i,1]]
                antecedent_span_label = span_labels[chosen_edges[i,0], chosen_edges[i,2]]
                if proform_span_label != -1 and proform_span_label == antecedent_span_label:
                    i += 1
                    continue
                proform_user_label = user_labels[chosen_edges[i][0], chosen_edges[i,1]]
                antecedent_user_label = user_labels[chosen_edges[i,0], chosen_edges[i,2]]
                num_labels_queried += 1
                # ask if edge is coreferent
                coreferent = (proform_user_label == antecedent_user_label) & (proform_user_label != -1)
                if not coreferent and proform_user_label != -1:
                    # query a coreferent spans from user
                    user_spans_in_proform_cluster_mask = user_labels[chosen_edges[i,0]] == proform_user_label
                    user_spans_in_proform_cluster = user_spans_in_proform_cluster_mask.nonzero()
                    if len(user_spans_in_proform_cluster) > 1:  # more than 1 element in cluster
                        # if we are 1st span in cluster, choose arbitrary span outside of spans already in our cluster
                        if chosen_edges[i,1] == user_spans_in_proform_cluster[0]:
                            spans_in_curr_proform_cluster_mask = (proform_span_label != -1) & (span_labels[chosen_edges[i,0]] == proform_span_label)
                            user_spans_outside_curr_proform_cluster = (user_spans_in_proform_cluster_mask &
                                              ~spans_in_curr_proform_cluster_mask).nonzero()
                            user_spans_outside_curr_proform_cluster = user_spans_outside_curr_proform_cluster[
                                user_spans_outside_curr_proform_cluster != chosen_edges[i,1]]
                            # TODO: if there are spans in the cluster besides what we have currently, choose random among them
                            # otherwise, no additional spans to add, delete edge (and all ingoing and outgoing edges from cluster)
                            if len(user_spans_outside_curr_proform_cluster.nonzero()) > 0:
                                new_antecedent = user_spans_outside_curr_proform_cluster[torch.randint(len(user_spans_outside_curr_proform_cluster), (), dtype=torch.long)]
                                chosen_edges[i,2] = chosen_edges[i,1]
                                chosen_edges[i,1] = new_antecedent

                                # redirect future edges out of new_antecedent -> S to go from S -> first span
                                outgoing_new_antecedent_span_mask = (chosen_edges[:,1] == new_antecedent)
                                outgoing_new_antecedent_span_mask[:i+1] = 0   # ensure only setting future edges
                                chosen_edges[:,1][outgoing_new_antecedent_span_mask] = chosen_edges[:,2][outgoing_new_antecedent_span_mask]
                                chosen_edges[:,2][outgoing_new_antecedent_span_mask] = chosen_edges[i,2]  # point to first span of cluster (the same as chosen_edges[i] is pointing to)
                                # delete future edges that are now invalid (because impossible to be before first span and still in clustetr)
                                invalid_edges_mask = (chosen_edges[:,2] == chosen_edges[i,2]) & (chosen_edges[:,1] <= chosen_edges[:,2])
                                invalid_edges_mask[:i+1] = 0  # ensure edges that have been verified aren't deemed 'invalid'
                                chosen_edges = chosen_edges[~invalid_edges_mask]
                                edge_scores = edge_scores[~invalid_edges_mask]

                                coreferent = True
                            else:
                                coreferent = False
                                # cluster is complete; delete all remaining ingoing and outgoing edges from cluster (and within cluster)
                                if i < len(chosen_edges) - 1:  # we are not on last edge (there are still remaining edges to check)
                                    remaining_edges_to_and_from_cluster = ((chosen_edges[i+1:].unsqueeze(0) - user_spans_in_proform_cluster.unsqueeze(-1)).abs() == 0).nonzero()
                                    if len(remaining_edges_to_and_from_cluster) > 0:
                                        remaining_edges_to_and_from_cluster[:, 1] += i + 1
                                        # delete rows at remaining_edges_to_and_from_cluster[:,1]
                                        chosen_edges[remaining_edges_to_and_from_cluster[:,1], :] = -1
                                        mask = chosen_edges[:,0] != -1
                                        chosen_edges = chosen_edges[mask]
                                        edge_scores = edge_scores[mask]
                        else:
                            # TODO: don't ask current antecedent (chosen_edges[i,2]) to ANY element of cluster involving current proform (chosen_edges[i,1])
                            new_antecedent = user_spans_in_proform_cluster[0]
                            delete_first_antecedent_span_mask = (chosen_edges[:,1] != new_antecedent)
                            delete_first_antecedent_span_mask[:i+1] = 1  # ensure we don't delete verified edges
                            chosen_edges = chosen_edges[delete_first_antecedent_span_mask]
                            edge_scores = edge_scores[delete_first_antecedent_span_mask]
                            chosen_edges[i, 2] = new_antecedent
                            coreferent = True
                if coreferent:  # set score positive
                    edge_scores[i] = edge_scores[i].abs()
                    # update span labels...
                    span_labels = self._update_clusters_with_edge(span_labels, chosen_edges[i])
                else:  # set score negative
                    edge_scores[i] = -edge_scores[i].abs()
                i += 1
        if return_all_edges:
            # add unadded positive edges to span_labels--i is last unchecked edge
            unadded_pos_edges = chosen_edges[i:][edge_scores[i:] >= 0]
            chosen_edges = chosen_edges[edge_scores >= 0]
            for edge in unadded_pos_edges:
                span_labels = self._update_clusters_with_edge(span_labels, edge)
        else:
            chosen_edges = chosen_edges[edge_scores >= 0][:num_labels_to_query]
        return chosen_edges, num_labels_queried, total_possible_queries, span_labels

    def _query_user_labels_mention(self, mention, output_dict, all_spans, user_labels, translation_reference=None):
        # returns: 
        # 1. edge: indA of edge, if coreferent, will be identical to indA_edge_ask, otherwise,
        #          edge antecedent will be "fixed", pointing to -1 if there are no other spans
        #          in cluster, pointing to itself if it is 1st/HEAD span of cluster, or pointing
        #          to 1st span of in cluster if in cluster and not HEAD span
        indC_antecedent = output_dict['predicted_antecedents'][mention[0], mention[1]]
        if indC_antecedent < 0:
            # ask about most likely non-null antecedent
            indC_antecedent = \
                output_dict['coreference_scores'][:,:,1:][mention[0], mention[1]].argmax()
        edge_ask = torch.cat((mention, indC_antecedent.unsqueeze(0)))
        indA_edge_ask = self._translate_to_indA(edge_ask.unsqueeze(0), output_dict, all_spans, translation_reference).squeeze()
        proform_label = user_labels[indA_edge_ask[0], indA_edge_ask[1]]
        antecedent_label = user_labels[indA_edge_ask[0], indA_edge_ask[2]]
        # proform and antecedent both belong to a cluster, and it is the same cluster
        coreferent = (proform_label == antecedent_label) & (proform_label != -1)
        edge = indA_edge_ask.clone()  # batch & proform are the same, set antecedent by cases
        if not coreferent:
            if proform_label == -1:
                # proform not coreferent to anything else in document
                edge[2] = -1
            else:
                # query user for correct label (point to 1st span in cluster--if it is 1st span, will point to itself)
                edge[2] = (user_labels[indA_edge_ask[0]] == proform_label).nonzero()[0]
        return edge, edge_ask, indA_edge_ask

    def _find_next_most_uncertain_mention(self, model_labels, output_dict, queried_mentions_mask):
        '''
        model_labels: batch x num_spans tensor detailing cluster ID of cluster each span belongs to, according to model edges
        and user corrections. IMPORTANT: indexes into TOP_SPANS, not all spans.
        '''
        coref_scores_mask = output_dict['coreference_scores'] != -float("inf")
        mention_confidence_scores = torch.zeros(output_dict['top_spans'].size()[:2], dtype=torch.float).cuda(self._cuda_devices[0])
        for b, score_instance in enumerate(output_dict['coreference_scores']):
            # Sum across all probabilities link to each cluster for each mention
            top_span_mask = torch.ones(output_dict['top_spans'][b].size(0), dtype=torch.float).cuda(self._cuda_devices[0]).unsqueeze(-1)
            coreference_probs = util.masked_log_softmax(score_instance, top_span_mask).exp()
            # output_dict['antecedent_indices'], and what cluster that antecedent belongs to (-1 for no cluster)
            model_output_mention_pair_clusters = model_labels[b, output_dict['antecedent_indices'][b]]
            model_output_mention_pair_clusters[~coref_scores_mask[b, :, 1:]] = -1  # ensure -infty (impossible antecedents are in no cluster)
            # add 1st column of empties to match coreference_probs size
            model_output_mention_pair_clusters = torch.cat((-torch.ones(model_output_mention_pair_clusters.size(0),
                dtype=torch.long).cuda(self._cuda_devices[0]).unsqueeze(-1), model_output_mention_pair_clusters), dim=1)

            using_sum = True
            if using_sum:
                # select cluster corresponding to predicted antecedent for each mention
                predicted_antecedents = output_dict['predicted_antecedents'][b].unsqueeze(-1).expand_as(
                    model_output_mention_pair_clusters) + 1
                # all elements of each rows should be the same (since we replicated predicted_antecedents)
                antecedent_clusters = torch.gather(model_output_mention_pair_clusters, 1, predicted_antecedents)[:,0]

                # Get scores of mentions pointing to elements in clusters
                clustered_mask = antecedent_clusters != -1  # mask for mentions selected antecedents in clusters
                if self._selector == 'entropy':
                    mention_pair_cluster_mask = (model_output_mention_pair_clusters != -1)
                    if len(clustered_mask.nonzero()) > 0:
                        # get rows of those in selected clusters, add scores of each cluster
                        num_clusters = model_output_mention_pair_clusters.max() + 1
                        row_increment_range_vec = torch.arange(0, model_output_mention_pair_clusters.size(0) * num_clusters, num_clusters, dtype=torch.long).cuda(self._cuda_devices[0]).unsqueeze(1)
                        row_cluster_sum = (model_output_mention_pair_clusters + row_increment_range_vec)[mention_pair_cluster_mask].bincount(
                            coreference_probs[mention_pair_cluster_mask], minlength=row_increment_range_vec.max() + num_clusters)
                        row_cluster_sum = row_cluster_sum.view(-1, num_clusters)
                        # TODO VERIFY: for i, row in enumerate(row_cluster_sum): assert(len(coreference_probs[i][mention_pair_cluster_mask[i]]) == 0 or len(((row - model_output_mention_pair_clusters[i][mention_pair_cluster_mask[i]].bincount(coreference_probs[i][mention_pair_cluster_mask[i]], minlength=len(row))).abs() > 0.0001).nonzero()) == 0)
                        # add entropies of clusters
                        row_cluster_entropy = row_cluster_sum * row_cluster_sum.log()
                        row_cluster_entropy[row_cluster_entropy != row_cluster_entropy] = 0  # don't want to add nan caused by log-ing 0 probabilities
                        mention_confidence_scores[b] = -row_cluster_entropy.sum(1)
                    row_non_cluster_entropy = coreference_probs * coreference_probs.log()
                    row_non_cluster_entropy[mention_pair_cluster_mask] = 0  # don't add values in clusters (which we've already added)
                    row_non_cluster_entropy[row_non_cluster_entropy != row_non_cluster_entropy] = 0  # don't want to add nan caused by log-ing 0 probabilities
                    mention_confidence_scores[b] += -row_non_cluster_entropy.sum(1)
                elif self._selector == 'score':
                    if len(clustered_mask.nonzero()) > 0:
                        # mask for mentions belonging in chosen cluster
                        chosen_cluster_rows_mask = (model_output_mention_pair_clusters[clustered_mask] == antecedent_clusters[
                                                 clustered_mask].unsqueeze(-1).expand_as(model_output_mention_pair_clusters[clustered_mask]))
                        # get rows of those in selected clusters, add scores
                        mention_confidence_scores[b][clustered_mask] = (coreference_probs[clustered_mask] * chosen_cluster_rows_mask.float()).sum(1)
                    try:
                        assert(len(predicted_antecedents[~clustered_mask].nonzero()) == 0)
                    except:
                        pdb.set_trace()
                    mention_confidence_scores[b][~clustered_mask] = coreference_probs[~clustered_mask][:,0]
        '''
        if len(clustered_mask.nonzero()) > 0:
            torch.save(mention_confidence_scores, "mention_confidence_scores.txt")
            torch.save(coreference_probs, "coreference_probs.txt")
            torch.save(model_output_mention_pair_clusters, "model_output_mention_pair_clusters.txt")
            os.system("python verify_scorer.py")
            pdb.set_trace()
            self.DEBUG_BREAK_FLAG = False
        '''
        if self._selector == 'entropy':
            opt_score = mention_confidence_scores.max()
        elif self._selector == 'score':
            opt_score = mention_confidence_scores.min()
        # choose arbitrary unchosen, least-confident mention
        batch_and_mentions = ((mention_confidence_scores == opt_score) & ~queried_mentions_mask).nonzero()
        # check if edge belongs to 
        try:
            assert(len(batch_and_mentions) > 0)
        except:
            pdb.set_trace()
        # return least confident mention and associated score
        return batch_and_mentions[0], opt_score

    def _update_clusters_with_edge(self, span_labels, edge, delete=False, all_edges=None):
        '''
        delete: whether we are deleting or adding the edge to the clusters
        if delete true, all_edges must be true to allow us to know how to split the cluster
            also, all_edges can be before/after deletion of edge
        '''
        proform_label = span_labels[edge[0], edge[1]].item()
        antecedent_label = span_labels[edge[0], edge[2]].item()

        if delete:
            assert all_edges is not None
            if proform_label != -1 and proform_label == antecedent_label:
                # ONLY make changes after having affirmed both belong to same cluster
                # find edges involving spans in cluster
                spans_in_cluster_mask = (span_labels[edge[0]] == proform_label)
                spans_in_cluster = spans_in_cluster_mask.nonzero()
                edges_in_cluster_mask = ((all_edges[:,1] == spans_in_cluster) | (all_edges[:,2] == spans_in_cluster)).max(0)[0]
                edges_in_cluster = all_edges[edges_in_cluster_mask]

                # delete original cluster
                span_labels[edge[0]][spans_in_cluster_mask] = -1
                decrement_mask = -(span_labels[edge[0]] > proform_label).type(torch.long)
                span_labels[edge[0]] += decrement_mask

                # construct new clusters by adding in edges used to create original cluster
                for remaining_edge in edges_in_cluster:
                    if (remaining_edge != edge).sum() > 0:  # (don't add the edge we want to delete)
                        span_labels = self._update_clusters_with_edge(span_labels, remaining_edge)
            return span_labels

        # NOTE: Do not modify num_gold_clusters field in metadata, which is used to keep track of
        # the original, gold clusters
        if proform_label != -1 and antecedent_label != -1:
            # Case 0: both in clusters (merge clusters iff were newly created, non-gold clusters)
            if proform_label == antecedent_label:
                # If already in same clusters, no need to merge
                return span_labels
            # Otherwise, merge clusters: merge larger cluster id into smaller cluster id
            min_cluster_id = min(proform_label, antecedent_label)
            max_cluster_id = max(proform_label, antecedent_label)

            span_labels[edge[0]][span_labels[edge[0]] == max_cluster_id] = min_cluster_id
            # decrease by 1 the index of all clusters > removed max_cluster in span_labels
            decrement_mask = -(span_labels[edge[0]] > max_cluster_id).type(torch.long)
            span_labels[edge[0]] += decrement_mask
        elif antecedent_label != -1:
            # Case 1: antecedent in cluster, proform not (update proform's label,
            # add proform to cluster)
            span_labels[edge[0], edge[1]] = antecedent_label
        elif proform_label != -1:
            # Case 2: proform in cluster, antecedent not (update antecedent's label,
            # add antecedent to cluster)
            span_labels[edge[0], edge[2]] = proform_label
        else:
            # Case 3: neither in cluster (create new cluster with both)
            cluster_id = span_labels.max() + 1
            span_labels[edge[0], edge[2]] = cluster_id
            span_labels[edge[0], edge[1]] = cluster_id
        return span_labels

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter, validation_metric_per_epoch = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")

        self._enable_gradient_clipping()
        self._enable_activation_logging()

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()
        first_epoch_after_last_data_add = 0

        if self._do_active_learning:
            # save initial model state to retrain from scratch every iteration
            # TODO: have this specified by user, and make the directory when necessary
            init_model_path = os.path.join("active_learning_model_states", "init_model_state.th")
            init_optimizer_path = os.path.join("active_learning_model_states", "init_optimizer_state.th")
            if not os.path.exists("active_learning_model_states"):
                os.makedirs("active_learning_model_states")
                init_model_state = self.model.state_dict()
                init_optimizer_state = self.optimizer.state_dict()
                torch.save(init_model_state, init_model_path)
                torch.save(init_optimizer_state, init_optimizer_path)

        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)
            query_this_epoch = False

            if self._validation_data is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = self._get_metrics(val_loss, num_batches, reset=True)

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]

                    # Check validation metric to see if it's the best so far
                    is_best_so_far = self._is_best_so_far(this_epoch_val_metric, validation_metric_per_epoch)
                    validation_metric_per_epoch.append(this_epoch_val_metric)
                    if self._do_active_learning and len(self._held_out_train_data) > 0:
                        if self._should_stop_early(validation_metric_per_epoch[first_epoch_after_last_data_add:],
                                                   self._active_learning_patience):
                            # still have more data to add
                            query_this_epoch = True
                            logger.info("Ran out of patience.  Adding more data.")
                    else:
                        if self._should_stop_early(validation_metric_per_epoch[first_epoch_after_last_data_add:], self._patience):
                            logger.info("Ran out of patience.  Stopping training.")
                            break

            else:
                # No validation set, so just assume it's the best so far.
                is_best_so_far = True
                val_metrics = {}
                this_epoch_val_metric = None

            self._metrics_to_tensorboard(epoch, train_metrics, val_metrics=val_metrics)
            self._metrics_to_console(train_metrics, val_metrics)

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = time.strftime("%H:%M:%S", time.gmtime(training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if is_best_so_far:
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics['best_epoch'] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

            if self._serialization_dir:
                dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)

            if self._learning_rate_scheduler and (self._held_out_train_data is None or
                                                  len(self._held_out_train_data) == 0):
                # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
                # if it doesn't, the validation metric passed here is ignored.
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)

            self._save_checkpoint(epoch, validation_metric_per_epoch, is_best=is_best_so_far)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time)))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            # ''' ACTIVE LEARNING BY SELF-TRAINING/EM:
            # 1. evaluate on held-out training data
            # 2. use active learning/gold labels to confirm/deny labels on held-out training data
            # 3. add correct instances in held-out training data to actual train data, then re-train
            if self._do_active_learning and len(self._held_out_train_data) > 0 and (query_this_epoch or
                                             epoch - first_epoch_after_last_data_add >= self._active_learning_epoch_interval):
                # take a subset of training data to evaluate on, and add to actual training set
                # TODO: currently arbitrarily choosing next 1 instance (by order in file), perhaps change this future(?)
                train_data_to_add = self._held_out_train_data[:280]
                self._held_out_train_data = self._held_out_train_data[280:]
                held_out_generator = self._held_out_iterator(train_data_to_add, num_epochs=1, shuffle=False)
                num_held_out_batches = self.iterator.get_num_batches(train_data_to_add)
                held_out_generator_tqdm = Tqdm.tqdm(held_out_generator, total=num_held_out_batches)
                conll_coref = ConllCorefScores()
                total_labels = 0
                total_num_queried = 0

                if not self._percent_label_experiments:
                    with torch.no_grad():
                        logger.info("Held-Out Training")
                        # Run model on held out training data
                        self.model.eval()

                        num_batches = 0
                        held_out_loss = 0
                        for batch_ind, batch in enumerate(held_out_generator_tqdm):
                            batch['get_scores'] = True
                            if self._multiple_gpu:
                                output_dict = self._data_parallel(batch)
                            else:
                                batch = util.move_to_device(batch, self._cuda_devices[0])
                                output_dict = self.model(**batch)

                            batch_size = len(output_dict['predicted_antecedents'])

                            #BOOKMARK
                            if self._query_type == 'discrete' and self._selector_clusters:
                                # Create reference for translation to and from indA
                                spans_in_text = torch.tensor([len(batch['metadata'][i]['original_text']) for i in range(len(batch['metadata']))],
                                    dtype=torch.long).cuda(self._cuda_devices[0])
                                # translate all spans and top_spans to unique integer keys
                                all_spans_keys = batch['spans'][:,:,0] * spans_in_text + batch['spans'][:,:,1]
                                top_spans_keys = output_dict['top_spans'][:,:,0] * spans_in_text + output_dict['top_spans'][:,:,1]
                                top_span_to_all_span_inds = -torch.ones(output_dict['top_spans'].size()[:2], dtype=torch.long).cuda(self._cuda_devices[0])
                                for b in range(len(top_spans_keys)):
                                    for i, span_key in enumerate(top_spans_keys[b]):
                                        top_span_to_all_span_inds[b,i] = (all_spans_keys[b] == span_key).nonzero()

                                # history of mentions that have already been queried/exist in gold data (index in top_spans)
                                all_queried_mentions = (batch['span_labels'] != -1).nonzero()
                                queried_mentions_mask = torch.zeros(output_dict['coreference_scores'].size()[:2],
                                    dtype=torch.uint8).cuda(self._cuda_devices[0])  # should be all false
                                # convert ot indices of top_spans for consistency's sake
                                if len(all_queried_mentions) > 0:
                                    all_queried_mentions_spans = batch['spans'][all_queried_mentions[:,0], all_queried_mentions[:,1]]
                                    top_queried_mentions_spans = ((all_queried_mentions_spans.unsqueeze(1) - output_dict['top_spans']).abs().sum(-1) == 0).nonzero()
                                    batch_inds = all_queried_mentions[top_queried_mentions_spans[:,0]][:,0]
                                    # ASSUMES 1 INSTANCE/BATCH
                                    queried_mentions_mask[batch_inds, top_queried_mentions_spans[:,1]] = 1

                                confirmed_clusters = batch['span_labels'].clone()
                                # TODO: fix if batch['span_labels'] is not all -1
                                confirmed_non_coref_edges = torch.tensor([], dtype=torch.long).cuda(self._cuda_devices[0])

                                # Update span_labels with model-predicted clusters
                                output_dict = self.model.decode(output_dict)
                                has_antecedent_mask = (output_dict['predicted_antecedents'] != -1)
                                model_edges = torch.empty(0, dtype=torch.long).cuda(self._cuda_devices[0])
                                if len(has_antecedent_mask.nonzero()) > 0:
                                    model_edges = torch.cat((has_antecedent_mask.nonzero(), output_dict['predicted_antecedents'][has_antecedent_mask].unsqueeze(-1)), dim=-1)
                                indA_model_edges = self._translate_to_indA(model_edges, output_dict, batch['spans'], translation_reference=top_span_to_all_span_inds)
                                for edge in indA_model_edges:
                                    batch['span_labels'] = self._update_clusters_with_edge(batch['span_labels'], edge)

                                if self._use_percent_labels:
                                    # upper bound is asking question about every span
                                    num_to_query = int(self._active_learning_percent_labels * len(output_dict['top_spans'][0]))
                                    total_possible_queries = len(output_dict['top_spans'][0])
                                else:
                                    total_possible_queries = len(output_dict['top_spans'][0])
                                    num_to_query = min(self._active_learning_num_labels, total_possible_queries)
                                num_queried = 0
                                while num_queried < num_to_query:
                                    top_spans_model_labels = torch.gather(batch['span_labels'], 1, top_span_to_all_span_inds)
                                    mention, mention_score = self._find_next_most_uncertain_mention(top_spans_model_labels, output_dict, queried_mentions_mask)
                                    indA_edge, edge_asked, indA_edge_asked = self._query_user_labels_mention(mention, output_dict, batch['spans'], batch['user_labels'], translation_reference=top_span_to_all_span_inds)
                                    # add mention to queried before (arbitrarily set it in predicted_antecedents and coreference_scores to no cluster, even if not truly
                                    # the case--the only thing that matters is that it has a value that it is 100% confident of)
                                    queried_mentions_mask[mention[0], mention[1]] = 1
                                    # arbitrarily set to 1st span (most of time 1st span in cluster isn't even in range of current mention)
                                    output_dict['predicted_antecedents'][mention[0], mention[1]] = -1
                                    output_dict['coreference_scores'][mention[0], mention[1], 1:] = -float("inf")

                                    # If asked edge was deemed not coreferent, delete it
                                    if indA_edge_asked[2] != indA_edge[2] and len(indA_model_edges) > 0:
                                        # (both lines below implicitly check whether indA_edge_asked was actually added before)
                                        edge_asked_mask = (indA_model_edges == indA_edge_asked).sum(1)
                                        batch['span_labels'] = self._update_clusters_with_edge(batch['span_labels'], indA_edge_asked, delete=True, all_edges=indA_model_edges)
                                        indA_model_edges = indA_model_edges[edge_asked_mask < 3]
                                        # Add to confirmed non-coreferent
                                        if len(confirmed_non_coref_edges) == 0:
                                            confirmed_non_coref_edges = indA_edge_asked.unsqueeze(0)
                                        else:
                                            confirmed_non_coref_edges = torch.cat((confirmed_non_coref_edges, indA_edge_asked.unsqueeze(0)), dim=0)
                                        # Do pruning
                                        cluster_ant = confirmed_clusters[indA_edge_asked[0], indA_edge_asked[2]]
                                        cluster_pro = confirmed_clusters[indA_edge_asked[0], indA_edge_asked[1]]
                                        #pdb.set_trace()

                                    # Add edge deemed coreferent
                                    if indA_edge[2] != -1:
                                        # Add new edge deemed coreferent, if not already in there
                                        if len(indA_model_edges) == 0 or ((indA_model_edges == indA_edge).sum(1) == 3).sum() == 0:
                                            indA_model_edges = torch.cat((indA_model_edges, indA_edge.unsqueeze(0)), dim=0)
                                            batch['span_labels'] = self._update_clusters_with_edge(batch['span_labels'], indA_edge)
                                        confirmed_clusters = self._update_clusters_with_edge(confirmed_clusters, indA_edge)
                                        # Do pruning
                                        #pdb.set_trace()

                                    num_queried += 1

                                self.DEBUG_BREAK_FLAG = False
                                edges_to_add = indA_model_edges

                            elif self._query_type == 'discrete': # selector is random or score
                                # get rid of 1st, dummy column to ensure nothing selected from it
                                coref_scores_no_dummy = output_dict['coreference_scores'][:,:,1:]
                                max_mention_scores, _ = coref_scores_no_dummy.max(2, keepdim=True)
                                max_mention_scores_mask = coref_scores_no_dummy.eq(max_mention_scores)

                                # BOTH-ENDS MASK-MAKING CODE: ensures we don't select edges for which both mentions are already in clusters
                                spans_in_clusters = batch['spans'][batch['span_labels'] > 0]
                                both_ends_in_clusters_mask = torch.zeros(max_mention_scores_mask.size(), dtype=torch.uint8).cuda(self._cuda_devices[0])
                                if len(spans_in_clusters) > 0:
                                    indB_spans_in_clusters = ((spans_in_clusters.unsqueeze(1).unsqueeze(1) -
                                                               output_dict['top_spans'].unsqueeze(0)).abs().sum(-1) == 0).nonzero()[:, 1:]
                                    # if [0,a] and [0,b] both in indB_spans_in_clusters, mask['antecedent_indices'][0,a,i] =
                                    # mask['antecedent_indices'][0,b,j] = 1 (both ends in gold clusters), whereby output_dict['antecedent_indices'][0,a,i] = b, and
                                    # output_dict['antecedent_indices'][0,b,j] = a
                                    # TODO: this code ONLY WORKS for batchsize = 1
                                    in_cluster_mention_rows = output_dict['antecedent_indices'][indB_spans_in_clusters[:,0], indB_spans_in_clusters[:,1]] # rows to search within
                                    # inds where some element of indB_spans_in_clusters is present in in_cluster_mention_rows
                                    both_ends_in_clusters_mask[indB_spans_in_clusters[:,0], indB_spans_in_clusters[:,1]] = (
                                        (in_cluster_mention_rows.unsqueeze(-1) - indB_spans_in_clusters[:,1].unsqueeze(0).unsqueeze(0)).abs() == 0).sum(-1) != 0

                                max_mention_scores_mask &= ~both_ends_in_clusters_mask  # exclude edges w/ both ends in gold clusters
                                # add back dummy column to fit to shape of output_dict['coreference_scores']
                                max_mention_scores_mask = torch.cat([torch.zeros(
                                    batch_size, coref_scores_no_dummy.size(1), 1, dtype=torch.uint8
                                    ).cuda(self._cuda_devices[0]), max_mention_scores_mask], dim=-1)

                                sorted_max_mention_edges, sorted_max_mention_edges_score = \
                                    self._get_sorted_masked_edges(max_mention_scores_mask, output_dict, batch['spans'], farthest_from_zero=False)

                                if self._use_percent_labels:
                                    num_to_query = int(self._active_learning_percent_labels * len(sorted_max_mention_edges))
                                else:
                                    num_to_query = self._active_learning_num_labels

                                edges_to_add, num_to_query, total_possible_queries, batch['span_labels'] = \
                                    self._query_user_labels_discrete(sorted_max_mention_edges, sorted_max_mention_edges_score, num_to_query,
                                                                     True, output_dict, batch)
                            else: # query type is pairwise
                                # get all > 0 edges (to know which to assign next)
                                larger_than_zero_mask = (output_dict['coreference_scores'] > 0)
                                sorted_larger_than_zero_edges, larger_than_zero_scores = \
                                    self._get_sorted_masked_edges(larger_than_zero_mask, output_dict, batch['spans'], farthest_from_zero=True)

                                # get scores of edges, and check most uncertain subset of edges
                                predicted_scores, _ = output_dict['coreference_scores'].max(2, keepdim=True)
                                predicted_scores_mask = output_dict['coreference_scores'].eq(predicted_scores)
                                predicted_scores_mask[:,:,0] = 0  # get rid of negative-scoring edges (ones for which "max edge" is predicted to be w/ null)

                                # get mask of scores of all possible edges originating from nodes not predicted to have any proform
                                neg_edge_inds_mask = (output_dict['coreference_scores'] < 0) & \
                                                     (output_dict['coreference_scores'] != -float("inf"))

                                chosen_edges_mask = (neg_edge_inds_mask + predicted_scores_mask) > 0
                                edges_to_add, edge_scores = self._get_sorted_masked_edges(chosen_edges_mask, output_dict,
                                                                                          batch['spans'], farthest_from_zero=False)
                                edges_to_add = self._filter_gold_cluster_edges(edges_to_add, batch['span_labels'])

                                if self._use_percent_labels:
                                    num_to_query = int(self._active_learning_percent_labels * len(edges_to_add))
                                    num_alts_to_check = int(self._active_learning_percent_labels * len(sorted_larger_than_zero_edges))
                                else:
                                    percent_to_alt = float(len(sorted_larger_than_zero_edges)) / float(len(edges_to_add))
                                    num_alts_to_check = int(percent_to_alt * self._active_learning_num_labels)
                                    num_to_query = self._active_learning_num_labels - num_alts_to_check
                                edges_to_add, num_to_query, total_possible_queries = \
                                    self._query_user_labels_pairwise(edges_to_add, edge_scores, batch['user_labels'], num_to_query,
                                                                     True, self._replace_with_next_pos_edge,
                                                                     sorted_larger_than_zero_edges, num_alts_to_check, larger_than_zero_scores)

                            # keep track of which instances we have to update in training data
                            train_instances_to_update = {}
                            # Update gold clusters based on (corrected) model edges, in span_labels
                            for edge in edges_to_add:
                                if self._query_type != 'discrete':
                                    batch['span_labels'] = self._update_clusters_with_edge(batch['span_labels'], edge)
                                ind_instance = edge[0].item()  # index of instance in batch
                                if ind_instance not in train_instances_to_update:
                                    train_instances_to_update[ind_instance] = 0
                                train_instances_to_update[ind_instance] += 1

                            # update train data itself
                            for ind_instance in train_instances_to_update:
                                ind_instance_overall = batch_ind * batch_size + ind_instance  # index in entire train data
                                train_data_to_add[ind_instance_overall].fields['span_labels'] = SequenceLabelField(
                                    batch['span_labels'][ind_instance].tolist(),
                                    train_data_to_add[ind_instance_overall].fields['span_labels'].sequence_field
                                )

                            if output_dict['loss'] is not None:
                                num_batches += 1
                                held_out_loss += output_dict['loss'].detach().cpu().numpy()

                            # Update the description with the latest metrics
                            held_out_metrics = self._get_metrics(held_out_loss, num_batches)
                            for i, metadata in enumerate(batch['metadata']):
                                predicted_clusters = []
                                for cluster in range(batch['span_labels'][i].max() + 1):
                                    # convert spans to tuples
                                    predicted_clusters.append(batch['spans'][i][batch['span_labels'][i] == cluster].tolist())
                                predicted_clusters, mention_to_predicted = conll_coref.get_gold_clusters(predicted_clusters)
                                gold_clusters, mention_to_gold = conll_coref.get_gold_clusters(batch['metadata'][i]['clusters'])
                                if self.DEBUG_BREAK_FLAG:
                                    import pickle
                                    pickle.dump(predicted_clusters, open('predicted_clusters.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                                    pickle.dump(gold_clusters, open('gold_clusters.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                                    pickle.dump(mention_to_predicted, open('mention_to_predicted.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                                    pickle.dump(mention_to_gold, open('mention_to_gold.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                                    os.system("python verify_clusters.py")
                                    pdb.set_trace()
                                    #for span in spans: print(str(span) + " " + str(((output_dict['top_spans'][:,:,0] == span[0]) & (output_dict['top_spans'][:,:,1] == span[1])).nonzero()))
                                for scorer in conll_coref.scorers:
                                    scorer.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
                            new_P, new_R, new_F1 = conll_coref.get_metric()
                            description_display = {'old_P': held_out_metrics['coref_precision'], 'new_P': new_P,
                                                   'old_R': held_out_metrics['coref_recall'], 'new_R': new_R,
                                                   'old_F1': held_out_metrics['coref_f1'], 'new_F1': new_F1,
                                                   'MR': held_out_metrics['mention_recall'], 'loss': held_out_metrics['loss']}
                            description = self._description_from_metrics(description_display)
                            total_num_queried += num_to_query
                            total_labels += total_possible_queries
                            description += ' # labels: ' + str(total_num_queried) + '/' + str(total_labels) + ' ||'
                            held_out_generator_tqdm.set_description(description, refresh=False)
                else:
                    for batch_ind, batch in enumerate(held_out_generator_tqdm):
                        for i, metadata in enumerate(batch['metadata']):
                            # eliminate singletons
                            user_clusters_mask = batch['user_labels'][i] >= 0
                            if len(user_clusters_mask.nonzero()) > 0:
                                singleton_clusters = (batch['user_labels'][i][user_clusters_mask].bincount() == 1).nonzero().squeeze(-1)
                            else:
                                singleton_clusters = user_clusters_mask.nonzero()
                            for cluster in singleton_clusters:
                                user_cluster_idx = (batch['user_labels'][i] == cluster).nonzero()
                                batch['user_labels'][i, user_cluster_idx] = -1
                            # labelled in user_labels to not in span_labels
                            possible_mentions_idx = ((batch['user_labels'][i] >= 0) & (batch['span_labels'][i] == -1)).nonzero().squeeze(-1)
                            num_labels = len(possible_mentions_idx)
                            num_labels_to_pick = int(self._percent_labels * num_labels)
                            ''' Try to avoid singletons 
                            # uniformly choose in possible_labels_idx, without replacement
                            chosen_labels_idx = possible_labels_idx[
                                torch.multinomial(torch.ones(len(possible_labels_idx)), num_labels_to_pick, replacement=False)
                            ]
                            batch['span_labels'][i, chosen_labels_idx] = batch['user_labels'][i, chosen_labels_idx]
                            '''
                            seen_clusters = {}
                            labels = 0
                            while labels < num_labels_to_pick:
                                chosen_cluster_mention = possible_mentions_idx[torch.randint(len(possible_mentions_idx), (), dtype=torch.long)]
                                chosen_cluster = batch['user_labels'][i][chosen_cluster_mention].item()
                                if chosen_cluster not in seen_clusters:
                                    possible_cluster_mentions = ((batch['user_labels'][i] == chosen_cluster) & (
                                                batch['span_labels'][i] != chosen_cluster)).nonzero().squeeze(-1)
                                    num_mentions_to_pick = min(2, len(possible_cluster_mentions))
                                else:
                                    num_mentions_to_pick = 1
                                    possible_cluster_mentions = seen_clusters[chosen_cluster]
                                for m in range(num_mentions_to_pick):
                                    # possible_cluster_mentions can't be empty at this point
                                    chosen_mention = possible_cluster_mentions[torch.randint(len(possible_cluster_mentions), (), dtype=torch.long)]
                                    possible_cluster_mentions = possible_cluster_mentions[possible_cluster_mentions != chosen_mention]
                                    batch['span_labels'][i, chosen_mention] = chosen_cluster
                                    labels += 1
                                    possible_mentions_idx = possible_mentions_idx[possible_mentions_idx != chosen_mention]
                                seen_clusters[chosen_cluster] = possible_cluster_mentions
                            # add to train_data_to_add
                            ind_instance_overall = batch_ind * len(batch['metadata']) + i  # index in entire train data
                            train_data_to_add[ind_instance_overall].fields['span_labels'] = SequenceLabelField(
                                batch['span_labels'][i].tolist(),
                                train_data_to_add[ind_instance_overall].fields['span_labels'].sequence_field
                            )

                            # create & evaluate clusters
                            chosen_clusters = []
                            for cluster in range(max(batch['span_labels'][i]) + 1):
                                if len((batch['span_labels'][i] == cluster).nonzero()) <= 0:
                                    continue
                                # convert spans to tuples
                                chosen_clusters.append(batch['spans'][i][batch['span_labels'][i] == cluster].tolist())
                            chosen_clusters, mention_to_predicted = conll_coref.get_gold_clusters(chosen_clusters)
                            gold_clusters, mention_to_gold = conll_coref.get_gold_clusters(batch['metadata'][i]['clusters'])
                            for scorer in conll_coref.scorers:
                                scorer.update(chosen_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
                        new_P, new_R, new_F1 = conll_coref.get_metric()
                        description_display = {'coref_precision': new_P, 'coref_recall': new_R, 'coref_f1': new_F1}
                        description = self._description_from_metrics(description_display)
                        total_num_queried += num_labels_to_pick
                        total_labels += num_labels
                        description += ' # chosen labels: ' + str(total_num_queried) + ', total labels: ' + str(total_labels) + ' ||'
                        held_out_generator_tqdm.set_description(description, refresh=False)

                # add instance(s) from held-out training dataset to actual dataset (already removed from held-out
                # above)
                self.train_data.extend(train_data_to_add)

                first_epoch_after_last_data_add = epoch + 1

                # at last epoch, retrain from scratch, resetting model params to intial state
                if len(self._held_out_train_data) == 0:
                    init_model_state = torch.load(init_model_path, map_location=util.device_mapping(-1))
                    init_optimizer_state = torch.load(init_optimizer_path, map_location=util.device_mapping(-1))
                    self.model.load_state_dict(init_model_state)
                    self.optimizer.load_state_dict(init_optimizer_state)
                    move_optimizer_to_cuda(self.optimizer)

            epochs_trained += 1

        return metrics

    def _is_best_so_far(self,
                        this_epoch_val_metric: float,
                        validation_metric_per_epoch: List[float]):
        if not validation_metric_per_epoch:
            return True
        elif self._validation_metric_decreases:
            return this_epoch_val_metric < min(validation_metric_per_epoch)
        else:
            return this_epoch_val_metric > max(validation_metric_per_epoch)

    def _description_from_metrics(self, metrics: Dict[str, float]) -> str:
        if (not self._warned_tqdm_ignores_underscores and
                    any(metric_name.startswith("_") for metric_name in metrics)):
            logger.warning("Metrics with names beginning with \"_\" will "
                           "not be logged to the tqdm progress bar.")
            self._warned_tqdm_ignores_underscores = True
        return ', '.join(["%s: %.4f" % (name, value) for name, value in
                          metrics.items() if not name.startswith("_")]) + " ||"

    def _save_checkpoint(self,
                         epoch: Union[int, str],
                         val_metric_per_epoch: List[float],
                         is_best: Optional[bool] = None) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        is_best: bool, optional (default = None)
            A flag which causes the model weights at the given epoch to
            be copied to a "best.th" file. The value of this flag should
            be based on some validation metric computed by your model.
        """
        if self._serialization_dir is not None:
            model_path = os.path.join(self._serialization_dir, "model_state_epoch_{}.th".format(epoch))
            model_state = self.model.state_dict()
            torch.save(model_state, model_path)

            training_state = {'epoch': epoch,
                              'val_metric_per_epoch': val_metric_per_epoch,
                              'optimizer': self.optimizer.state_dict(),
                              'batch_num_total': self._batch_num_total}
            if self._learning_rate_scheduler is not None:
                training_state["learning_rate_scheduler"] = \
                    self._learning_rate_scheduler.lr_scheduler.state_dict()
            training_path = os.path.join(self._serialization_dir,
                                         "training_state_epoch_{}.th".format(epoch))
            torch.save(training_state, training_path)
            if is_best:
                logger.info("Best validation performance so far. "
                            "Copying weights to '%s/best.th'.", self._serialization_dir)
                shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best.th"))

            if self._num_serialized_models_to_keep and self._num_serialized_models_to_keep >= 0:
                self._serialized_paths.append([time.time(), model_path, training_path])
                if len(self._serialized_paths) > self._num_serialized_models_to_keep:
                    paths_to_remove = self._serialized_paths.pop(0)
                    # Check to see if we should keep this checkpoint, if it has been longer
                    # then self._keep_serialized_model_every_num_seconds since the last
                    # kept checkpoint.
                    remove_path = True
                    if self._keep_serialized_model_every_num_seconds is not None:
                        save_time = paths_to_remove[0]
                        time_since_checkpoint_kept = save_time - self._last_permanent_saved_checkpoint_time
                        if time_since_checkpoint_kept > self._keep_serialized_model_every_num_seconds:
                            # We want to keep this checkpoint.
                            remove_path = False
                            self._last_permanent_saved_checkpoint_time = save_time
                    if remove_path:
                        for fname in paths_to_remove[1:]:
                            os.remove(fname)

    def find_latest_checkpoint(self) -> Tuple[str, str]:
        """
        Return the location of the latest model and training state files.
        If there isn't a valid checkpoint then return None.
        """
        have_checkpoint = (self._serialization_dir is not None and
                           any("model_state_epoch_" in x for x in os.listdir(self._serialization_dir)))

        if not have_checkpoint:
            return None

        serialization_files = os.listdir(self._serialization_dir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        # Get the last checkpoint file.  Epochs are specified as either an
        # int (for end of epoch files) or with epoch and timestamp for
        # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
        found_epochs = [
                # pylint: disable=anomalous-backslash-in-string
                re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1)
                for x in model_checkpoints
        ]
        int_epochs: Any = []
        for epoch in found_epochs:
            pieces = epoch.split('.')
            if len(pieces) == 1:
                # Just a single epoch without timestamp
                int_epochs.append([int(pieces[0]), 0])
            else:
                # has a timestamp
                int_epochs.append([int(pieces[0]), pieces[1]])
        last_epoch = sorted(int_epochs, reverse=True)[0]
        if last_epoch[1] == 0:
            epoch_to_load = str(last_epoch[0])
        else:
            epoch_to_load = '{0}.{1}'.format(last_epoch[0], last_epoch[1])

        model_path = os.path.join(self._serialization_dir,
                                  "model_state_epoch_{}.th".format(epoch_to_load))
        training_state_path = os.path.join(self._serialization_dir,
                                           "training_state_epoch_{}.th".format(epoch_to_load))

        return (model_path, training_state_path)

    def _restore_checkpoint(self) -> Tuple[int, List[float]]:
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from  model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        Returns
        -------
        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        latest_checkpoint = self.find_latest_checkpoint()

        if latest_checkpoint is None:
            # No checkpoint to restore, start at 0
            return 0, []

        model_path, training_state_path = latest_checkpoint

        # Load the parameters onto CPU, then transfer to GPU.
        # This avoids potential OOM on GPU for large models that
        # load parameters onto GPU then make a new GPU copy into the parameter
        # buffer. The GPU transfer happens implicitly in load_state_dict.
        model_state = torch.load(model_path, map_location=util.device_mapping(-1))
        training_state = torch.load(training_state_path, map_location=util.device_mapping(-1))
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self._learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
            self._learning_rate_scheduler.lr_scheduler.load_state_dict(
                    training_state["learning_rate_scheduler"])
        move_optimizer_to_cuda(self.optimizer)

        # We didn't used to save `validation_metric_per_epoch`, so we can't assume
        # that it's part of the trainer state. If it's not there, an empty list is all
        # we can do.
        if "val_metric_per_epoch" not in training_state:
            logger.warning("trainer state `val_metric_per_epoch` not found, using empty list")
            val_metric_per_epoch: List[float] = []
        else:
            val_metric_per_epoch = training_state["val_metric_per_epoch"]

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get('batch_num_total')
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return, val_metric_per_epoch

    # Requires custom from_params.
    @classmethod
    def from_params(cls,  # type: ignore
                    model: Model,
                    serialization_dir: str,
                    iterator: DataIterator,
                    train_data: Iterable[Instance],
                    validation_data: Optional[Iterable[Instance]],
                    params: Params,
                    validation_iterator: DataIterator = None,
                    held_out_train_data: Optional[Iterable[Instance]] = None,
                    held_out_iterator: DataIterator = None) -> 'Trainer':
        # pylint: disable=arguments-differ
        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        if lr_scheduler_params:
            scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            scheduler = None

        num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
        keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)

        active_learning = params.pop("active_learning", None)

        params.assert_empty(cls.__name__)

        return cls(model, optimizer, iterator,
                   train_data, held_out_train_data, validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   held_out_iterator=held_out_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=scheduler,
                   num_serialized_models_to_keep=num_serialized_models_to_keep,
                   keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                   model_save_interval=model_save_interval,
                   summary_interval=summary_interval,
                   histogram_interval=histogram_interval,
                   should_log_parameter_statistics=should_log_parameter_statistics,
                   should_log_learning_rate=should_log_learning_rate,
                   active_learning=active_learning)


Trainer.register("default")(Trainer)
