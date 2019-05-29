import pdb
import torch
from allennlp.nn import util
import os


def translate_to_indA(edges, output_dict, all_spans, translation_reference=None) -> torch.LongTensor:
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
        indA_proforms = torch.empty(instances.size(), dtype=torch.long, device=proform_spans.device)
        indA_antecedents = torch.empty(instances.size(), dtype=torch.long, device=proform_spans.device)
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


def filter_gold_cluster_edges(chosen_edges, span_labels):
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


def update_clusters_with_edge(span_labels, edge, delete=False, all_edges=None):
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
                    span_labels = update_clusters_with_edge(span_labels, remaining_edge)
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


def get_sorted_masked_edges(selector, coreference_mask, output_dict, all_spans, translation_reference,
                            farthest_from_zero=False) -> torch.LongTensor:
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
    if selector == 'score':
        # get sorted closest/furthest from 0 scores
        _, ind_max_edge_scores = edge_scores.abs().sort(descending=farthest_from_zero)
    elif selector == 'random':
        # using random selector
        ind_max_edge_scores = torch.randperm(len(masked_edge_inds))
    else:  # selector is entropy
        coreference_probs = torch.zeros(output_dict['coreference_scores'].size(), dtype=torch.float,
                                        device=edge_scores.device)
        for b in range(len(output_dict['top_spans'])):
            top_span_mask = torch.ones(output_dict['top_spans'][b].size(0), dtype=torch.float, device=edge_scores.device
                                       ).unsqueeze(-1)
            coreference_probs[b] = util.masked_log_softmax(output_dict['coreference_scores'][b], top_span_mask)
        mentions_to_query = coreference_mask.max(-1)[0].nonzero()
        mention_entropies = coreference_probs.exp() * coreference_probs
        # delete nan caused by log-ing 0s
        mention_entropies[mention_entropies != mention_entropies] = 0
        mention_entropies = mention_entropies[mentions_to_query[:, 1]]
        edge_entropies = -mention_entropies.sum(-1)
        _, ind_max_edge_entropies = edge_entropies.sort(descending=True)
    sorted_edges = translate_to_indA(masked_edge_inds[ind_max_edge_scores], output_dict, all_spans, translation_reference)
    return sorted_edges, edge_scores[ind_max_edge_scores]


def query_user_labels_pairwise(sample_from_training, chosen_edges, edge_scores, user_labels, num_labels_to_query,
                               return_all_edges, use_alt_edges=False, all_candidate_alt_edges=None, num_alts_to_check=0,
                               alt_edge_scores=None):
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
    total_possible_queries = len(chosen_edges) + len(
        all_candidate_alt_edges) if all_candidate_alt_edges is not None else len(chosen_edges)
    pos_edges_mask = (edge_scores[:num_labels_to_query] > 0)
    num_labels_to_query = len(chosen_edges[:num_labels_to_query])
    num_alt_edge_queried = 0
    if num_labels_to_query > 0:
        if sample_from_training:
            proform_user_labels = user_labels[
                chosen_edges[:num_labels_to_query][:, 0], chosen_edges[:num_labels_to_query][:, 1]]
            antecedent_user_labels = user_labels[
                chosen_edges[:num_labels_to_query][:, 0], chosen_edges[:num_labels_to_query][:, 2]]
            coreferent_mask = (proform_user_labels == antecedent_user_labels) & (proform_user_labels != -1)
            # ensure all edges deemed coreferent are assigned positive scores
            edge_scores[:num_labels_to_query][coreferent_mask] = edge_scores[:num_labels_to_query][
                coreferent_mask].abs()
            non_coreferent_pos_edges = chosen_edges[:num_labels_to_query][~coreferent_mask & pos_edges_mask]
            if len(non_coreferent_pos_edges) > 0 and not use_alt_edges:
                # set all non-coreferent edges to -1
                chosen_edges[:num_labels_to_query][~coreferent_mask & pos_edges_mask] = -1
            elif len(non_coreferent_pos_edges) > 0:
                # use alternate edges
                assert all_candidate_alt_edges is not None
                assert alt_edge_scores is not None
                # replace non-coreferent and (+) edges with alternate edges (same proform, next largest antecedent)
                allalt_differences = (
                            non_coreferent_pos_edges.unsqueeze(0) - all_candidate_alt_edges.unsqueeze(1)).abs()
                same_proform_diff_antecedent_mask = (allalt_differences[:, :, 1] == 0) & (
                        allalt_differences[:, :, 1] == 0) & (
                                                            allalt_differences[:, :, 2] != 0)
                # [inds of possible alternates in all_candidate_alt_edges, inds of edges to replace in non_coreferent_pos_edges]
                possible_alt_inds_to_query = same_proform_diff_antecedent_mask.nonzero()[:num_alts_to_check]
                num_alt_edge_queried += len(possible_alt_inds_to_query)
                alternate_pos_edges = -torch.ones(non_coreferent_pos_edges.size(), dtype=torch.long,
                                                  device=edge_scores.device)
                chosen_alternate_edge_scores = -torch.ones(non_coreferent_pos_edges.size(0), dtype=torch.float,
                                                           device=edge_scores.device)
                if possible_alt_inds_to_query.size(0) > 0:
                    possible_alts_to_query = all_candidate_alt_edges[possible_alt_inds_to_query[:, 0]]
                    alt_proforms = user_labels[possible_alts_to_query[:, 0], possible_alts_to_query[:, 1]]
                    alt_antecedents = user_labels[possible_alts_to_query[:, 0], possible_alts_to_query[:, 2]]
                    coreferent_alts_mask = (alt_proforms >= 0) & (alt_proforms == alt_antecedents)
                    # flip since for the same proform, want to set to highest-scoring coreferent antecedent, which
                    # means want highest-scoring at the end
                    coreferent_alt_inds = possible_alt_inds_to_query[coreferent_alts_mask].flip(0)
                    if len(coreferent_alt_inds) > 0:
                        alternate_pos_edges[coreferent_alt_inds[:, 1]] = all_candidate_alt_edges[
                            coreferent_alt_inds[:, 0]]
                        # also set new edge score
                        chosen_alternate_edge_scores[coreferent_alt_inds[:, 1]] = alt_edge_scores[
                            coreferent_alt_inds[:, 0]]
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


def query_user_labels_discrete(chosen_edges, edge_scores, num_labels_to_query, return_all_edges, output_dict, batch):
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
        # use for loop, as may want to change things depending on which edge we're currently querying, also a few
        # hundred mentions = not too expensive
        while i < len(chosen_edges):
            if num_labels_queried >= num_labels_to_query:
                break
            proform_span_label = span_labels[chosen_edges[i, 0], chosen_edges[i, 1]]
            antecedent_span_label = span_labels[chosen_edges[i, 0], chosen_edges[i, 2]]
            if proform_span_label != -1 and proform_span_label == antecedent_span_label:
                i += 1
                continue
            proform_user_label = user_labels[chosen_edges[i][0], chosen_edges[i, 1]]
            antecedent_user_label = user_labels[chosen_edges[i, 0], chosen_edges[i, 2]]
            num_labels_queried += 1
            # ask if edge is coreferent
            coreferent = (proform_user_label == antecedent_user_label) & (proform_user_label != -1)
            if not coreferent and proform_user_label != -1:
                # query a coreferent spans from user
                user_spans_in_proform_cluster_mask = user_labels[chosen_edges[i, 0]] == proform_user_label
                user_spans_in_proform_cluster = user_spans_in_proform_cluster_mask.nonzero()
                if len(user_spans_in_proform_cluster) > 1:  # more than 1 element in cluster
                    # if we are 1st span in cluster, choose arbitrary span outside of spans already in our cluster
                    if chosen_edges[i, 1] == user_spans_in_proform_cluster[0]:
                        spans_in_curr_proform_cluster_mask = (proform_span_label != -1) & (
                                    span_labels[chosen_edges[i, 0]] == proform_span_label)
                        user_spans_outside_curr_proform_cluster = (user_spans_in_proform_cluster_mask &
                                                                   ~spans_in_curr_proform_cluster_mask).nonzero()
                        user_spans_outside_curr_proform_cluster = user_spans_outside_curr_proform_cluster[
                            user_spans_outside_curr_proform_cluster != chosen_edges[i, 1]]
                        # TODO: if there are spans in the cluster besides what we have currently, choose random among them
                        # otherwise, no additional spans to add, delete edge (and all ingoing and outgoing edges from cluster)
                        if len(user_spans_outside_curr_proform_cluster.nonzero()) > 0:
                            new_antecedent = user_spans_outside_curr_proform_cluster[
                                torch.randint(len(user_spans_outside_curr_proform_cluster), (), dtype=torch.long)]
                            chosen_edges[i, 2] = chosen_edges[i, 1]
                            chosen_edges[i, 1] = new_antecedent

                            # redirect future edges out of new_antecedent -> S to go from S -> first span
                            outgoing_new_antecedent_span_mask = (chosen_edges[:, 1] == new_antecedent)
                            outgoing_new_antecedent_span_mask[:i + 1] = 0  # ensure only setting future edges
                            chosen_edges[:, 1][outgoing_new_antecedent_span_mask] = chosen_edges[:, 2][
                                outgoing_new_antecedent_span_mask]
                            chosen_edges[:, 2][outgoing_new_antecedent_span_mask] = chosen_edges[
                                i, 2]  # point to first span of cluster (the same as chosen_edges[i] is pointing to)
                            # delete future edges that are now invalid (because impossible to be before first span and still in clustetr)
                            invalid_edges_mask = (chosen_edges[:, 2] == chosen_edges[i, 2]) & (
                                        chosen_edges[:, 1] <= chosen_edges[:, 2])
                            invalid_edges_mask[:i + 1] = 0  # ensure edges that have been verified aren't deemed 'invalid'
                            chosen_edges = chosen_edges[~invalid_edges_mask]
                            edge_scores = edge_scores[~invalid_edges_mask]

                            coreferent = True
                        else:
                            coreferent = False
                            # cluster is complete; delete all remaining ingoing and outgoing edges from cluster (and within cluster)
                            if i < len(
                                    chosen_edges) - 1:  # we are not on last edge (there are still remaining edges to check)
                                remaining_edges_to_and_from_cluster = ((chosen_edges[i + 1:].unsqueeze(
                                    0) - user_spans_in_proform_cluster.unsqueeze(-1)).abs() == 0).nonzero()
                                if len(remaining_edges_to_and_from_cluster) > 0:
                                    remaining_edges_to_and_from_cluster[:, 1] += i + 1
                                    # delete rows at remaining_edges_to_and_from_cluster[:,1]
                                    chosen_edges[remaining_edges_to_and_from_cluster[:, 1], :] = -1
                                    mask = chosen_edges[:, 0] != -1
                                    chosen_edges = chosen_edges[mask]
                                    edge_scores = edge_scores[mask]
                    else:
                        # TODO: don't ask current antecedent (chosen_edges[i,2]) to ANY element of cluster involving current proform (chosen_edges[i,1])
                        new_antecedent = user_spans_in_proform_cluster[0]
                        delete_first_antecedent_span_mask = (chosen_edges[:, 1] != new_antecedent)
                        delete_first_antecedent_span_mask[:i + 1] = 1  # ensure we don't delete verified edges
                        chosen_edges = chosen_edges[delete_first_antecedent_span_mask]
                        edge_scores = edge_scores[delete_first_antecedent_span_mask]
                        chosen_edges[i, 2] = new_antecedent
                        coreferent = True
            if coreferent:  # set score positive
                edge_scores[i] = edge_scores[i].abs()
                # update span labels...
                span_labels = update_clusters_with_edge(span_labels, chosen_edges[i])
            else:  # set score negative
                edge_scores[i] = -edge_scores[i].abs()
            i += 1
    if return_all_edges:
        # add unadded positive edges to span_labels--i is last unchecked edge
        unadded_pos_edges = chosen_edges[i:][edge_scores[i:] >= 0]
        chosen_edges = chosen_edges[edge_scores >= 0]
        for edge in unadded_pos_edges:
            span_labels = update_clusters_with_edge(span_labels, edge)
    else:
        chosen_edges = chosen_edges[edge_scores >= 0][:num_labels_to_query]
    return chosen_edges, num_labels_queried, total_possible_queries, span_labels


def query_user_labels_mention(mention, output_dict, all_spans, user_labels, translation_reference=None):
    # returns:
    # 1. edge: indA of edge, if coreferent, will be identical to indA_edge_ask, otherwise,
    #          edge antecedent will be "fixed", pointing to -1 if there are no other spans
    #          in cluster, pointing to itself if it is 1st/HEAD span of cluster, or pointing
    #          to 1st span of in cluster if in cluster and not HEAD span
    indC_antecedent = output_dict['predicted_antecedents'][mention[0], mention[1]]
    if indC_antecedent < 0:
        # ask about most likely non-null antecedent
        indC_antecedent = \
            output_dict['coreference_scores'][:, :, 1:][mention[0], mention[1]].argmax()
    edge_ask = torch.cat((mention, indC_antecedent.unsqueeze(0)))
    indA_edge_ask = translate_to_indA(edge_ask.unsqueeze(0), output_dict, all_spans,
                                              translation_reference).squeeze()
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


def find_next_most_uncertain_mention(selector, model_labels, output_dict, queried_mentions_mask, DEBUG_BREAK_FLAG=False):
    '''
    model_labels: batch x num_spans tensor detailing cluster ID of cluster each span belongs to, according to model edges
    and user corrections. IMPORTANT: indexes into TOP_SPANS, not all spans.
    '''
    if selector == 'random':
        # choose random one which hasn't been queried before
        batch_and_mentions = (~queried_mentions_mask).nonzero()
        batch_and_mention = batch_and_mentions[torch.randint(len(batch_and_mentions), (), dtype=torch.int,
                                                             device=model_labels.device)]
        return batch_and_mention, torch.rand(())

    coref_scores_mask = output_dict['coreference_scores'] != -float("inf")
    mention_confidence_scores = torch.zeros(output_dict['top_spans'].size()[:2], dtype=torch.float,
                                            device=model_labels.device)

    for b, score_instance in enumerate(output_dict['coreference_scores']):
        # Sum across all probabilities link to each cluster for each mention
        top_span_mask = torch.ones(output_dict['top_spans'][b].size(0), dtype=torch.float, device=model_labels.device
                                   ).unsqueeze(-1)
        coreference_probs = util.masked_log_softmax(score_instance, top_span_mask).exp()
        # output_dict['antecedent_indices'], and what cluster that antecedent belongs to (-1 for no cluster)
        model_output_mention_pair_clusters = model_labels[b, output_dict['antecedent_indices'][b]]
        model_output_mention_pair_clusters[
            ~coref_scores_mask[b, :, 1:]] = -1  # ensure -infty (impossible antecedents are in no cluster)
        # add 1st column of empties to match coreference_probs size
        model_output_mention_pair_clusters = \
            torch.cat((-torch.ones(model_output_mention_pair_clusters.size(0), dtype=torch.long,
                                   device=model_labels.device).unsqueeze(-1), model_output_mention_pair_clusters), dim=1)

        using_sum = True
        if using_sum:
            if selector == 'qbc':
                num_models = output_dict['coreference_scores_models'].size(0)
                _, model_pred_ants = output_dict['coreference_scores_models'][:, b].max(-1)
                model_pred_ants = model_pred_ants.unsqueeze(-1).expand(-1, -1,
                                                                       model_output_mention_pair_clusters.size(1))
                model_output_mention_pair_clusters = model_output_mention_pair_clusters.unsqueeze(0).expand_as(
                    model_pred_ants)
                # (model x num_mentions) for whether mention is predicted to be pointing to something in a cluster
                model_pred_clusters = torch.gather(model_output_mention_pair_clusters, -1, model_pred_ants)[:, :, 0]
                cluster_mask = model_pred_clusters != -1
                # count cluster votes
                if len(cluster_mask.nonzero()) > 0:
                    num_clusters = model_labels.max() + 1
                    # columns are [m1,m2,m3,m4,m5], rows are each model
                    # [-1,-1,-1, 0,-1]    [.,  .,  .,0+9,   .]
                    # [-1, 1,-1, 1,-1] -> [.,1+3,  .,1+9,   .]
                    # [-1,-1, 0, 1, 2]    [.,  .,0+6,1+9,2+12]
                    # [-1,-1,-1,-1,-1]    [.,  .,  .,  .,   .]
                    # model counts for all of same column (mention) and cluster (label) get mapped to the same bin
                    # [m1c1,m1c2,m1c3,...,m2c1,m2c2,m2c3,...]
                    mention_increment_range_vec = torch.arange(0, model_output_mention_pair_clusters.size(
                        1) * num_clusters, num_clusters, dtype=torch.long, device=model_labels.device)
                    mention_cluster_votes = (model_pred_clusters + mention_increment_range_vec)[
                        cluster_mask].bincount(minlength=len(mention_increment_range_vec) * num_clusters)
                    # (mentions x clusters)
                    mention_cluster_votes = mention_cluster_votes.view(-1, num_clusters)
                    cluster_vote_entropy = mention_cluster_votes.float() / num_models
                    cluster_vote_entropy = cluster_vote_entropy * cluster_vote_entropy.log()
                    # filter nan
                    cluster_vote_entropy[cluster_vote_entropy != cluster_vote_entropy] = 0
                    cluster_vote_entropy = cluster_vote_entropy.sum(-1)
                    mention_confidence_scores[b] += -cluster_vote_entropy.squeeze()
                # count antecedent votes (including empty)
                model_pred_ants = model_pred_ants[:, :, 0]
                num_antecedents = output_dict['coreference_scores'].size(2)
                mention_increment_range_vec = torch.arange(0, model_output_mention_pair_clusters.size(
                    1) * num_antecedents, num_antecedents, dtype=torch.long, device=model_labels.device)
                mention_antecedent_votes = (model_pred_ants + mention_increment_range_vec)[~cluster_mask].bincount(
                    minlength=len(mention_increment_range_vec) * num_antecedents)
                mention_antecedent_votes = mention_antecedent_votes.view(-1, num_antecedents)
                ant_vote_entropy = mention_antecedent_votes.float() / num_models
                ant_vote_entropy = ant_vote_entropy * ant_vote_entropy.log()
                ant_vote_entropy[ant_vote_entropy != ant_vote_entropy] = 0
                ant_vote_entropy = ant_vote_entropy.sum(-1)
                mention_confidence_scores[b] += -ant_vote_entropy
                if DEBUG_BREAK_FLAG:
                    torch.save(mention_confidence_scores, "mention_confidence_scores.txt")
                    torch.save(output_dict['coreference_scores_models'], "coref_scores_models.txt")
                    torch.save(output_dict['coreference_scores'], "coref_scores.txt")
                    torch.save(model_output_mention_pair_clusters, "mention_clusters.txt")
                    os.system("python verify_qbc_scorer.py")
            else:
                # select cluster corresponding to predicted antecedent for each mention
                predicted_antecedents = output_dict['predicted_antecedents'][b].unsqueeze(-1).expand_as(
                    model_output_mention_pair_clusters) + 1
                # all elements of each rows should be the same (since we replicated predicted_antecedents)
                antecedent_clusters = torch.gather(model_output_mention_pair_clusters, 1, predicted_antecedents)[:,0]

                # Get scores of mentions pointing to elements in clusters
                clustered_mask = antecedent_clusters != -1  # mask for mentions selected antecedents in clusters
                if selector == 'entropy':
                    mention_pair_cluster_mask = (model_output_mention_pair_clusters != -1)
                    if len(clustered_mask.nonzero()) > 0:
                        # get rows of those in selected clusters, add scores of each cluster
                        num_clusters = model_output_mention_pair_clusters.max() + 1
                        row_increment_range_vec = torch.arange(0, model_output_mention_pair_clusters.size(
                            0) * num_clusters, num_clusters, dtype=torch.long, device=model_labels.device).unsqueeze(1)
                        row_cluster_sum = (model_output_mention_pair_clusters + row_increment_range_vec)[
                            mention_pair_cluster_mask].bincount(
                            coreference_probs[mention_pair_cluster_mask],
                            minlength=row_increment_range_vec.max() + num_clusters)
                        row_cluster_sum = row_cluster_sum.view(-1, num_clusters)
                        # TODO VERIFY: for i, row in enumerate(row_cluster_sum): assert(len(coreference_probs[i][mention_pair_cluster_mask[i]]) == 0 or len(((row - model_output_mention_pair_clusters[i][mention_pair_cluster_mask[i]].bincount(coreference_probs[i][mention_pair_cluster_mask[i]], minlength=len(row))).abs() > 0.0001).nonzero()) == 0)
                        # add entropies of clusters
                        row_cluster_entropy = row_cluster_sum * row_cluster_sum.log()
                        row_cluster_entropy[
                            row_cluster_entropy != row_cluster_entropy] = 0  # don't want to add nan caused by log-ing 0 probabilities
                        mention_confidence_scores[b] = -row_cluster_entropy.sum(1)
                    row_non_cluster_entropy = coreference_probs * coreference_probs.log()
                    row_non_cluster_entropy[
                        mention_pair_cluster_mask] = 0  # don't add values in clusters (which we've already added)
                    row_non_cluster_entropy[
                        row_non_cluster_entropy != row_non_cluster_entropy] = 0  # don't want to add nan caused by log-ing 0 probabilities
                    mention_confidence_scores[b] += -row_non_cluster_entropy.sum(1)
                elif selector == 'score':
                    if len(clustered_mask.nonzero()) > 0:
                        # mask for mentions belonging in chosen cluster
                        chosen_cluster_rows_mask = (
                                    model_output_mention_pair_clusters[clustered_mask] == antecedent_clusters[
                                clustered_mask].unsqueeze(-1).expand_as(
                                model_output_mention_pair_clusters[clustered_mask]))
                        # get rows of those in selected clusters, add scores
                        mention_confidence_scores[b][clustered_mask] = (
                                    coreference_probs[clustered_mask] * chosen_cluster_rows_mask.float()).sum(1)
                    try:
                        assert (len(predicted_antecedents[~clustered_mask].nonzero()) == 0)
                    except:
                        pdb.set_trace()
                    mention_confidence_scores[b][~clustered_mask] = coreference_probs[~clustered_mask][:, 0]
    if DEBUG_BREAK_FLAG and len(clustered_mask.nonzero()) > 0:
        torch.save(mention_confidence_scores, "mention_confidence_scores.txt")
        torch.save(coreference_probs, "coreference_probs.txt")
        torch.save(model_output_mention_pair_clusters, "model_output_mention_pair_clusters.txt")
        os.system("python verify_scorer.py")
    if selector == 'entropy' or selector == 'qbc':
        opt_score = mention_confidence_scores.max()
    elif selector == 'score':
        opt_score = mention_confidence_scores.min()
    # choose arbitrary unchosen, least-confident mention
    batch_and_mentions = ((mention_confidence_scores == opt_score) & ~queried_mentions_mask).nonzero()
    # check if edge belongs to
    try:
        assert (len(batch_and_mentions) > 0)
    except:
        pdb.set_trace()
    # return least confident mention and associated score
    return batch_and_mentions[0], opt_score