# pylint: disable=access-member-before-definition
from typing import Dict

from overrides import overrides
import torch

from allennlp.data.fields.field import Field
from allennlp.data.fields.span_field import SpanField


class SpanPairField(Field[torch.Tensor]):
    def __init__(self, first_item: SpanField, sec_item: SpanField) -> None:
        self.first_item = first_item
        self.sec_item = sec_item

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        # pylint: disable=no-self-use
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        # pylint: disable=unused-argument
        tensor = torch.stack([self.first_item.as_tensor(padding_lengths), self.sec_item.as_tensor(padding_lengths)], dim=0)
        return tensor

    @overrides
    def empty_field(self):
        return SpanPairField(SpanField.empty_field, SpanField.empty_field)

    def __str__(self) -> str:
        return f"SpanField with spans: ({self.first_item}, {self.sec_item})."
