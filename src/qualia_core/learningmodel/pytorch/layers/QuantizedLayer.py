from __future__ import annotations

from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qualia_core.learningmodel.pytorch.Quantizer import Quantizer  # noqa: TCH001


class QuantizedLayer:
    quantizer_input: Quantizer | None
    quantizer_act: Quantizer | None
    quantizer_w: Quantizer | None

    def __init__(self) -> None:
        super().__init__()
        self.quantizer_input = None
        self.quantizer_act = None
        self.quantizer_w = None

    @property
    def input_q(self) -> int | None:
        if self.quantizer_input is None:
            return None
        return self.quantizer_input.fractional_bits

    @property
    def activation_q(self) -> int | None:
        if self.quantizer_act is None:
            return None
        return self.quantizer_act.fractional_bits

    @property
    def weights_q(self) -> int | None:
        if self.quantizer_w is None:
            return None
        return self.quantizer_w.fractional_bits

    @property
    def input_round_mode(self) -> str | None:
        if self.quantizer_input is None:
            return None
        return self.quantizer_input.roundtype

    @property
    def activation_round_mode(self) -> str | None:
        if self.quantizer_act is None:
            return None
        return self.quantizer_act.roundtype

    @property
    def weights_round_mode(self) -> str | None:
        if self.quantizer_w is None:
            return None
        return self.quantizer_w.roundtype
