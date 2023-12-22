from __future__ import annotations

from typing import Protocol

from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qualia_core.learningmodel.pytorch.Quantizer import Quantizer  # noqa: TCH001

class QuantizerInputProtocol(Protocol):
    def __init__(self, *_: object, **__: object) -> None:
        super().__init__()

    quantizer_input: Quantizer

class QuantizerActProtocol(Protocol):
    def __init__(self, *_: object, **__: object) -> None:
        super().__init__()

    quantizer_act: Quantizer

class QuantizerWProtocol(Protocol):
    def __init__(self, *_: object, **__: object) -> None:
        super().__init__()

    quantizer_w: Quantizer

class QuantizedLayer:
    def __init__(self, *_: object, **__: object) -> None:
        super().__init__()
        self.quantizer_input: Quantizer | None = None
        self.quantizer_act: Quantizer | None = None
        self.quantizer_w: Quantizer | None = None

    @property
    def input_q(self) -> int | None:
        """Number of bits used to encode the fractional part of the input in case of fixed-point quantization.

        See :meth:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer.fractional_bits`.

        :return: Fractional part bits for the input or ``None`` if not applicable.
        """
        if self.quantizer_input is None:
            return None
        return self.quantizer_input.fractional_bits

    @property
    def activation_q(self) -> int | None:
        """Number of bits used to encode the fractional part of the output in case of fixed-point quantization.

        See :meth:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer.fractional_bits`.

        :return: Fractional part bits for the output or ``None`` if not applicable.
        """
        if self.quantizer_act is None:
            return None
        return self.quantizer_act.fractional_bits

    @property
    def weights_q(self) -> int | None:
        """Number of fractional part bits for the weights in case of fixed-point quantization.

        See :meth:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer.fractional_bits`.

        :return: Fractional part bits for the weights or ``None`` if not applicable.
        """
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
