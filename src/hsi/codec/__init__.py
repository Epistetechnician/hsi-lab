from hsi.codec.base import Codec, Trace, fit_to_budget
from hsi.codec.text import TextCodec
from hsi.codec.typed import TypedEventCodec

CODECS: dict[str, Codec] = {"text": TextCodec(), "typed_event": TypedEventCodec()}

__all__ = ["CODECS", "Codec", "TextCodec", "Trace", "TypedEventCodec", "fit_to_budget"]
