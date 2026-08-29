from .parsec import (
    load_parsec_isochrone,
    load_parsec_track,
)

from .girardi2000 import (
    load_girardi2000_isochrone,
    load_girardi2000_track,
)

__all__ = [
    "load_parsec_track",
    "load_parsec_isochrone",
    "load_girardi2000_track",
    "load_girardi2000_isochrone",
]
