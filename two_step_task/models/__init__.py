"""Model implementations for two-step task.

This package contains lightweight model classes for M1, M2, and M3 as
described in the experiment outline. These classes provide parameter
definitions and small helper methods; full agent inference will be
implemented in subsequent steps."""

from .m1_static import M1_StaticPrecision
from .m2_entropy import M2_EntropyCoupled
from .m3_profiles import M3_ProfileBased
from .generative import build_A_matrices, build_B_matrices

__all__ = [
    "M1_StaticPrecision",
    "M2_EntropyCoupled",
    "M3_ProfileBased",
    "build_A_matrices",
    "build_B_matrices",
]
