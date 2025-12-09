"""Core data models for DeepMIMO.

This module contains the fundamental data structures that represent:
- Physical world geometry (Scene, Face, PhysicalElement)
- Material properties (Material, MaterialList)
- Ray tracing parameters (RayTracingParameters)
- Transmitter/Receiver configuration (TxRxSet, TxRxPair)
"""

from .materials import Material, MaterialList
from .rt_params import RayTracingParameters
from .scene import (
    BoundingBox,
    Face,
    PhysicalElement,
    PhysicalElementGroup,
    Scene,
)
from .txrx import TxRxPair, TxRxSet, get_txrx_pairs, get_txrx_sets, print_available_txrx_pair_ids

__all__ = [
    # Materials
    "Material",
    "MaterialList",
    # Ray Tracing Parameters
    "RayTracingParameters",
    # Scene
    "BoundingBox",
    "Face",
    "PhysicalElement",
    "PhysicalElementGroup",
    "Scene",
    # TX/RX
    "TxRxPair",
    "TxRxSet",
    "get_txrx_pairs",
    "get_txrx_sets",
    "print_available_txrx_pair_ids",
]

