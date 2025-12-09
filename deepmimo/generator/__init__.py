"""DeepMIMO Channel Generator Module.

This module provides channel generation functionality:
- Channel parameter management
- MIMO channel computation
- Beamforming geometry calculations
- Antenna pattern handling
"""

from .channel import ChannelParameters
from .geometry import steering_vec

__all__ = [
    "ChannelParameters",
    "steering_vec",
]
