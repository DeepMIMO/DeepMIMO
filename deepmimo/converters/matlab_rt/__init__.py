"""MATLAB Ray Tracing converter MVP.

This package provides a narrow, file-based MATLAB RT JSON to DeepMIMO scenario
converter. MATLAB Engine execution and rich scene conversion are intentionally
out of scope.
"""

from .converter import convert_matlab_rt_json
from .errors import (
    MatlabRTConversionError,
    MatlabRTSchemaError,
    MatlabRTValidationError,
    MatlabRTWriterError,
    UnsupportedMatlabRTFeatureError,
)
from .schema import (
    MatlabRTExport,
    MatlabRTInteraction,
    MatlabRTLink,
    MatlabRTMetadata,
    MatlabRTReceiver,
    MatlabRTRay,
    MatlabRTScene,
    MatlabRTTransmitter,
)
from .writer import MatlabRTWriteResult

__all__ = [
    "convert_matlab_rt_json",
    "MatlabRTConversionError",
    "MatlabRTExport",
    "MatlabRTInteraction",
    "MatlabRTLink",
    "MatlabRTMetadata",
    "MatlabRTReceiver",
    "MatlabRTRay",
    "MatlabRTScene",
    "MatlabRTSchemaError",
    "MatlabRTTransmitter",
    "MatlabRTValidationError",
    "MatlabRTWriteResult",
    "MatlabRTWriterError",
    "UnsupportedMatlabRTFeatureError",
]
