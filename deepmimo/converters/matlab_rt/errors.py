"""Typed exceptions for the MATLAB RT converter MVP."""


class MatlabRTConversionError(Exception):
    """Base class for MATLAB RT conversion failures."""


class MatlabRTSchemaError(MatlabRTConversionError):
    """Raised when MATLAB RT input does not match the supported schema."""


class UnsupportedMatlabRTFeatureError(MatlabRTConversionError):
    """Raised when valid MATLAB RT input uses a feature outside the MVP scope."""


class MatlabRTValidationError(MatlabRTConversionError):
    """Raised when parsed MATLAB RT data is internally inconsistent."""


class MatlabRTWriterError(MatlabRTConversionError):
    """Raised when writing a DeepMIMO scenario fails."""
