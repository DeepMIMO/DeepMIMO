"""Common utilities for DeepMIMO API modules."""

from pathlib import Path
from typing import Any, TypedDict

# API configuration
API_BASE_URL = "https://deepmimo.net"

# Headers for HTTP requests
HEADERS = {
    "User-Agent": "DeepMIMO-Python/4.0",
    "Accept": "*/*",
}

# File size limits
FILE_SIZE_LIMIT = 1 * 1024**3  # Scenario zip file size limit: 1GB
RT_FILE_SIZE_LIMIT = 5 * 1024**3  # RT source zip file size limit: 5GB
IMAGE_SIZE_LIMIT = 10 * 1024**2  # Image size limit: 10MB

# Request configuration
REQUEST_TIMEOUT = 30  # seconds
HTTP_OK = 200

# Frequency band limits
SUB6_UPPER_GHZ = 6
MMW_UPPER_GHZ = 100

# Upload limits
MAX_IMAGES_PER_UPLOAD = 5


class SubmissionConfig(TypedDict):
    """Configuration for scenario submission."""

    params_dict: dict
    details: list[str] | None
    extra_metadata: dict | None
    include_images: bool


class _ProgressFileReader:
    """Progress file reader for uploading files to the DeepMIMO API."""

    def __init__(self, file_path: str | Path, progress_bar: Any) -> None:
        self.file_path = file_path
        self.progress_bar = progress_bar
        self.file_object = Path(file_path).open("rb")  # noqa: SIM115
        self.len = Path(file_path).stat().st_size
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self.file_object.read(size)
        self.bytes_read += len(data)
        self.progress_bar.n = self.bytes_read
        self.progress_bar.refresh()
        return data

    def close(self) -> None:
        self.file_object.close()

