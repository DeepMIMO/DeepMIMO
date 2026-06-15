"""Typed schema objects for MATLAB RT converter inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MatlabRTMetadata:
    """Top-level MATLAB RT export metadata."""

    experiment: str | None
    matlab_version: str | None
    description: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class MatlabRTScene:
    """Top-level MATLAB RT scene metadata."""

    coordinate_system: str
    frequency_hz: float
    raw: dict[str, Any]


@dataclass(frozen=True)
class MatlabRTTransmitter:
    """One MATLAB ``txsite`` exported as converter input."""

    index: int
    name: str
    antenna_position_m: tuple[float, float, float]
    transmitter_frequency_hz: float | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class MatlabRTReceiver:
    """One MATLAB ``rxsite`` exported as converter input."""

    index: int
    name: str
    antenna_position_m: tuple[float, float, float]
    raw: dict[str, Any]


@dataclass(frozen=True)
class MatlabRTInteraction:
    """One supported MATLAB ray interaction."""

    index: int | None
    type: str
    location_m: tuple[float, float, float]
    material_name: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class MatlabRTRay:
    """One MATLAB ``comm.Ray`` exported as converter input."""

    index: int
    line_of_sight: bool
    transmitter_location_m: tuple[float, float, float]
    receiver_location_m: tuple[float, float, float]
    frequency_hz: float | None
    path_loss_db: float
    phase_shift_rad: float | None
    phase_shift_deg: float | None
    propagation_delay_s: float
    propagation_distance_m: float
    angle_of_departure_deg: tuple[float, float]
    angle_of_arrival_deg: tuple[float, float]
    interactions: tuple[MatlabRTInteraction, ...]
    path_coordinates_m: tuple[tuple[float, float, float], ...]
    raw: dict[str, Any]

    @property
    def num_interactions(self) -> int:
        """Return the number of parsed interactions."""
        return len(self.interactions)


@dataclass(frozen=True)
class MatlabRTLink:
    """One MATLAB TX/RX link with zero or more rays."""

    index: int
    tx_index: int
    rx_index: int
    tx_name: str | None
    rx_name: str | None
    tx_position_m: tuple[float, float, float]
    rx_position_m: tuple[float, float, float]
    rays: tuple[MatlabRTRay, ...]
    raw: dict[str, Any]

    @property
    def num_rays(self) -> int:
        """Return the number of parsed rays."""
        return len(self.rays)


@dataclass(frozen=True)
class MatlabRTExport:
    """Validated MATLAB RT JSON export normalized for conversion."""

    source_path: Path | None
    metadata: MatlabRTMetadata
    scene: MatlabRTScene
    propagation_model: dict[str, Any]
    transmitters: tuple[MatlabRTTransmitter, ...]
    receivers: tuple[MatlabRTReceiver, ...]
    links: tuple[MatlabRTLink, ...]
    raw: dict[str, Any]

    @property
    def num_tx(self) -> int:
        """Return transmitter count."""
        return len(self.transmitters)

    @property
    def num_rx(self) -> int:
        """Return receiver count."""
        return len(self.receivers)
