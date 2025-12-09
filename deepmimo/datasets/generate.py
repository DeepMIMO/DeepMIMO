"""DeepMIMO Dataset Generation Module.

This module provides the main generate() function for creating DeepMIMO datasets
with channel matrices computed.
"""

from typing import Any

from deepmimo.datasets.dataset import Dataset, DynamicDataset, MacroDataset
from deepmimo.datasets.load import load
from deepmimo.generator.channel import ChannelParameters


def generate(
    scen_name: str,
    *,
    load_params: dict[str, Any] | None = None,
    trim_params: dict[str, Any] | None = None,
    ch_params: dict[str, Any] | None = None,
) -> Dataset | MacroDataset | DynamicDataset:
    """Generate a DeepMIMO dataset for a given scenario.

    This function wraps loading scenario data, computing channels, and organizing results.

    Args:
        scen_name (str): Name of the scenario to generate data for
        load_params (dict): Parameters for loading the scenario. Defaults to {}.
        trim_params (dict, optional): Parameters for dataset trimming. Supports:
            - idxs (array-like): UE indices to keep (applied first)
            - idxs_mode (str): One of 'active'|'linear'|'uniform'|'row'|'col'|'limits'
            - idxs_kwargs (dict): Keyword args for the idxs mode
            - bs_fov (list|tuple [h_deg, v_deg])
            - ue_fov (list|tuple [h_deg, v_deg])
            - path_depth (int)
            - path_types (list[str])
        ch_params (dict): Parameters for channel generation. Defaults to {}.

    Returns:
        Dataset: Generated DeepMIMO dataset containing channel matrices and metadata

    Raises:
        ValueError: If scenario name is invalid or required files are missing

    """
    if ch_params is None:
        ch_params = {}
    if trim_params is None:
        trim_params = {}
    if load_params is None:
        load_params = {}
    dataset = load(scen_name, **load_params)

    if trim_params:
        if "idxs" not in trim_params:
            trim_params["idxs"] = dataset.get_idxs(
                trim_params["idxs_mode"],
                **trim_params.get("idxs_kwargs", {}),
            )

        dataset = dataset.trim(**trim_params)

    _ = dataset.compute_channels(ch_params if ch_params else ChannelParameters())

    return dataset

