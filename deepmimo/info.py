"""DeepMIMO parameter and material help utilities."""

from textwrap import dedent

from . import consts as c

# Dictionary of help messages for fundamental matrices
FUNDAMENTAL_MATRICES_HELP = {
    # Power and phase
    c.POWER_PARAM_NAME: dedent(
        """\
        Tap power. Received power in dBW for each path, assuming 0 dBW transmitted power.
        10*log10(|a|²), where a is the complex channel amplitude
        \t[num_rx, num_paths]
        """
    ),
    c.PHASE_PARAM_NAME: dedent(
        """\
        Tap phase. Phase of received signal for each path in degrees.
        ∠a (angle of a), where a is the complex channel amplitude
        \t[num_rx, num_paths]
        """
    ),
    # Delay
    c.DELAY_PARAM_NAME: dedent(
        """\
        Tap delay. Propagation delay for each path in seconds
        \t[num_rx, num_paths]
        """
    ),
    # Angles
    c.AOA_AZ_PARAM_NAME: dedent(
        """\
        Angle of arrival (azimuth) for each path in degrees
        \t[num_rx, num_paths]
        """
    ),
    c.AOA_EL_PARAM_NAME: dedent(
        """\
        Angle of arrival (elevation) for each path in degrees
        \t[num_rx, num_paths]
        """
    ),
    c.AOD_AZ_PARAM_NAME: dedent(
        """\
        Angle of departure (azimuth) for each path in degrees
        \t[num_rx, num_paths]
        """
    ),
    c.AOD_EL_PARAM_NAME: dedent(
        """\
        Angle of departure (elevation) for each path in degrees
        \t[num_rx, num_paths]
        """
    ),
    # Interactions
    c.INTERACTIONS_PARAM_NAME: dedent(
        """\
        Type of interactions along each path
        \tCodes: 0: LOS, 1: Reflection, 2: Diffraction, 3: Scattering, 4: Transmission
        \tCode meaning: 121 -> Tx-R-D-R-Rx
        \t[num_rx, num_paths]
        """
    ),
    c.INTERACTIONS_POS_PARAM_NAME: dedent(
        """\
        3D coordinates in meters of each interaction point along paths
        \t[num_rx, num_paths, max_interactions, 3]
        """
    ),
    # Positions
    c.RX_POS_PARAM_NAME: "Receiver positions in 3D coordinates in meters\n\t[num_rx, 3]",
    c.TX_POS_PARAM_NAME: "Transmitter positions in 3D coordinates in meters\n\t[num_tx, 3]",
}

# Dictionary of help messages for computed/derived matrices
COMPUTED_MATRICES_HELP = {
    c.LOS_PARAM_NAME: dedent(
        """\
        Line of sight status for each path.
        \t1: Direct path between TX and RX.
        \t0: Indirect path (reflection, diffraction, scattering, or transmission).
        \t-1: No paths between TX and RX.
        \t[num_rx, ]
        """
    ),
    c.CHANNEL_PARAM_NAME: dedent(
        """\
        Channel matrix between TX and RX antennas
        \t[num_rx, num_rx_ant, num_tx_ant, X], with X = number of paths in time domain
        \t or X = number of subcarriers in frequency domain
        """
    ),
    c.PWR_LINEAR_PARAM_NAME: "Linear power for each path (W)\n\t[num_rx, num_paths]",
    c.PATHLOSS_PARAM_NAME: "Pathloss for each path (dB)\n\t[num_rx, num_paths]",
    c.DIST_PARAM_NAME: "Distance between TX and RX for each path (m)\n\t[num_rx, num_paths]",
    c.NUM_PATHS_PARAM_NAME: "Number of paths for each user\n\t[num_rx]",
    c.INTER_STR_PARAM_NAME: dedent(
        """\
        Interaction string for each path.
        \tInteraction codes: 0 -> "", 1 -> "R", 2 -> "D", 3 -> "S", 4 -> "T"
        \tExample interaction integer to string: 121 -> "RDR"
        \t[num_rx, num_paths]
        """
    ),
    c.DOPPLER_PARAM_NAME: dedent(
        """\
        Doppler frequency shifts [Hz] for each user and path
        \t[num_rx, num_paths]
        """
    ),
    c.INTER_OBJECTS_PARAM_NAME: dedent(
        """\
        Object ids at each interaction point
        \t[num_rx, num_paths, max_interactions]
        """
    ),
}

# Dictionary of help messages for configuration/other parameters
ADDITIONAL_HELP = {
    c.SCENE_PARAM_NAME: "Scene parameters",
    c.MATERIALS_PARAM_NAME: "List of available materials and their electromagnetic properties",
    c.TXRX_PARAM_NAME: "Transmitter/receiver parameters",
    c.RT_PARAMS_PARAM_NAME: "Ray-tracing parameters",
}

CHANNEL_HELP_MESSAGES = {
    # BS/UE Antenna Parameters
    c.PARAMSET_ANT_BS: "Base station antenna array configuration parameters.\n",
    c.PARAMSET_ANT_UE: "User equipment antenna array configuration parameters.\n",
    # Antenna Parameters
    c.PARAMSET_ANT_BS + "." + c.PARAMSET_ANT_SHAPE: dedent(
        """\
        Antenna array dimensions [X, Y] or [X, Y, Z] elements
        \t Default: [1, 1]  |  Type: list[int]  |  Units: number of elements
        """
    ),
    c.PARAMSET_ANT_BS + "." + c.PARAMSET_ANT_SPACING: dedent(
        """\
        Spacing between antenna elements
        \t Default: 0.5  |  Type: float  |  Units: wavelengths
        """
    ),
    c.PARAMSET_ANT_BS + "." + c.PARAMSET_ANT_ROTATION: dedent(
        """\
        Rotation angles [azimuth, elevation, polarization]
        \t Default: [0, 0, 0]  |  Type: list[float]  |  Units: degrees
        """
    ),
    c.PARAMSET_ANT_BS + "." + c.PARAMSET_ANT_RAD_PAT: dedent(
        """\
        Antenna element radiation pattern
        \t Default: "isotropic"  |  Type: str  |  Options: "isotropic", "halfwave-dipole"
        """
    ),
    # Channel Configuration
    c.PARAMSET_DOPPLER_EN: dedent(
        """\
        Enable/disable Doppler effect simulation
        \t Default: False  |  Type: bool
        """
    ),
    c.PARAMSET_NUM_PATHS: dedent(
        """\
        Maximum number of paths to consider per user
        \t Default: 10  |  Type: int  |  Units: number of paths
        """
    ),
    c.PARAMSET_FD_CH: dedent(
        """\
        Channel domain
        \t Default: 0  |  Type: int  |  Options: 0 (time domain), 1 (frequency domain/OFDM)
        """
    ),
    # OFDM Parameters
    c.PARAMSET_OFDM: dedent(
        f"""\
        OFDM channel configuration parameters. Used (and needed!) only if {c.PARAMSET_FD_CH}=1.
        \t Default: None  |  Type: dict
        """
    ),
    c.PARAMSET_OFDM + "." + c.PARAMSET_OFDM_BANDWIDTH: dedent(
        """\
        System bandwidth
        \t Default: 10e6  |  Type: float  |  Units: Hz
        """
    ),
    c.PARAMSET_OFDM + "." + c.PARAMSET_OFDM_SC_NUM: dedent(
        """\
        Total number of OFDM subcarriers
        \t Default: 512  |  Type: int  |  Units: number of subcarriers
        """
    ),
    c.PARAMSET_OFDM + "." + c.PARAMSET_OFDM_SC_SAMP: dedent(
        """\
        Indices of subcarriers to generate
        \t Default: None (all subcarriers)  |  Type: list[int]  |  Units: subcarrier indices
        """
    ),
    c.PARAMSET_OFDM + "." + c.PARAMSET_OFDM_LPF: dedent(
        """\
        Enable/disable receive low-pass filter / ADC filter
        \t Default: False  |  Type: bool
        """
    ),
}

# Combined dictionary for parameter lookups
ALL_PARAMS = {
    **FUNDAMENTAL_MATRICES_HELP,
    **COMPUTED_MATRICES_HELP,
    **ADDITIONAL_HELP,
    **CHANNEL_HELP_MESSAGES,
}


def _print_section(title: str, params: dict) -> None:
    """Print a section of parameter descriptions.

    Args:
        title: Section title to display
        params: Dictionary of parameter names and their descriptions

    """
    print(f"\n{title}:")
    print("=" * 30)
    for param, msg in params.items():
        print(f"{param}: {msg}")


def info(param_name: str | object | None = None) -> None:
    """Display help information about DeepMIMO dataset parameters and materials.

    Args:
        param_name: Name of the parameter to get info about, or object to get help for.
                   If a string, must be one of the valid parameter names or 'materials'.
                   If an object, displays Python's built-in help for that object.
                   If None or 'all', displays information for all parameters.
                   If the parameter name is an alias, shows info for the resolved parameter.

    Returns:
        None

    """
    if not isinstance(param_name, (str, type(None))):
        help(param_name)
        return

    # Check if it's an alias and resolve it first
    if param_name in c.DATASET_ALIASES:
        resolved_name = c.DATASET_ALIASES[param_name]
        print(f"'{param_name}' is an alias for '{resolved_name}'")
        param_name = resolved_name

    if param_name is None or param_name == "all":
        _print_section("Fundamental Matrices", FUNDAMENTAL_MATRICES_HELP)
        _print_section("Computed/Derived Matrices", COMPUTED_MATRICES_HELP)
        _print_section("Additional Dataset Fields", ADDITIONAL_HELP)

    elif param_name in ["ch_params", "channel_params"]:
        _print_section("Channel Generation Parameters", CHANNEL_HELP_MESSAGES)

    elif param_name in ALL_PARAMS:
        print(f"{param_name}: {ALL_PARAMS[param_name]}")
    else:
        print(f"Unknown parameter: {param_name}")
        print("Use info() or info('all') to see all available parameters")

    return
