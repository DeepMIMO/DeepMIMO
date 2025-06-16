"""
Module for providing help and information about DeepMIMO dataset parameters and materials.

This module contains utilities to display helpful information about various DeepMIMO
parameters, material properties, and dataset variables through a simple info() interface.
"""

from . import consts as c

# Dictionary of help messages for fundamental matrices
FUNDAMENTAL_MATRICES_HELP = {
    # Power and phase
    c.POWER_PARAM_NAME:
        'Tap power. Received power in dBW for each path, assuming 0 dBW transmitted power. \n'
        '10*log10(|a|²), where a is the complex channel amplitude\n'
        '\t[num_rx, num_paths]',
    c.PHASE_PARAM_NAME:
        'Tap phase. Phase of received signal for each path in degrees. \n'
        '∠a (angle of a), where a is the complex channel amplitude\n'
        '\t[num_rx, num_paths]',
    # Delay
    c.DELAY_PARAM_NAME:
        'Tap delay. Propagation delay for each path in seconds\n'
        '\t[num_rx, num_paths]',
    # Angles
    c.AOA_AZ_PARAM_NAME: 
        'Angle of arrival (azimuth) for each path in degrees\n'
        '\t[num_rx, num_paths]',
    c.AOA_EL_PARAM_NAME: 
        'Angle of arrival (elevation) for each path in degrees\n'
        '\t[num_rx, num_paths]',
    c.AOD_AZ_PARAM_NAME:
        'Angle of departure (azimuth) for each path in degrees\n'
        '\t[num_rx, num_paths]',
    c.AOD_EL_PARAM_NAME:
        'Angle of departure (elevation) for each path in degrees\n'
        '\t[num_rx, num_paths]',
    # Interactions
    c.INTERACTIONS_PARAM_NAME:
        'Type of interactions along each path\n'
        '\tCodes: 0: LOS, 1: Reflection, 2: Diffraction, 3: Scattering, 4: Transmission\n'
        '\tCode meaning: 121 -> Tx-R-D-R-Rx\n'
        '\t[num_rx, num_paths]',
    c.INTERACTIONS_POS_PARAM_NAME:
        '3D coordinates in meters of each interaction point along paths\n'
        '\t[num_rx, num_paths, max_interactions, 3]',
    # Positions
    c.RX_POS_PARAM_NAME:
        'Receiver positions in 3D coordinates in meters\n'
        '\t[num_rx, 3]',
    c.TX_POS_PARAM_NAME:
        'Transmitter positions in 3D coordinates in meters\n'
        '\t[num_tx, 3]',
}

# Dictionary of help messages for computed/derived matrices
COMPUTED_MATRICES_HELP = {
    c.LOS_PARAM_NAME:
        'Line of sight status for each path. \n'
        '\t1: Direct path between TX and RX. \n'
        '\t0: Indirect path (reflection, diffraction, scattering, or transmission). \n'
        '\t-1: No paths between TX and RX. \n'
        '\t[num_rx, ]',
    c.CHANNEL_PARAM_NAME:
        'Channel matrix between TX and RX antennas\n'
        '\t[num_rx, num_rx_ant, num_tx_ant, X], with X = number of paths in time domain \n'
        '\t or X = number of subcarriers in frequency domain',
    c.PWR_LINEAR_PARAM_NAME:
        'Linear power for each path (W)\n'
        '\t[num_rx, num_paths]',
    c.PATHLOSS_PARAM_NAME:
        'Pathloss for each path (dB)\n'
        '\t[num_rx, num_paths]',
    c.DIST_PARAM_NAME:
        'Distance between TX and RX for each path (m)\n'
        '\t[num_rx, num_paths]',
    c.NUM_PATHS_PARAM_NAME:
        'Number of paths for each user\n'
        '\t[num_rx]',
    c.INTER_STR_PARAM_NAME:
        'Interaction string for each path.\n'
        '\tInteraction codes: 0 -> "", 1 -> "R", 2 -> "D", 3 -> "S", 4 -> "T"\n'
        '\tExample interaction integer to string: 121 -> "RDR"\n'
        '\t[num_rx, num_paths]',
    c.DOPPLER_PARAM_NAME:
        'Doppler frequency shifts [Hz] for each user and path\n'
        '\t[num_rx, num_paths]',
    c.INTER_OBJECTS_PARAM_NAME:
        'Object ids at each interaction point\n'
        '\t[num_rx, num_paths, max_interactions]',
}

# Dictionary of help messages for configuration/other parameters
ADDITIONAL_HELP = {
    c.SCENE_PARAM_NAME:
        'Scene parameters',
    c.MATERIALS_PARAM_NAME:
        'List of available materials and their electromagnetic properties',
    c.TXRX_PARAM_NAME:
        'Transmitter/receiver parameters',
    c.RT_PARAMS_PARAM_NAME:
        'Ray-tracing parameters',
}

CHANNEL_HELP_MESSAGES = {
    # BS/UE Antenna Parameters
    c.PARAMSET_ANT_BS: 
        'Base station antenna array configuration parameters. \n',
    c.PARAMSET_ANT_UE: 
        'User equipment antenna array configuration parameters. \n',
    
    # Antenna Parameters
    c.PARAMSET_ANT_BS + '.' + c.PARAMSET_ANT_SHAPE: 
        'Antenna array dimensions [X, Y] or [X, Y, Z] elements\n'
        '\t Default: [1, 1]  |  Type: list[int]  |  Units: number of elements',
    c.PARAMSET_ANT_BS + '.' + c.PARAMSET_ANT_SPACING: 
        'Spacing between antenna elements\n'
        '\t Default: 0.5  |  Type: float  |  Units: wavelengths',
    c.PARAMSET_ANT_BS + '.' + c.PARAMSET_ANT_ROTATION: 
        'Rotation angles [azimuth, elevation, polarization]\n'
        '\t Default: [0, 0, 0]  |  Type: list[float]  |  Units: degrees',
    c.PARAMSET_ANT_BS + '.' + c.PARAMSET_ANT_RAD_PAT: 
        'Antenna element radiation pattern\n'
        '\t Default: "isotropic"  |  Type: str  |  Options: "isotropic", "halfwave-dipole"',
    
    # Channel Configuration
    c.PARAMSET_DOPPLER_EN: 
        'Enable/disable Doppler effect simulation\n'
        '\t Default: False  |  Type: bool',
    c.PARAMSET_NUM_PATHS: 
        'Maximum number of paths to consider per user\n'
        '\t Default: 10  |  Type: int  |  Units: number of paths',
    c.PARAMSET_FD_CH: 
        'Channel domain\n'
        '\t Default: 0  |  Type: int  |  Options: 0 (time domain), 1 (frequency domain/OFDM)',
    
    # OFDM Parameters
    c.PARAMSET_OFDM: 
        f'OFDM channel configuration parameters. Used (and needed!) only if {c.PARAMSET_FD_CH}=1. \n'
        '\t Default: None  |  Type: dict',
    c.PARAMSET_OFDM + '.' + c.PARAMSET_OFDM_BANDWIDTH: 
        'System bandwidth\n'
        '\t Default: 10e6  |  Type: float  |  Units: Hz',
    c.PARAMSET_OFDM + '.' + c.PARAMSET_OFDM_SC_NUM: 
        'Total number of OFDM subcarriers\n'
        '\t Default: 512  |  Type: int  |  Units: number of subcarriers',
    c.PARAMSET_OFDM + '.' + c.PARAMSET_OFDM_SC_SAMP: 
        'Indices of subcarriers to generate\n'
        '\t Default: None (all subcarriers)  |  Type: list[int]  |  Units: subcarrier indices',
    c.PARAMSET_OFDM + '.' + c.PARAMSET_OFDM_LPF: 
        'Enable/disable receive low-pass filter / ADC filter\n'
        '\t Default: False  |  Type: bool',
}

# Combined dictionary for parameter lookups
ALL_PARAMS = {
    **FUNDAMENTAL_MATRICES_HELP,
    **COMPUTED_MATRICES_HELP,
    **ADDITIONAL_HELP,
    **CHANNEL_HELP_MESSAGES
}

def _print_section(title: str, params: dict) -> None:
    """Helper function to print a section of parameter descriptions.
    
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

    if param_name is None or param_name == 'all':
        _print_section("Fundamental Matrices", FUNDAMENTAL_MATRICES_HELP)
        _print_section("Computed/Derived Matrices", COMPUTED_MATRICES_HELP) 
        _print_section("Additional Dataset Fields", ADDITIONAL_HELP)
    
    elif param_name in ['ch_params', 'channel_params']:
        _print_section("Channel Generation Parameters", CHANNEL_HELP_MESSAGES)
    
    else:
        if param_name in ALL_PARAMS:
            print(f"{param_name}: {ALL_PARAMS[param_name]}")
        else:
            print(f"Unknown parameter: {param_name}")
            print("Use info() or info('all') to see all available parameters")
    
    return
            