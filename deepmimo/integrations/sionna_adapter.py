"""DeepMIMO dataset adapter for Sionna.

This module provides functionality to adapt DeepMIMO datasets for use with Sionna.
It handles:
- Channel data formatting and conversion
- Multi-user scenario support
- Multi-basestation scenario support

The adapter assumes BSs are transmitters and users are receivers. Uplink channels
can be generated using (transpose) reciprocity.
"""

# Standard library imports

# Third-party imports
import numpy as np
from numpy.typing import NDArray

from deepmimo.datasets.dataset import Dataset, MacroDataset


class SionnaAdapter:
    """Class for converting DeepMIMO dataset format to Sionna format.

    This class handles the conversion of channel data from DeepMIMO format to
    the format expected by Sionna, supporting various configurations of BSs and UEs.

    The dataset provided will be used to the fullest. If some data is not needed -
    for example, only one BS instead of all - then the dataset should be subsetted
    before calling this class.

    Attributes:
        dataset (dict): The loaded DeepMIMO dataset.
        num_rx_ant (int): Number of receiver antennas.
        num_tx_ant (int): Number of transmitter antennas.
        num_samples_bs (int): Number of basestation samples.
        num_samples_ue (int): Number of user samples.
        num_samples (int): Total number of channel samples.
        num_rx (int): Number of receivers per sample.
        num_tx (int): Number of transmitters per sample.
        num_paths (int): Number of paths per channel.
        num_time_steps (int): Number of time steps (1 for static).
        ch_shape (tuple): Required shape for channel coefficients.
        t_shape (tuple): Required shape for path delays.

    """

    def __init__(self, dataset: Dataset | MacroDataset) -> None:
        """Initialize the Sionna adapter.

        Args:
            dataset (Dataset | MacroDataset): A loaded DeepMIMO dataset, using dm.load().

        """
        if isinstance(dataset, Dataset):
            self.dataset = [dataset]  # single BS
            self.num_tx = 1
        else:
            self.dataset = dataset  # multiple BSs
            self.num_tx = len(self.dataset)

        self.num_rx = 1
        # (Assumption: all receivers have the same number of UEs)

        # Extract number of antennas from the DeepMIMO dataset
        # (Assumption: all BSs and UEs have the same number of antennas)
        first_ds = self.dataset[0]
        self.num_rx_ant = first_ds.channels.shape[1]
        self.num_tx_ant = first_ds.channels.shape[2]

        if first_ds.ch_params.freq_domain:
            msg = "Sionna adapter needs time domain channels"
            raise ValueError(msg)

        self.num_paths = first_ds.channels.shape[-1]
        self.num_time_steps = 1  # Time step = 1 for static scenarios

        # The required path power shape for Sionna
        self.ch_shape = (
            self.num_rx,
            self.num_rx_ant,
            self.num_tx,
            self.num_tx_ant,
            self.num_paths,
            self.num_time_steps,
        )

        # The required path delay shape for Sionna
        self.t_shape = (self.num_rx, self.num_tx, self.num_paths)

        self.num_rx_samples = first_ds.n_ue  # number of UE samples
        self.num_tx_samples = self.num_tx  # number of BS samples
        self.num_samples = self.num_rx_samples * self.num_tx_samples  # total number of samples

    def __len__(self) -> int:
        """Get number of available channel samples.

        Returns:
            int: Total number of channel samples.

        """
        return self.num_samples

    def __call__(self) -> tuple[NDArray, NDArray]:
        """Generate channel samples in Sionna format.

        This function yields channel samples one at a time, converting them
        from DeepMIMO format to Sionna format.

        Returns:
            tuple[NDArray, NDArray]: Tuple containing:
                - Channel coefficients array of shape ch_shape.
                - Path delays array of shape t_shape.

        """
        # # Generate zero vectors for the Sionna sample
        # a = np.zeros(self.ch_shape, dtype=np.csingle)
        # tau = np.zeros(self.t_shape, dtype=np.single)

        # for i in range(self.num_rx_samples):  # For each UE sample
        #     for j in range(self.num_tx_samples):  # For each BS sample
        #         # Place the DeepMIMO dataset power and delays into the channel sample

        #         n_paths = self.dataset[j].num_paths[i]
        #         a[i, :, j, :, :, 0] = self.dataset[j].channels[i, :, :, :]
        #         tau[i, j, :n_paths] = self.dataset[j].toa[i, :n_paths]
        # yield (a, tau)  # yield this sample

        for i in range(self.num_rx_samples):  # For each UE sample
            for j in range(self.num_tx_samples):  # For each BS sample
                # Generate zero vectors for the Sionna sample
                a = np.zeros(self.ch_shape, dtype=np.csingle)
                tau = np.zeros(self.t_shape, dtype=np.single)

                # Place the DeepMIMO dataset power and delays into the channel sample for Sionna
                for i_ch in range(self.num_rx):  # for each receiver in the sample
                    for j_ch in range(self.num_tx):  # for each transmitter in the sample
                        n_paths = self.dataset[j].num_paths[i]
                        a[i_ch, :, j_ch, :, :, 0] = self.dataset[j].channels[i, :, :, :]
                        tau[i_ch, j_ch, :n_paths] = self.dataset[j].toa[i, :n_paths]
                yield (a, tau)  # yield this sample
