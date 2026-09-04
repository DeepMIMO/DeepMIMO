pyenv("Version", "/opt/anaconda3/envs/DeepMIMO/bin/python", "ExecutionMode", "OutOfProcess")

% Code Section: Load DeepMIMO Variables
[aoa_az, aoa_el, aod_az, aod_el, delay, power, phase, rx_pos, tx_pos, los, carrier_frequency, bandwidth] = pyrunfile("./DeepMIMO_Matlab_Integration.py", ["aoa_az", "aoa_el", "aod_az", "aod_el", "delay", "power", "phase", "rx_pos", "tx_pos", "los", "carrier_frequency", "bandwidth"]);


aoa_az = double(py.numpy.array(aoa_az, dtype='float64')); % Shape: (131931, 10)
aoa_el = double(py.numpy.array(aoa_el, dtype='float64'));
aod_az = double(py.numpy.array(aod_az, dtype='float64'));
aod_el = double(py.numpy.array(aod_el, dtype='float64'));
delay = double(py.numpy.array(delay, dtype='float64'));
power = double(py.numpy.array(power, dtype='float64'));
phase = double(py.numpy.array(phase, dtype='float64'));
rx_pos = double(py.numpy.array(rx_pos, dtype='float64')); % Shape: (131931, 3)
tx_pos = double(py.numpy.array(tx_pos, dtype='float64')); % Shape: (1, 3)
los = double(py.numpy.array(los, dtype='float64'));
carrier_frequency = double(carrier_frequency);
bandwidth = double(bandwidth);


% Process data for a single user (e.g., user_idx = 1)
user_idx = 10; % Adjust as needed
aoa_az_user = aoa_az(user_idx, :); % Shape: (1, 10)
aoa_el_user = aoa_el(user_idx, :);
aod_az_user = aod_az(user_idx, :);
aod_el_user = aod_el(user_idx, :);
delay_user = delay(user_idx, :);
power_user = power(user_idx, :);
phase_user = phase(user_idx, :);
los_user = los(user_idx); % Scalar LOS status

% Map los_user to HasLOSCluster
has_los = (los_user == 1); % true if los_user is 1, false if -1 or 0
fprintf('User %d LOS status: %d (HasLOSCluster: %d)\n', user_idx, los_user, has_los);

% Filter out NaN values (valid paths)
valid_paths = ~isnan(power_user); % Logical array for non-NaN paths
num_paths = sum(valid_paths); % Number of valid paths
if num_paths == 0
    warning('No valid paths for user %d. Skipping.', user_idx);
    return;
end

% Extract valid path parameters
aoa_az_valid = aoa_az_user(valid_paths); % Azimuth AoA (degrees)
aoa_el_valid = aoa_el_user(valid_paths); % Elevation AoA (degrees)
aod_az_valid = aod_az_user(valid_paths); % Azimuth AoD (degrees)
aod_el_valid = aod_el_user(valid_paths); % Elevation AoD (degrees)
delay_valid = delay_user(valid_paths); % Delays (seconds)
power_valid = power_user(valid_paths); % ***check whether change to dB*** Convert to dB
phase_valid = phase_user(valid_paths); % Phases (radians)

% % Debug: Display valid paths
% fprintf('User %d: %d valid paths\n', user_idx, num_paths);
% disp('Valid aoa_az:'); disp(aoa_az_valid);

% Initialize nrCDLChannel
cdl = nrCDLChannel;
cdl.DelayProfile = 'Custom'; % Use custom delay profile

% Set path parameters
cdl.PathDelays = delay_valid; % Path delays (seconds)
cdl.AveragePathGains = power_valid; % Path gains (dB)
cdl.AnglesAoA = aoa_az_valid; % Azimuth AoA (degrees)
cdl.AnglesZoA = 180 - aoa_el_valid; % Zenith AoA (degrees)
cdl.AnglesAoD = aod_az_valid; % Azimuth AoD (degrees)
cdl.AnglesZoD = 180 - aod_el_valid; % Zenith AoD (degrees)

% LOS detection
cdl.HasLOSCluster = has_los; % Set from los_user

% Channel configuration
cdl.CarrierFrequency = carrier_frequency; % 3.5 GHz
cdl.NormalizeChannelOutputs = false; % do not normalize by the number of receive antennas, this would change the receive power
cdl.NormalizePathGains = false;      % set to false to retain the path gains
cdl.MaximumDopplerShift = 0; % Static scenario

% Antenna configuration
cdl.TransmitAntennaArray.Size = [1, 1, 1, 1, 1];
cdl.ReceiveAntennaArray.Size = [1, 1, 1, 1, 1];

% displayChannel(cdl,'LinkEnd','Tx');

% displayChannel(cdl,'LinkEnd','Rx');

% % Create a random input waveform
% cdl_info = info(cdl);
% Nt = cdl_info.NumTransmitAntennas;
% T = cdl.SampleRate * 1e-3; % 1 ms
% txWaveform = complex(randn(T, Nt), randn(T, Nt));
% 
% % Pass through channel
% [rxWaveform, pathGains] = cdl(txWaveform);
% 
% % Plot path gains
% figure;
% plot(abs(pathGains));
% xlabel('Sample'); ylabel('Path Gain Magnitude');
% title(sprintf('Path Gains for User %d', user_idx));


% Set SampleRate based on bandwidth (required for discrete-time CIR)
cdl.SampleRate = bandwidth;  % Assuming bandwidth is in Hz (system sampling rate)

% Get channel info (required for NumTransmitAntennas)
cdl_info = info(cdl);

% CIR Extraction Parameters
num_realizations = 10;  % Number of stochastic realizations (adjust as needed)
max_delay = max(delay_valid);  % Maximum path delay (s)
filter_tail = 20;  % Extra samples to capture filter response tail (adjust based on your bandwidth/delays)
num_samples_cir = ceil(max_delay * cdl.SampleRate) + filter_tail + 1;  % Length for impulse input to cover all delays

% Prepare impulse waveform (delta at t=0)
txWaveform_impulse = zeros(num_samples_cir, cdl_info.NumTransmitAntennas);
txWaveform_impulse(1, 1) = 1;  % Impulse for SISO (extend for MIMO if needed)

% Arrays to store CIRs for all realizations (optional, for further analysis)
all_cirs = zeros(num_realizations, num_samples_cir);

% Plot setup for multiple CIR magnitudes
figure;
hold on;
title(sprintf('Magnitude of CIR for User %d (Multiple Realizations)', user_idx));
xlabel('Delay (s)');
ylabel('|CIR|');
grid on;

for r = 1:num_realizations
    reset(cdl);  % Reset channel to re-initialize random phases for a new stochastic realization

    % Pass impulse through channel (ignore pathGains if not needed)
    [cirWaveform, ~] = cdl(txWaveform_impulse);

    % Store CIR (complex-valued discrete-time response)
    all_cirs(r, :) = cirWaveform(:, 1);  % For first Rx antenna (adjust for MIMO)

    % Plot magnitude vs. delay for this realization
    tau = (0:num_samples_cir-1) / cdl.SampleRate;  % Delay axis
    plot(tau, abs(cirWaveform(:, 1)), 'LineWidth', 1.5);
end

hold off;
legend(arrayfun(@(x) sprintf('Realization %d', x), 1:num_realizations, 'UniformOutput', false));



% Optional: Plot phases for one realization (e.g., first one)
figure;
plot(tau, angle(all_cirs(1,:)));
title(sprintf('Phase of CIR for User %d (Realization 1)', user_idx));
xlabel('Delay (s)');
ylabel('Phase (rad)');
grid on;
