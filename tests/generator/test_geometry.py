import numpy as np
import pytest

from deepmimo.generator.geometry import (
	_array_response as array_response,
	_array_response_batch as array_response_batch,
	_ant_indices as ant_indices,
	_apply_FoV as apply_FoV,
	_apply_FoV_batch as apply_FoV_batch,
	_rotate_angles as rotate_angles,
	_rotate_angles_batch as rotate_angles_batch,
	steering_vec,
)


def test_array_response_batch_matches_single():
	# Parameters
	panel_size = (4, 2)  # 8 antennas
	kd = 2 * np.pi * 0.5
	n_users = 10
	n_paths = 5

	rng = np.random.default_rng(42)
	theta = rng.uniform(0, np.pi, size=(n_users, n_paths))
	phi = rng.uniform(0, 2 * np.pi, size=(n_users, n_paths))

	# Add NaNs
	nan_mask = rng.random(size=(n_users, n_paths)) < 0.2
	theta[nan_mask] = np.nan
	phi[nan_mask] = np.nan

	ant_ind = ant_indices(panel_size)

	# Batch
	batch_responses = array_response_batch(ant_ind, theta, phi, kd)  # [B, N, P]

	# Single
	n_ant = len(ant_ind)
	single_responses = np.zeros((n_users, n_paths, n_ant), dtype=np.complex128)
	for i in range(n_users):
		for j in range(n_paths):
			if not np.isnan(theta[i, j]):
				single_responses[i, j, :] = array_response(
					ant_ind,
					theta[i, j],
					phi[i, j],
					kd,
				).ravel()

	# Align shapes
	batch_responses = np.moveaxis(batch_responses, 1, 2)  # [B, P, N]
	np.testing.assert_allclose(batch_responses, single_responses, rtol=1e-10, atol=1e-12)


def test_array_response_batch_edge_cases():
	panel_size = (2, 2)
	kd = 2 * np.pi * 0.5
	ant_ind = ant_indices(panel_size)
	n_ant = len(ant_ind)

	# Single user/path
	theta = np.array([[np.pi / 4]])
	phi = np.array([[np.pi / 3]])
	result = array_response_batch(ant_ind, theta, phi, kd)
	assert result.shape == (1, n_ant, 1)

	# All NaN
	theta = np.full((2, 3), np.nan)
	phi = np.full((2, 3), np.nan)
	result = array_response_batch(ant_ind, theta, phi, kd)
	assert result.shape == (2, n_ant, 3)
	assert np.all(result == 0)

	# Partial NaN
	theta = np.array([[np.pi / 4, np.nan, np.pi / 3], [np.nan, np.pi / 6, np.pi / 2]])
	phi = np.array([[np.pi / 3, np.nan, np.pi / 4], [np.nan, np.pi / 2, np.pi / 6]])
	result = array_response_batch(ant_ind, theta, phi, kd)

	expected = np.zeros((2, n_ant, 3), dtype=np.complex128)
	for i in range(2):
		for j in range(3):
			if not np.isnan(theta[i, j]):
				expected[i, :, j] = array_response(ant_ind, theta[i, j], phi[i, j], kd).ravel()

	assert result.shape == (2, n_ant, 3)
	np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
	"fov_deg,theta_rad,phi_rad,expected",
	[
		((360.0, 180.0), np.pi / 4, np.pi / 2, True),
		((180.0, 90.0), np.pi / 2, 0.0, True),
		((60.0, 30.0), np.pi / 2, np.pi, False),
	],
)
def test_apply_fov_single_vs_batch(fov_deg, theta_rad, phi_rad, expected):
	mask_single = apply_FoV(fov=fov_deg, theta=theta_rad, phi=phi_rad)
	mask_batch = apply_FoV_batch(fov=fov_deg, theta=np.array([[theta_rad]]), phi=np.array([[phi_rad]]))
	assert bool(mask_single) == bool(mask_batch[0, 0]) == expected


def test_apply_fov_batch_equivalence_random():
	rng = np.random.default_rng(0)
	n_users, n_paths = 50, 7
	theta = rng.uniform(0, np.pi, (n_users, n_paths))
	phi = rng.uniform(0, 2 * np.pi, (n_users, n_paths))
	fov = (180.0, 90.0)

	mask_single = np.zeros((n_users, n_paths), dtype=bool)
	for i in range(n_users):
		for j in range(n_paths):
			mask_single[i, j] = apply_FoV(fov=fov, theta=theta[i, j], phi=phi[i, j])
	mask_batch = apply_FoV_batch(fov=fov, theta=theta, phi=phi)

	assert mask_single.shape == mask_batch.shape
	np.testing.assert_array_equal(mask_single, mask_batch)


def test_rotate_angles_zero_rotation():
	theta_single = np.array([45.0])  # degrees
	phi_single = np.array([90.0])  # degrees
	rotation_zero = np.array([0.0, 0.0, 0.0])  # degrees

	theta_orig, phi_orig = rotate_angles(rotation=rotation_zero, theta=theta_single[0], phi=phi_single[0])
	theta_batch, phi_batch = rotate_angles_batch(rotation=rotation_zero, theta=theta_single, phi=phi_single)

	# Functions return radians; ensure equality
	assert np.isclose(theta_orig, theta_batch[0])
	assert np.isclose(phi_orig, phi_batch[0])


def test_rotate_angles_batch_equivalence():
	rng = np.random.default_rng(1)
	n_users, n_paths = 20, 4
	theta = rng.uniform(0, 180, (n_users, n_paths))  # degrees
	phi = rng.uniform(0, 360, (n_users, n_paths))  # degrees
	rotation_single = np.array([30.0, 45.0, 60.0])  # degrees
	rotation_per_user = rng.uniform(-90, 90, (n_users, 3))  # degrees

	theta_rot_1 = np.zeros_like(theta, dtype=float)
	phi_rot_1 = np.zeros_like(phi, dtype=float)
	for i in range(n_users):
		theta_rot_1[i], phi_rot_1[i] = rotate_angles(rotation=rotation_single, theta=theta[i], phi=phi[i])

	theta_rot_2, phi_rot_2 = rotate_angles_batch(rotation=rotation_single, theta=theta, phi=phi)
	np.testing.assert_allclose(theta_rot_1, theta_rot_2, rtol=1e-12, atol=1e-12)
	np.testing.assert_allclose(phi_rot_1, phi_rot_2, rtol=1e-12, atol=1e-12)

	theta_rot_3 = np.zeros_like(theta, dtype=float)
	phi_rot_3 = np.zeros_like(phi, dtype=float)
	for i in range(n_users):
		theta_rot_3[i], phi_rot_3[i] = rotate_angles(rotation=rotation_per_user[i], theta=theta[i], phi=phi[i])

	theta_rot_4, phi_rot_4 = rotate_angles_batch(rotation=rotation_per_user, theta=theta, phi=phi)
	np.testing.assert_allclose(theta_rot_3, theta_rot_4, rtol=1e-12, atol=1e-12)
	np.testing.assert_allclose(phi_rot_3, phi_rot_4, rtol=1e-12, atol=1e-12)


def test_steering_vec_properties():
	panel_size = (4, 2)
	N = panel_size[0] * panel_size[1]
	vec = steering_vec(panel_size, phi=30.0, theta=20.0, spacing=0.5)
	assert vec.shape == (N, 1) or vec.shape == (N,)
	if vec.ndim == 2:
		vec = vec.ravel()
	np.testing.assert_allclose(np.linalg.norm(vec), 1.0, rtol=1e-12, atol=1e-12)


def test_ant_indices():
	"""Test antenna indices generation."""
	panel_size = [6, 4]
	ants = ant_indices(panel_size)
	assert ants.shape == (24, 3)
	
	# Check grid structure (roughly)
	ys = ants[:, 1]
	zs = ants[:, 2]
	assert len(np.unique(ys)) == 6 # y dims
	assert len(np.unique(zs)) == 4 # z dims
