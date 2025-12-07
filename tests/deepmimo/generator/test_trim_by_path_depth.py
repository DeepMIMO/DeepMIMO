import pytest
import deepmimo as dm


@pytest.mark.skip(reason="Requires local scenarios and plotting; skipped in CI")
def test_trim_by_path_depth_demo():
	# %% Trimming by path depth test
	dataset = dm.load("asu_campus_3p5")
	dataset_t = dataset.trim_by_path_depth(1)

	# Only test if it runs, don't plot in CI
	# dataset.plot_coverage(dataset.los, title="Full dataset")
	# dataset_t.plot_coverage(dataset_t.los, title="Trimmed dataset")

