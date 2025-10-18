## Breaking Changes

### FoV handling refactor (trim-based)

- Removed: `Dataset.apply_fov(...)`, lazy FoV-cached angles and FoV mask (`_ao*_rot_fov`, `_ao*_rot_fov`, `_fov_mask`).
- Change: FoV is now applied via physical trimming that returns a new `Dataset` with out-of-FoV paths removed.
- Affected computations now use rotated angles only (array response, antenna-gain power). `los` and `num_paths` operate on the currently trimmed data.
- New: Unified trimming API `Dataset.trim(...)` that composes multiple trims in this order: index -> FoV -> path depth -> path type.
- Deprecation: `trim_by_path_depth(...)` and `trim_by_path_type(...)` are now internal; call `trim(...)` instead. Thin wrappers remain for backward compatibility but will be removed in a future release.

#### Migrating from apply_fov

Before:

```python
# Old (removed)
dataset.apply_fov(bs_fov=[90, 90], ue_fov=[120, 90])
channel = dataset.compute_channels()
```

After:

```python
# New (returns a new dataset with paths physically trimmed)
dataset_t = dataset.trim(bs_fov=[90, 90], ue_fov=[120, 90])
channel = dataset_t.compute_channels()
```

#### Migrating from trim_by_path_depth & trim_by_path_type

Combine trims efficiently in one call:

```python
dataset_t = dataset.trim(
    idxs=np.arange(0, dataset.n_ue, 5),  # optional UE subset first
    bs_fov=[90, 90],                      # FoV at BS (UE FoV optional)
    path_depth=1,                         # keep at most 1 interaction
    path_types=['LoS', 'R']               # filter by interaction types
)
```

Notes:
- FoV arguments use degrees as `[horizontal, vertical]`. `None` means full FoV (no trim).
- Because trimming is physical, downstream operations are lighter and reflect only in-FoV paths.
