## Breaking Changes

We simplified APIs for user sampling/indexing, dataset trimming, and field-of-view application. Current changes use less code, less memory, and give the user less functions to worry about, while offering more (!) flexibility and functionality.


### 1) Index selection unified API

- Public per-mode index selection helpers were removed in favor of a single dispatcher:
  - Removed public methods: `get_active_idxs`, `get_linear_idxs`, `get_uniform_idxs`, `get_row_idxs`, `get_col_idxs`.
  - Use `get_idxs(mode, **kwargs)` instead. Supported modes: `active`, `linear`, `uniform`, `row`, `col`, `limits`.
  - Internals remain available for advanced composition: `_get_active_idxs`, `_get_linear_idxs`, `_get_uniform_idxs`, `_get_row_idxs`, `_get_col_idxs`.

Examples:

```python
# Active
idxs = dataset.get_idxs('active')

# Linear
idxs = dataset.get_idxs('linear', start_pos=[0,0,0], end_pos=[100,0,0], n_steps=50)

# Uniform grid
idxs = dataset.get_idxs('uniform', steps=[4,4])

# Rows / Cols
row_idxs = dataset.get_idxs('row', row_idxs=np.arange(40,60))
col_idxs = dataset.get_idxs('col', col_idxs=np.arange(10,20))

# Position limits (bounds)
idxs = dataset.get_idxs('limits', x_min=-50, x_max=50, y_min=-20, y_max=20)
```


### 2) Dataset Trimming unified API

- Removed `subset(...)`. Use `trim(idxs=...)` for subsetting users.
- FoV is applied via physical trimming that returns a new `Dataset` with out-of-FoV paths removed.
- New unified trimming API: `Dataset.trim(...)` applies trims in this order: index -> FoV -> path depth -> path type.
- `trim_by_path_depth(...)` and `trim_by_path_type(...)` are now internal; call `trim(...)` instead.

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

Note: Because trimming is physical, downstream operations are lighter and reflect only in-FoV paths.

### 3) FoV simplification & integration in trim

- Removed: `Dataset.apply_fov(...)`, and FoV-cached variables/mask (`_ao*_rot_fov`, `_fov_mask`).
- Computations now rely on rotated angles only; FoV is no longer a lazy cache.
- `los` and `num_paths` reflect the currently present (possibly trimmed) paths.
- new simpler call: `dataset.trim(bs_fov | ue_fov)`. Example below.

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

- FoV arguments use degrees as `[horizontal, vertical]`. `None` means full FoV (no trim).
