# Local data mount

This directory is intentionally empty in the public repository. Put processed
tuple directories under `data/processed/` after obtaining the source data under
the applicable data-use agreement. Each configuration expects:

```text
data/processed/<dataset-name>/train_tuple.pkl
data/processed/<dataset-name>/val_tuple.pkl
data/processed/<dataset-name>/test_tuple.pkl
```

`data/processed/` is ignored by Git so protected data cannot be committed
accidentally.
