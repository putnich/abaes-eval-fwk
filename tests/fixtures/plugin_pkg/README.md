# aes-bm25-plugin

The separately installed extension used by `tests/test_registry.py` to prove the acceptance
criterion of guide section 9: a package that is not in this checkout registers a retriever and a
metric through `abaes_eval_fwk.api` alone, and is discovered only because its distribution
metadata declares the `abaes_eval_fwk.plugins` entry-point group.

It registers two components:

- `aes_bm25.retrieval.bm25` — Okapi BM25 over the calibration pool, with optional character
  n-grams, declared deterministic.
- `aes_bm25.metric.mean_absolute_error` — mean absolute error against the selected human targets,
  or an explicit `unavailable_reason` when a prediction has no reference.

Set `AES_BM25_PLUGIN_API_RANGE` to a range the running public API falls outside (for example
`>=2.0,<3.0`) to make the same installed distribution declare an incompatible plugin, which the
registry rejects before any of its components reach a registry.

`examples/extension_package` is the shorter, readable version of the same idea, and builds with
setuptools; this fixture carries an in-tree build backend only so its install needs no network.
