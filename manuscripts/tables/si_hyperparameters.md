# Supplementary Table S1 · Method hyperparameters

_All methods are evaluated with the same hyperparameter set across_
_every case; this table is auto-generated from_
`benchmark/hyperparameters.py` _and locked by the CI test_
`benchmark/tests/test_hyperparameters.py`.

| Method | Sampler | Surrogate / kernel | Hyperparameters |
|---|---|---|---|
| `akmcs` | lhs_opt | gp | output_idx=0, threshold=0e+00, direction=above, n_init_frac=0.5, n_candidates=512 |
| `botorch_qei` | lhs_opt | botorch_singletaskgp | n_init_frac=0.6, n_candidates=512, output_idx=0 |
| `botorch_saasbo` | lhs_opt | botorch_saas | n_init_frac=0.9, n_candidates=64, output_idx=0, warmup_steps=4, num_samples=4, thinning=1 |
| `ego_ei` | lhs_opt | gp | n_init_frac=0.6, n_candidates=256, output_idx=0 |
| `halton+gp` | halton | gp | length_scale_init=1.0, n_restarts=3 |
| `lhs+gp` | lhs | gp | length_scale_init=1.0, n_restarts=3 |
| `lhs_opt+gb` | lhs_opt | gb | max_iter=200, learning_rate=0.1, max_depth=6 |
| `lhs_opt+gp` | lhs_opt | gp | length_scale_init=1.0, n_restarts=3 |
| `lhs_opt+gp_rbf` | lhs_opt | gp_rbf | length_scale_init=1.0, n_restarts=3 |
| `lhs_opt+knn` | lhs_opt | knn | n_neighbors=5, weights=distance |
| `lhs_opt+mlp` | lhs_opt | mlp | hidden_layer_sizes=(64,64), max_iter=2000, alpha=1e-04 |
| `lhs_opt+poly3` | lhs_opt | poly | degree=3, alpha=1e-03 |
| `lhs_opt+poly5` | lhs_opt | poly_chaos | degree=5, alpha=0.01 |
| `lhs_opt+rbf` | lhs_opt | rbf | kernel=thin_plate_spline, smoothing=0e+00 |
| `lhs_opt+rf` | lhs_opt | rf | n_estimators=200, max_depth=None |
| `lhs_opt+svr` | lhs_opt | svr | C=1.0, epsilon=0.05, gamma=scale |
| `max_variance` | lhs_opt | gp | n_init_frac=0.6, n_candidates=256 |
| `random+gp` | random | gp | length_scale_init=1.0, n_restarts=3 |
| `random+linear` | random | linear | -- |
| `sobol+gp` | sobol | gp | length_scale_init=1.0, n_restarts=3 |
