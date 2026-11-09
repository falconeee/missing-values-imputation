# Missing-value imputation for MSCVAE

Experiments on imputing missing values and sensor outliers **before** prediction with a multivariate anomaly detection model (MSCVAE). The tests use data from two real industrial processes, **anonymized** as *Process A* and *Process B*.

The question is: when a TAG has no data (NaN) or a sensor fails with an absurd value, which imputation strategy rebuilds the signal best without raising false alarms and without hiding a genuine process fault?

## Structure

```
.
├── main_processo_A_MSCVAE.ipynb   # experiment on Process A
├── main_processo_B_MSCVAE.ipynb   # experiment on Process B
├── src/
│   ├── ppd.py          # dataset loading, subsystem TAGs, removal of shutdown/transient periods
│   ├── MSCVAE.py       # MSCVAE model (ConvLSTM + attention + VAE) and SPOT/POT threshold
│   ├── imputation.py   # imputation experiment: training, outliers, gaps, metrics and plots
│   ├── model.py        # train_imputer: trains the IterativeImputer in the deployment format
│   └── utils.py        # anomaly index plot (graph_predict)
├── data/               # datasets (not versioned)
├── models/             # generated imputer .pkl files (not versioned)
├── pyproject.toml      # dependencies (uv)
└── uv.lock
```

## Installation

Python 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # creates the .venv from pyproject.toml + uv.lock
source .venv/bin/activate

python -m ipykernel install --user --name missing-values-imputation --display-name "Python (missing-values-imputation)"
```

torch comes from PyPI: on macOS it uses MPS and, on Linux, the default wheel already includes CUDA (it needs a compatible NVIDIA driver). To pin another build (CPU only or another CUDA version), configure the PyTorch index with `[[tool.uv.index]]` + `[tool.uv.sources]` in `pyproject.toml`.

Open the notebooks **from the repository root**, because they import `from src import ...` and read `data/`. MSCVAE uses the GPU (CUDA or MPS) if available. The imputers run on the CPU.

## Data

The data is not versioned (`data/` is in `.gitignore`). Each process needs two files in `data/`:

| File | Content |
|---|---|
| `processo_X.parquet` | One `Timestamp` column (1-min sampling) and one numeric column per TAG (`TAG_001`, `TAG_002`, …). |
| `processo_X_subsistema.csv` | `;` separator, columns `VARIAVEL;DESC;SISTEMA`: the TAG name, its description and its subsystem. |

The TAGs, descriptions and subsystems were anonymized. The values and timestamps are the original ones.

## Notebook flow

Both notebooks have the same structure. Only the process parameters change.

1. **Data loading**: `ppd.load_dataset_principal` drops columns, converts to numeric and sorts by time. `ppd.set_tags_config` matches the dataset with the subsystem file.
2. **ON/OFF**: `ppd.drop_transitorio_desligado` removes the periods with the equipment off, with a margin before and after (transient). The reference is one TAG (`pp_var_ref_desligado`) and a threshold.
3. **Training periods**: one or more normal-operation intervals. Repeating a period in `df_train_list` increases its weight in training.
4. **MSCVAE**: `MSCVAE().fit(df_train_list, gain, epochs)`. The threshold comes from SPOT (Peaks-Over-Threshold) and is multiplied by `gain`.
5. **Imputer training**: uses the same periods as MSCVAE, without duplicates. First, `fimp.limpar_falhas_treino` removes rows with gross sensor failures. `fimp.treinar_imputers` trains one `IterativeImputer` per estimator with the deployment `train_imputer`, which includes a temporal holdout with 10–240 min gaps, one TAG at a time.
6. **Extreme outliers in the test set**: `fimp.detectar_outliers` flags the real sensor failures in the test period, whether an absurd value or a full-scale reading. For cross-checking, `fimp.listar_desvios` lists the process deviations that are **not** treated.
7. **Synthetic gaps**: `fimp.injetar_lacunas` creates NaN gaps in a copy of the test set, away from the real outliers. Since the true value is known, the error can be measured.
8. **Imputation**: `fimp.aplicar_imputacao` applies the production rule (below) with each method.
9. **Metrics**: error on the synthetic gaps (`fimp.metricas_lacunas`) and effect on the anomaly index (`fimp.resumo_efeito_predicao`), with before/after plots per TAG.

## Compared methods

| Method | Description |
|---|---|
| `ffill (baseline)` | `ffill`/`bfill` on every gap. |
| `BayesianRidge` | Linear `IterativeImputer`. |
| `HistGB (produção)` | `IterativeImputer` with `HistGradientBoostingRegressor` (the deployment's default estimator). |
| `ExtraTrees` | `IterativeImputer` with a shallow `ExtraTreesRegressor`, to keep the `.pkl` small. |
| `XGBoost` | `IterativeImputer` with `XGBRegressor`. |
| `LightGBM` | `IterativeImputer` with `LGBMRegressor`. |

The prediction scenarios also include the **Reference** (original data, without synthetic gaps) and **No imputation (dropna)**, which drops the rows with NaN.

To test another estimator, edit the `estimadores` dict in the notebook.

## Main rules

**Extreme outlier (sensor failure).** Only sensor failures are imputed. Process deviations, even large ones, are exactly what MSCVAE needs to detect and are passed to it raw. There are two rules.

*Absurd value.* A value is flagged when it meets **both** conditions:

1. `|x − median| > FATOR_EXTREMO × max(p1–p99 range, |median|)` of the training set. The value is absurd for the TAG's level.
2. `|x[t] − x[t−1]| > SALTO_EXTREMO × p99.9` of the training jumps. The value was reached in a jump the sensor never made.

After it fires, the flag carries forward until the value returns to the normal range (± `K_RETORNO` ranges). A process deviation evolves gradually and does not meet condition 2. This rule catches the sensor that jumps to ~6e6 or to 10× its range.

*Full scale.* Applies only to the TAGs configured in `LIMITES_ESCALA = {tag: (lower_limit, upper_limit)}`, with `None` on the side without a limit. It covers the failure that stays close to the operating range, where the previous rule does not reach: a sensor at ~1000 that drops to −58 has a factor of ~1. The value is flagged when it meets **both** conditions:

1. It is in the limit zone: within `TOL_ESCALA` of the distance between the normal range (p1/p99) and the limit.
2. It got there abruptly: within 2 samples, a jump > `SALTO_ESCALA` × p99.9 of the training jumps. A process that rises gradually until it saturates the instrument is a real deviation and stays raw.

After it fires, the flag continues while the value is closer to the limit than to the normal range. This includes the interpolated ramp the historian records after the failure, but not an equipment stop right after a spike at the limit.

Ideally, `LIMITES_ESCALA` is filled with the instruments' real ranges. `fimp.candidatos_fundo_escala` lists the values at which each sensor stopped after repeated abrupt exits, as a starting point, but it also returns false candidates (integer TAGs, stops).

In both rules, what comes before the trigger is never flagged, just like in the runtime, which only sees the past. That is why the first sample of the transition (the average of the minute when the sensor froze) stays raw.

**Application rule (same as production):**

1. an outlier becomes NaN;
2. a gap shorter than `START_MINUTES` with a valid previous value is filled with `ffill`;
3. the rest goes to the `IterativeImputer`;
4. a row with ≥ `MAX_BAD_RATIO` of its variables invalid is not reconstructed.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `FATOR_FALHA_GROSSEIRA` | 50 | Cleaning of the imputer's training data: \|x − median\| > factor × robust scale. |
| `FATOR_EXTREMO` | 3.0 | Condition 1 of the extreme outlier. |
| `SALTO_EXTREMO` | 20.0 | Condition 2 of the extreme outlier. |
| `K_RETORNO` | 1.0 | Return band that ends the absurd-value flag. |
| `LIMITES_ESCALA` | per process | Instrument limits per TAG for the full-scale rule. |
| `TOL_ESCALA` | 0.05 | Limit zone: fraction of the distance between the normal range and the limit. |
| `SALTO_ESCALA` | 3.0 | Minimum jump (× p99.9), within 2 samples, to trigger the full-scale rule. |
| `START_MINUTES` | 10 | Gaps shorter than this (with an anchor) go to `ffill`. |
| `MAX_BAD_RATIO` | 0.7 | Fraction of invalid variables above which the row is not reconstructed. |
| `LACUNAS_POR_TAG` | 3 | Synthetic gaps per TAG. |
| `DUR_MIN`, `DUR_MAX` | 10, 240 | Duration of the synthetic gaps (min). |
| `FRAC_SIMULTANEAS` | 0.3 | Fraction of the gaps that take down 2–3 TAGs at once. |
| `PASSO_AMOSTRAGEM` | 1 | Subsampling of the imputers' training data (> 1 speeds it up). |

## Main results

The numbers below come from the run before the full-scale rule. With it, Process A now imputes 83 sensor-failure points in the test set (none before), and Process B does not change. MAE and RMSE are normalized by the TAG's training standard deviation, to compare TAGs on different scales. The values are medians across TAGs.

### Synthetic gaps in the test set

| Method | Proc. A: norm. MAE | Proc. A: R² | Proc. B: norm. MAE | Proc. B: R² |
|---|---|---|---|---|
| XGBoost | **0.184** | **0.763** | 0.244 | 0.823 |
| LightGBM | 0.190 | 0.728 | **0.221** | 0.814 |
| HistGB (production) | 0.197 | 0.693 | 0.253 | **0.835** |
| ExtraTrees | 0.267 | 0.606 | 0.260 | 0.798 |
| BayesianRidge | 0.298 | 0.520 | 0.240 | 0.824 |
| ffill (baseline) | 0.584 | −0.126 | 0.284 | 0.683 |

### Effect on the MSCVAE prediction

| Scenario | Proc. A: coverage in the gaps | Proc. A: \|Δ phi/threshold\| | Proc. B: coverage in the gaps | Proc. B: % alarms on real outliers |
|---|---|---|---|---|
| Reference | 100% | — | 100% | 36.0% |
| No imputation (dropna) | 5.7% | 4.95 | 6.3% | 36.1% |
| ffill (baseline) | 100% | 0.18 | 100% | 8.0% |
| IterativeImputer (5 methods) | 100% | 0.08–0.09 | 100% | 7.9–8.0% |

### Conclusions

- **Without imputation, monitoring is lost.** `dropna` keeps only ~6% of the instants inside the gaps.
- **IterativeImputer beats ffill.** On Process A, the normalized MAE drops from 0.58 to 0.18–0.20 with the tree methods, and the deviation of the anomaly index from the reference is halved (0.18 → ~0.08). On Process B the difference is smaller (MAE from 0.28 to 0.22–0.26).
- **Imputing the sensor outlier removes false alarms.** On Process B, alarms on stretches with sensor failures drop from 36% to ~8% with any method.
- **Among the tree methods, the difference is small.** `HistGB (production)` is close to the best and has the smallest `.pkl` among the trees (1.7–3.7 MB, versus 6–8 MB for XGBoost/LightGBM). This matters in deployment, because the runtime downloads the imputer every cycle.
- **The training holdout shows poorly predictable TAGs** (about 20 per process with R² < 0.5). For those TAGs, imputation should be used with caution.

## Adding a new process

1. Put `processo_X.parquet` and `processo_X_subsistema.csv` in `data/`.
2. Copy one of the notebooks and adjust:
   - `process_name`;
   - the columns dropped on load;
   - the ON/OFF parameters (`pp_var_ref_desligado`, threshold and cuts);
   - the instruments' scale limits (`LIMITES_ESCALA`);
   - the training periods;
   - the MSCVAE `gain`.
3. Run the whole notebook. The imputers are saved in `models/`.
