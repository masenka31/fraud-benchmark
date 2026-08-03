# Leakage ablation

Average precision. Base rates here run from 0.10% to 0.52%, so a
negative-class-dominated metric would stay high for a model with no useful
precision and hide exactly what this table measures.

`gap` is leaky minus clean -- how much of the score the leaky columns supply.
Sparkov has no leaky columns, so its gap is a pure measurement of noise and
calibrates how much of every other row to believe.

| dataset | split | labels | model | leaky | clean | gap | seed sd |
|---|---|---|---|---:|---:|---:|---:|
| ibm_ccf | test | n/a | trivial_rule | 0.764 | - | - | - |
| ibm_ccf | test | oracle | logistic | 0.008 | 0.012 | -0.004 | - |
| ibm_ccf | test | oracle | xgboost | 0.041 | 0.019 | 0.022 | 0.006 |
| ibm_ccf | val | n/a | trivial_rule | 0.841 | - | - | - |
| ibm_ccf | val | oracle | logistic | 0.007 | 0.010 | -0.004 | - |
| ibm_ccf | val | oracle | xgboost | 0.011 | 0.007 | 0.004 | 0.003 |
| saml_d | test | censored | logistic | 0.075 | 0.064 | 0.012 | - |
| saml_d | test | censored | xgboost | 0.486 | 0.452 | 0.033 | 0.003 |
| saml_d | test | oracle | logistic | 0.072 | 0.064 | 0.009 | - |
| saml_d | test | oracle | xgboost | 0.500 | 0.477 | 0.024 | 0.010 |
| saml_d | val | censored | logistic | 0.076 | 0.064 | 0.012 | - |
| saml_d | val | censored | xgboost | 0.481 | 0.454 | 0.027 | 0.003 |
| saml_d | val | oracle | logistic | 0.075 | 0.066 | 0.009 | - |
| saml_d | val | oracle | xgboost | 0.508 | 0.484 | 0.024 | 0.006 |
| sparkov | test | censored | logistic | 0.303 | 0.303 | 0.000 | - |
| sparkov | test | censored | xgboost | 0.904 | 0.904 | 0.000 | 0.002 |
| sparkov | test | oracle | logistic | 0.303 | 0.303 | 0.000 | - |
| sparkov | test | oracle | xgboost | 0.909 | 0.909 | 0.000 | 0.001 |
| sparkov | val | censored | logistic | 0.431 | 0.431 | 0.000 | - |
| sparkov | val | censored | xgboost | 0.951 | 0.951 | 0.000 | 0.004 |
| sparkov | val | oracle | logistic | 0.431 | 0.431 | 0.000 | - |
| sparkov | val | oracle | xgboost | 0.954 | 0.954 | 0.000 | 0.002 |

## Cells with no results

These are in the grid (`fraud_benchmark.ablation.grid`) but have not been
run, so the table above is incomplete. Submit them with
`scripts/slurm/jobs/submit_all.sh`.

- `sparkov_slow` / leaky / oracle
- `sparkov_slow` / leaky / censored
- `sparkov_slow` / clean / oracle
- `sparkov_slow` / clean / censored
