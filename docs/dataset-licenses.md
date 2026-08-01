# Dataset licences

The MIT licence in `LICENSE` covers **this repository's code only**. Each dataset carries its
own terms, and you accept those directly with the upstream provider when you download.

This repository distributes **no data**. `data/` is gitignored, and every dataset is fetched
at run time using your own Kaggle credentials. Keep it that way: committing processed output
or publishing a derived dataset would put you under the terms below rather than outside them.

Licence values were read from the Kaggle API on 2026-08-01 (`licenseName` field), except
where noted. **Verify before relying on any of this for a publication or a commercial
product** — upstream terms can change, and this table is a convenience, not legal advice.

| Dataset | Adapter | Licence | Commercial use | Notes |
|---|---|---|---|---|
| PaySim | `paysim` | CC BY-SA 4.0 | Yes | ShareAlike: derived datasets must carry the same licence |
| BankSim | `banksim` | **CC BY-NC-SA 4.0** | **No** | NonCommercial **and** ShareAlike |
| Sparkov (Shenoy) | `sparkov` | CC0 1.0 | Yes | Public domain dedication, no attribution required |
| IBM CCF (Altman) | `ibm_ccf` | Apache-2.0 | Yes | Stated in the dataset description body, not the Kaggle licence field |
| SAML-D | `saml_d` | **CC BY-NC-SA 4.0** | **No** | NonCommercial **and** ShareAlike |
| Amaretto | `amaretto` | MIT | Yes | From the GitHub repository licence |
| IEEE-CIS / Vesta | `ieee_cis` | Kaggle competition rules | **Check the rules** | Not an SPDX licence. Must be accepted in a browser; competition terms generally prohibit redistribution |

## What this means in practice

**Two datasets are NonCommercial.** BankSim and SAML-D cannot be used in a commercial
product. If the benchmark is meant to be usable by industry, either exclude those two or
make the restriction prominent so users can opt out of them.

**Three datasets are ShareAlike.** PaySim, BankSim, and SAML-D require that a *derived
dataset* be released under the same licence. This constrains publishing processed outputs,
not running the pipeline locally, and not the code in this repo.

**IEEE-CIS is the most restrictive and the least well-defined.** Its terms live in the
competition rules at <https://www.kaggle.com/c/ieee-fraud-detection/rules>, which you must
accept while signed in. Read them before using this dataset in anything beyond private
research.

Each prepared dataset records its licence in `data/processed/<name>/dataset_card.json` under
`data_license`, so the terms travel with the output rather than living only in this file.
