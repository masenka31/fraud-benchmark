# fraud-benchmark

Downloads public fraud and AML transaction datasets, normalizes them to a shared schema,
assigns temporal train/validation/test splits, and attaches a synthetic label-availability
("fraud reported at") timestamp to fraudulent transactions.

See `docs/superpowers/specs/2026-08-01-fraud-benchmark-design.md` for the full design.

## Install

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

## Kaggle credentials

Required before any dataset can be downloaded. Pick **one** of the methods below —
`kagglehub` tries them in this order and uses the first it finds.

**1. Access token file (recommended)**

Copy your Kaggle access token into `~/.kaggle/access_token` as plain text — no
extension, no JSON, no quotes. Trailing whitespace is stripped, so a trailing newline
is fine.

```zsh
mkdir -p ~/.kaggle && read -rs "TOKEN?Paste token: " && \
  printf '%s' "$TOKEN" > ~/.kaggle/access_token && \
  chmod 600 ~/.kaggle/access_token && unset TOKEN
```

Using `read` keeps the token out of your shell history. In bash, use
`read -rsp "Paste token: " TOKEN` instead.

**2. Environment variable**

```bash
export KAGGLE_API_TOKEN=<token>
```

**3. Legacy `kaggle.json`**

Kaggle Settings → **API** → **Create New Token** downloads a `kaggle.json` containing
a username and key. Save it to `~/.kaggle/kaggle.json` and `chmod 600` it, or export
its two values as `KAGGLE_USERNAME` and `KAGGLE_KEY`.

Note that the `key` inside `kaggle.json` is **not** the same credential as an access
token — do not paste it into `~/.kaggle/access_token`.

**Verify:**

```bash
.venv/bin/python -c "import kagglehub; print(kagglehub.whoami())"
```

The credentials must live in your **home** directory, not in this repository. `.gitignore`
guards against committing them, but keeping them outside the repo is safer.

IEEE-CIS additionally requires accepting its competition rules once in a browser — see
`docs/kaggle-setup.md`.

## Usage

    fraud-benchmark list
    fraud-benchmark prepare paysim
    fraud-benchmark info paysim

## Tests

    pytest

Network tests are deselected by default. Run them with `pytest -m network` — they perform
real Kaggle downloads.

## Licence

The code in this repository is MIT licensed (see `LICENSE`).

**The datasets are not.** Each carries its own terms, which you accept directly with the
upstream provider when you download it. This repository distributes no data — `data/` is
gitignored and everything is fetched at run time with your own credentials.

Nine datasets are registered: seven distinct sources, plus `ibm_ccf_subsample_fast` and
`ibm_ccf_subsample_slow`. Those two are crops of IBM CCF to a recent, fully-labelled
window, row-identical to each other and differing only in reporting delay, so a model can
be compared across delay regimes on the same data. They reuse IBM CCF's raw download.

Two of the seven sources (BankSim, SAML-D) are **NonCommercial**, and three
(PaySim, BankSim, SAML-D) are **ShareAlike**. IEEE-CIS is governed by Kaggle competition
rules rather than an open licence. See **`docs/dataset-licenses.md`** for the full table,
and check `data_license` in any prepared dataset's `dataset_card.json`.
