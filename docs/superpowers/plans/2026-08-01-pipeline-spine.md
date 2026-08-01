# Pipeline Spine Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete processing spine — config, schema, Kaggle fetching, adapter registry, temporal splitting, and CLI — and prove it end-to-end by producing a real processed PaySim dataset.

**Architecture:** An adapter registry plus shared pipeline stages. Each dataset is a small class exposing a source spec and a `to_canonical()` method; everything downstream (validate, split, write) is shared code identical for all datasets. This plan implements the shared machinery and exactly one adapter (PaySim) to prove the spine works.

**Tech Stack:** Python 3.13, pandas, pyarrow (parquet), kagglehub 1.0.2, PyYAML, pytest, argparse.

**Scope:** Plans 2 and 3 follow. Plan 2 adds the remaining six adapters and the git source type. Plan 3 adds the label-delay stage. Do not implement those here.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, CLI entry point, pytest config |
| `.gitignore` | Exclude `data/`, venv, caches, credentials |
| `README.md` | What the repo is, how to install and run |
| `docs/kaggle-setup.md` | Human-followed credential setup guide |
| `configs/default.yaml` | Default paths, split ratios, per-dataset options |
| `src/fraud_benchmark/schema.py` | Canonical column definitions, validation, column ordering |
| `src/fraud_benchmark/config.py` | Load and merge YAML config into dataclasses |
| `src/fraud_benchmark/sources.py` | Source specs and fetching, with actionable error translation |
| `src/fraud_benchmark/datasets/base.py` | `DatasetAdapter` base class and registry |
| `src/fraud_benchmark/datasets/paysim.py` | PaySim adapter |
| `src/fraud_benchmark/splitting.py` | Tie-safe temporal split assignment |
| `src/fraud_benchmark/pipeline.py` | Stage orchestration, dataset card, atomic write |
| `src/fraud_benchmark/cli.py` | `list` / `prepare` / `info` commands |
| `tests/*` | Unit tests, no network |

---

### Task 1: Project skeleton

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`
- Create: `src/fraud_benchmark/__init__.py`, `src/fraud_benchmark/datasets/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
data/
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
.pytest_cache/
kaggle.json
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "fraud-benchmark"
version = "0.1.0"
description = "Download, normalize, split, and label-delay public fraud and AML datasets"
requires-python = ">=3.11"
dependencies = [
    "kagglehub>=1.0.2",
    "pandas>=2.2",
    "pyarrow>=16",
    "pyyaml>=6",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
fraud-benchmark = "fraud_benchmark.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["network: requires internet access and Kaggle credentials"]
addopts = "-m 'not network'"
```

- [ ] **Step 3: Create the package files**

`src/fraud_benchmark/__init__.py`:

```python
"""Fraud and AML benchmark dataset preparation."""

__version__ = "0.1.0"
```

`src/fraud_benchmark/datasets/__init__.py`:

```python
"""Per-dataset adapters."""
```

`tests/__init__.py`: empty file.

- [ ] **Step 4: Create `README.md`**

```markdown
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

See `docs/kaggle-setup.md`. Required before any dataset can be downloaded.

## Usage

    fraud-benchmark list
    fraud-benchmark prepare paysim
    fraud-benchmark info paysim

## Tests

    pytest
```

- [ ] **Step 5: Create the venv and install**

Run:

```bash
python -m venv .venv && .venv/bin/pip install -q -e ".[dev]" && .venv/bin/python -c "import fraud_benchmark, kagglehub, pandas, pyarrow, yaml; print('ok', fraud_benchmark.__version__)"
```

Expected: `ok 0.1.0`

- [ ] **Step 6: Verify pytest runs**

Run: `.venv/bin/pytest`
Expected: exit code 5, "no tests ran". That is success at this stage.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: project skeleton, packaging, and dev environment"
```

---

### Task 2: Kaggle credential setup (human step)

**Files:**
- Create: `docs/kaggle-setup.md`

This task requires the user to act. Do not attempt to create credentials on their behalf.

- [ ] **Step 1: Write `docs/kaggle-setup.md`**

````markdown
# Kaggle credentials setup

`kagglehub` needs an API token before it can download anything.

## 1. Create the token

1. Sign in at <https://www.kaggle.com>.
2. Go to <https://www.kaggle.com/settings/account>.
3. Under **API**, click **Create New Token**.
4. Your browser downloads `kaggle.json`, containing your username and key.

## 2. Install the token

Place the file where kagglehub looks for it, and restrict its permissions:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Alternatively, export the credentials as environment variables instead of using a file:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

A third option: set `KAGGLE_CONFIG_DIR` to a directory containing `kaggle.json`.

**Never commit `kaggle.json`.** It is listed in `.gitignore`.

## 3. Verify

```bash
.venv/bin/python -c "import kagglehub; print(kagglehub.whoami())"
```

Expected: a dict containing your username. If it raises instead, the token is missing
or wrong — repeat step 1.

## 4. Accept the IEEE-CIS competition rules

IEEE-CIS is a **competition**, not a plain dataset. The API returns `403 Forbidden` until
you have accepted its rules once, in a browser, while signed in:

<https://www.kaggle.com/c/ieee-fraud-detection/rules>

Click **I Understand and Accept**. This is a one-time action per Kaggle account and cannot
be done through the API. Only needed before preparing `ieee_cis` (Plan 2).
````

- [ ] **Step 2: Ask the user to follow the guide**

Stop and ask the user to complete steps 1, 2, and 4 of `docs/kaggle-setup.md`. Wait for
their confirmation. Do not proceed to step 3 until they confirm.

- [ ] **Step 3: Verify credentials work**

Run: `.venv/bin/python -c "import kagglehub; print(kagglehub.whoami())"`
Expected: a dict containing the user's Kaggle username.

If this fails, report the exact error to the user and stop. Every later task depends on it.

- [ ] **Step 4: Commit**

```bash
git add docs/kaggle-setup.md
git commit -m "docs: Kaggle credential setup guide"
```

---

### Task 3: Canonical schema

**Files:**
- Create: `src/fraud_benchmark/schema.py`
- Test: `tests/test_schema.py`

Background: adapters produce the four required columns. `split` is added by the pipeline
and `reported_at` by Plan 3, so neither is required at validation time.

- [ ] **Step 1: Write the failing tests**

`tests/test_schema.py`:

```python
import pandas as pd
import pytest

from fraud_benchmark.schema import SchemaError, order_columns, validate_canonical


def make_valid_frame():
    return pd.DataFrame(
        {
            "amount": pd.Series([1.0, 2.0], dtype="float64"),
            "event_time": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "extra": ["a", "b"],
            "entity_id": pd.Series(["c1", "c2"], dtype="string"),
            "is_fraud": pd.Series([True, False], dtype="bool"),
        }
    )


def test_validate_accepts_a_valid_frame():
    validate_canonical(make_valid_frame())


def test_validate_rejects_missing_column():
    df = make_valid_frame().drop(columns=["entity_id"])
    with pytest.raises(SchemaError, match="entity_id"):
        validate_canonical(df)


def test_validate_rejects_wrong_dtype():
    df = make_valid_frame()
    df["amount"] = df["amount"].astype("int64")
    with pytest.raises(SchemaError, match="amount"):
        validate_canonical(df)


def test_validate_rejects_nulls_in_required_column():
    df = make_valid_frame()
    df.loc[0, "entity_id"] = None
    with pytest.raises(SchemaError, match="null"):
        validate_canonical(df)


def test_validate_rejects_empty_frame():
    df = make_valid_frame().iloc[:0]
    with pytest.raises(SchemaError, match="empty"):
        validate_canonical(df)


def test_validate_rejects_implausible_timestamps():
    df = make_valid_frame()
    df["event_time"] = pd.to_datetime(["1900-01-01", "1900-01-02"])
    with pytest.raises(SchemaError, match="implausible"):
        validate_canonical(df)


def test_order_columns_puts_core_first_and_keeps_the_rest():
    ordered = order_columns(make_valid_frame())
    assert list(ordered.columns) == [
        "event_time",
        "entity_id",
        "amount",
        "is_fraud",
        "extra",
    ]


def test_order_columns_includes_split_when_present():
    df = make_valid_frame()
    df["split"] = pd.Series(["train", "test"], dtype="category")
    ordered = order_columns(df)
    assert list(ordered.columns)[:5] == [
        "event_time",
        "entity_id",
        "amount",
        "is_fraud",
        "split",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_schema.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'fraud_benchmark.schema'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/schema.py`:

```python
"""The canonical schema every processed dataset conforms to."""

from __future__ import annotations

import pandas as pd

# Core columns, in the order they appear in the output. `split` is added by the
# pipeline and `reported_at` by the label-delay stage, so both may be absent.
CORE_ORDER = ("event_time", "entity_id", "amount", "is_fraud", "split", "reported_at")

# Columns every adapter must produce, mapped to their required dtype.
REQUIRED_DTYPES = {
    "event_time": "datetime64[ns]",
    "entity_id": "string",
    "amount": "float64",
    "is_fraud": "bool",
}

# Any real transaction dataset falls inside this window. Outside it, the adapter
# almost certainly mis-parsed a relative offset.
MIN_PLAUSIBLE_TIME = pd.Timestamp("1990-01-01")
MAX_PLAUSIBLE_TIME = pd.Timestamp("2050-01-01")


class SchemaError(ValueError):
    """Raised when a frame does not conform to the canonical schema."""


def validate_canonical(df: pd.DataFrame) -> None:
    """Raise SchemaError if `df` is not a valid canonical frame."""
    if len(df) == 0:
        raise SchemaError("dataset is empty")

    missing = [c for c in REQUIRED_DTYPES if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {', '.join(missing)}")

    for column, expected in REQUIRED_DTYPES.items():
        actual = str(df[column].dtype)
        if actual != expected:
            raise SchemaError(
                f"column {column!r} has dtype {actual!r}, expected {expected!r}"
            )
        null_count = int(df[column].isna().sum())
        if null_count:
            raise SchemaError(f"column {column!r} has {null_count} null value(s)")

    times = df["event_time"]
    if times.min() < MIN_PLAUSIBLE_TIME or times.max() > MAX_PLAUSIBLE_TIME:
        raise SchemaError(
            f"column 'event_time' has implausible range "
            f"{times.min()} to {times.max()}; expected between "
            f"{MIN_PLAUSIBLE_TIME.date()} and {MAX_PLAUSIBLE_TIME.date()}"
        )


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with core columns first, then all remaining columns unchanged."""
    core = [c for c in CORE_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in core]
    return df[core + rest]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_schema.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/fraud_benchmark/schema.py tests/test_schema.py
git commit -m "feat: canonical schema definition and validation"
```

---

### Task 4: Configuration

**Files:**
- Create: `src/fraud_benchmark/config.py`, `configs/default.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Create `configs/default.yaml`**

```yaml
paths:
  raw: data/raw
  processed: data/processed

split:
  ratios: [0.8, 0.1, 0.1]

datasets:
  # PaySim's `step` column is an hour offset, not a real date, so it is
  # anchored to this start date.
  paysim:
    start_date: "2023-01-01"
```

- [ ] **Step 2: Write the failing tests**

`tests/test_config.py`:

```python
import pytest

from fraud_benchmark.config import Config, ConfigError, load_config


def test_load_default_config():
    config = load_config()
    assert config.split_ratios == (0.8, 0.1, 0.1)
    assert config.raw_dir.name == "raw"
    assert config.processed_dir.name == "processed"


def test_default_config_has_paysim_start_date():
    config = load_config()
    assert config.for_dataset("paysim")["start_date"] == "2023-01-01"


def test_for_dataset_returns_empty_dict_for_unknown_dataset():
    assert load_config().for_dataset("nonexistent") == {}


def test_user_config_overrides_defaults(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "split:\n"
        "  ratios: [0.6, 0.2, 0.2]\n"
        "datasets:\n"
        "  paysim:\n"
        "    start_date: '2020-05-01'\n"
    )
    config = load_config(path)
    assert config.split_ratios == (0.6, 0.2, 0.2)
    assert config.for_dataset("paysim")["start_date"] == "2020-05-01"
    # Unspecified keys still come from the defaults.
    assert config.raw_dir.name == "raw"


def test_ratios_must_sum_to_one(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("split:\n  ratios: [0.5, 0.2, 0.2]\n")
    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(path)


def test_ratios_must_have_three_entries(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("split:\n  ratios: [0.8, 0.2]\n")
    with pytest.raises(ConfigError, match="three"):
        load_config(path)


def test_config_is_frozen():
    config = load_config()
    with pytest.raises(Exception):
        config.split_ratios = (0.5, 0.25, 0.25)
    assert isinstance(config, Config)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'fraud_benchmark.config'`

- [ ] **Step 4: Write the implementation**

`src/fraud_benchmark/config.py`:

```python
"""Configuration loading. Defaults live in configs/default.yaml."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass(frozen=True)
class Config:
    raw_dir: Path
    processed_dir: Path
    split_ratios: tuple[float, float, float]
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)

    def for_dataset(self, name: str) -> dict[str, Any]:
        """Options for one dataset, or an empty dict if it has none."""
        return self.datasets.get(name, {})


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_ratios(ratios: Any) -> tuple[float, float, float]:
    if not isinstance(ratios, list) or len(ratios) != 3:
        raise ConfigError(f"split.ratios must be a list of three numbers, got {ratios!r}")
    values = tuple(float(r) for r in ratios)
    if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ConfigError(f"split.ratios must sum to 1, got {sum(values)}")
    if any(r <= 0 for r in values):
        raise ConfigError(f"split.ratios must all be positive, got {values}")
    return values  # type: ignore[return-value]


def load_config(path: Path | str | None = None) -> Config:
    """Load the default config, overlaying `path` on top of it when given."""
    with open(DEFAULT_CONFIG_PATH) as handle:
        data = yaml.safe_load(handle) or {}

    if path is not None:
        with open(path) as handle:
            data = _deep_merge(data, yaml.safe_load(handle) or {})

    paths = data.get("paths", {})
    return Config(
        raw_dir=Path(paths["raw"]),
        processed_dir=Path(paths["processed"]),
        split_ratios=_validate_ratios(data.get("split", {}).get("ratios")),
        datasets=data.get("datasets") or {},
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/fraud_benchmark/config.py configs/default.yaml tests/test_config.py
git commit -m "feat: YAML configuration with per-dataset overrides"
```

---

### Task 5: Source fetching

**Files:**
- Create: `src/fraud_benchmark/sources.py`
- Test: `tests/test_sources.py`

Background, verified against kagglehub 1.0.2:
- `kagglehub.dataset_download(handle, *, force_download=False, output_dir=None) -> str`
- `kagglehub.competition_download(handle, *, force_download=False, output_dir=None) -> str`
- `output_dir` bypasses the shared cache and downloads straight into our `data/raw/<name>/`.
- Relevant exceptions: `kagglehub.exceptions.CredentialError`, `.UnauthenticatedError`,
  `.KaggleApiHTTPError` (subclasses `requests.HTTPError`, so `.response.status_code` exists).

The 403-on-competition case gets its own message because Kaggle's generic auth error is
actively misleading there: the credentials are fine, the rules simply have not been accepted.

The git source type is deliberately **not** implemented here — it is only needed for
Amaretto in Plan 2.

- [ ] **Step 1: Write the failing tests**

`tests/test_sources.py`:

```python
import requests
import pytest
from kagglehub.exceptions import CredentialError, KaggleApiHTTPError

from fraud_benchmark.sources import (
    FetchError,
    KaggleCompetition,
    KaggleDataset,
    fetch,
)


def test_kaggle_dataset_download_is_called_with_output_dir(tmp_path, monkeypatch):
    calls = {}

    def fake_download(handle, *, force_download=False, output_dir=None):
        calls.update(handle=handle, force=force_download, out=output_dir)
        (tmp_path / "dest").mkdir(parents=True, exist_ok=True)
        (tmp_path / "dest" / "data.csv").write_text("a,b\n1,2\n")
        return str(tmp_path / "dest")

    monkeypatch.setattr("fraud_benchmark.sources.kagglehub.dataset_download", fake_download)

    result = fetch(KaggleDataset("ealaxi/paysim1"), tmp_path / "dest")

    assert result == tmp_path / "dest"
    assert calls["handle"] == "ealaxi/paysim1"
    assert calls["out"] == str(tmp_path / "dest")


def test_fetch_skips_download_when_already_present(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "data.csv").write_text("a,b\n1,2\n")

    def fail(*args, **kwargs):
        raise AssertionError("should not download when cached")

    monkeypatch.setattr("fraud_benchmark.sources.kagglehub.dataset_download", fail)

    assert fetch(KaggleDataset("ealaxi/paysim1"), dest) == dest


def test_force_redownloads_even_when_present(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "data.csv").write_text("a,b\n1,2\n")
    called = []

    def fake_download(handle, *, force_download=False, output_dir=None):
        called.append(force_download)
        return str(dest)

    monkeypatch.setattr("fraud_benchmark.sources.kagglehub.dataset_download", fake_download)

    fetch(KaggleDataset("ealaxi/paysim1"), dest, force=True)
    assert called == [True]


def test_credential_error_produces_actionable_message(tmp_path, monkeypatch):
    def fake_download(handle, **kwargs):
        raise CredentialError("no creds")

    monkeypatch.setattr("fraud_benchmark.sources.kagglehub.dataset_download", fake_download)

    with pytest.raises(FetchError, match="docs/kaggle-setup.md"):
        fetch(KaggleDataset("ealaxi/paysim1"), tmp_path / "dest")


def test_competition_403_explains_rule_acceptance(tmp_path, monkeypatch):
    def fake_download(handle, **kwargs):
        response = requests.Response()
        response.status_code = 403
        raise KaggleApiHTTPError("forbidden", response=response)

    monkeypatch.setattr(
        "fraud_benchmark.sources.kagglehub.competition_download", fake_download
    )

    with pytest.raises(FetchError, match="accept.*rules"):
        fetch(KaggleCompetition("ieee-fraud-detection"), tmp_path / "dest")


def test_competition_403_message_includes_the_rules_url(tmp_path, monkeypatch):
    def fake_download(handle, **kwargs):
        response = requests.Response()
        response.status_code = 403
        raise KaggleApiHTTPError("forbidden", response=response)

    monkeypatch.setattr(
        "fraud_benchmark.sources.kagglehub.competition_download", fake_download
    )

    with pytest.raises(FetchError, match="ieee-fraud-detection/rules"):
        fetch(KaggleCompetition("ieee-fraud-detection"), tmp_path / "dest")


def test_empty_download_directory_is_an_error(tmp_path, monkeypatch):
    def fake_download(handle, **kwargs):
        (tmp_path / "dest").mkdir(parents=True, exist_ok=True)
        return str(tmp_path / "dest")

    monkeypatch.setattr("fraud_benchmark.sources.kagglehub.dataset_download", fake_download)

    with pytest.raises(FetchError, match="no files"):
        fetch(KaggleDataset("ealaxi/paysim1"), tmp_path / "dest")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sources.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'fraud_benchmark.sources'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/sources.py`:

```python
"""Fetching raw dataset files, with errors a human can act on."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import kagglehub
from kagglehub.exceptions import (
    CredentialError,
    KaggleApiHTTPError,
    NotFoundError,
    UnauthenticatedError,
)

SETUP_HINT = (
    "Kaggle credentials are missing or invalid. "
    "Follow docs/kaggle-setup.md to create ~/.kaggle/kaggle.json "
    "or set KAGGLE_USERNAME and KAGGLE_KEY."
)


class FetchError(RuntimeError):
    """Raised when raw data could not be fetched."""


@dataclass(frozen=True)
class KaggleDataset:
    """A regular Kaggle dataset, e.g. 'ealaxi/paysim1'."""

    handle: str

    @property
    def url(self) -> str:
        return f"https://www.kaggle.com/datasets/{self.handle}"


@dataclass(frozen=True)
class KaggleCompetition:
    """A Kaggle competition, e.g. 'ieee-fraud-detection'."""

    handle: str

    @property
    def url(self) -> str:
        return f"https://www.kaggle.com/c/{self.handle}"

    @property
    def rules_url(self) -> str:
        return f"https://www.kaggle.com/c/{self.handle}/rules"


Source = KaggleDataset | KaggleCompetition


def _has_files(directory: Path) -> bool:
    return directory.is_dir() and any(p.is_file() for p in directory.rglob("*"))


def fetch(source: Source, dest: Path, *, force: bool = False) -> Path:
    """Download `source` into `dest`, returning `dest`.

    Downloads are cached: if `dest` already contains files, nothing is fetched
    unless `force` is set.
    """
    if _has_files(dest) and not force:
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    try:
        if isinstance(source, KaggleDataset):
            kagglehub.dataset_download(
                source.handle, force_download=force, output_dir=str(dest)
            )
        elif isinstance(source, KaggleCompetition):
            kagglehub.competition_download(
                source.handle, force_download=force, output_dir=str(dest)
            )
        else:
            raise FetchError(f"unsupported source type: {type(source).__name__}")
    except (CredentialError, UnauthenticatedError) as exc:
        raise FetchError(f"{SETUP_HINT}\nOriginal error: {exc}") from exc
    except KaggleApiHTTPError as exc:
        raise FetchError(_http_error_message(source, exc)) from exc
    except NotFoundError as exc:
        raise FetchError(f"{source.url} was not found on Kaggle: {exc}") from exc

    if not _has_files(dest):
        raise FetchError(f"download of {source.url} produced no files in {dest}")

    return dest


def _http_error_message(source: Source, exc: KaggleApiHTTPError) -> str:
    status = getattr(exc.response, "status_code", None)

    if status == 403 and isinstance(source, KaggleCompetition):
        return (
            f"Kaggle returned 403 for competition {source.handle!r}. "
            "This usually means your credentials are fine but you have not yet "
            "accepted the competition rules. Open the page below in a browser while "
            "signed in and click 'I Understand and Accept', then retry:\n"
            f"  {source.rules_url}"
        )
    if status in (401, 403):
        return f"Kaggle returned {status} for {source.url}.\n{SETUP_HINT}"
    return f"Kaggle request for {source.url} failed with status {status}: {exc}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_sources.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/fraud_benchmark/sources.py tests/test_sources.py
git commit -m "feat: Kaggle dataset and competition fetching with actionable errors"
```

---

### Task 6: Adapter base class and registry

**Files:**
- Create: `src/fraud_benchmark/datasets/base.py`
- Modify: `src/fraud_benchmark/datasets/__init__.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_registry.py`:

```python
import pandas as pd
import pytest

from fraud_benchmark.datasets import base
from fraud_benchmark.datasets.base import (
    DatasetAdapter,
    UnknownDatasetError,
    get_adapter,
    list_datasets,
    register,
)
from fraud_benchmark.sources import KaggleDataset


def make_fake_class(name="fake_for_tests", handle="someone/fake"):
    class _FakeAdapter(DatasetAdapter):
        pass

    _FakeAdapter.name = name
    _FakeAdapter.source = KaggleDataset(handle)
    _FakeAdapter.caveats = ("this dataset is fake",)
    _FakeAdapter.to_canonical = lambda self, raw_dir, options: pd.DataFrame()
    _FakeAdapter.column_mapping = lambda self, options: {"event_time": "ts"}
    return _FakeAdapter


@pytest.fixture
def fake_adapter():
    """Register a throwaway adapter, then remove it.

    Registration must not leak: pytest imports every test module during collection,
    so a module-level @register would leave this fake in the global registry and
    break `prepare --all` in tests/test_cli.py.
    """
    cls = make_fake_class()
    register(cls)
    yield cls
    del base._REGISTRY[cls.name]


def test_get_adapter_returns_an_instance(fake_adapter):
    adapter = get_adapter("fake_for_tests")
    assert isinstance(adapter, fake_adapter)
    assert adapter.source.handle == "someone/fake"


def test_list_datasets_includes_registered_adapter(fake_adapter):
    assert "fake_for_tests" in list_datasets()


def test_registry_is_clean_without_the_fixture():
    assert "fake_for_tests" not in list_datasets()


def test_list_datasets_is_sorted():
    names = list_datasets()
    assert names == sorted(names)


def test_unknown_dataset_raises_with_available_names(fake_adapter):
    with pytest.raises(UnknownDatasetError, match="fake_for_tests"):
        get_adapter("no_such_dataset")


def test_registering_a_duplicate_name_is_an_error(fake_adapter):
    with pytest.raises(ValueError, match="already registered"):
        register(make_fake_class(handle="someone/other"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_registry.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'fraud_benchmark.datasets.base'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/datasets/base.py`:

```python
"""The dataset adapter interface and its registry.

An adapter's only job is turning one dataset's raw files into a canonical frame.
It knows nothing about splitting, label delay, or output formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.sources import Source


class UnknownDatasetError(KeyError):
    """Raised when a dataset name has no registered adapter."""


class DatasetAdapter(ABC):
    """Base class for per-dataset adapters."""

    #: Short identifier used on the CLI and as the output directory name.
    name: str
    #: Where the raw files come from.
    source: Source
    #: Human-readable warnings recorded in the dataset card.
    caveats: tuple[str, ...] = ()

    @abstractmethod
    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        """Read raw files from `raw_dir` and return a canonical frame.

        The result must contain event_time, entity_id, amount, and is_fraud with the
        dtypes in schema.REQUIRED_DTYPES, plus any source columns to pass through.
        """

    @abstractmethod
    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        """Map each canonical column to the source column(s) it came from."""


_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    """Class decorator adding an adapter to the registry."""
    if cls.name in _REGISTRY:
        raise ValueError(f"dataset {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str) -> DatasetAdapter:
    """Return an adapter instance for `name`."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise UnknownDatasetError(f"unknown dataset {name!r}; available: {available}")
    return _REGISTRY[name]()


def list_datasets() -> list[str]:
    """All registered dataset names, sorted."""
    return sorted(_REGISTRY)
```

- [ ] **Step 4: Register adapters on package import**

`src/fraud_benchmark/datasets/__init__.py`:

```python
"""Per-dataset adapters.

Importing this package registers every adapter. Plan 2 adds the remaining six
imports here.
"""

from fraud_benchmark.datasets import paysim  # noqa: F401
```

Note: this import fails until Task 7 creates `paysim.py`. That is expected and is
fixed in the next task. Leave `__init__.py` untouched until then — write it in Task 7,
Step 5.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_registry.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/fraud_benchmark/datasets/base.py tests/test_registry.py
git commit -m "feat: dataset adapter base class and registry"
```

---

### Task 7: PaySim adapter

**Files:**
- Create: `src/fraud_benchmark/datasets/paysim.py`
- Create: `tests/fixtures/paysim/sample.csv`
- Modify: `src/fraud_benchmark/datasets/__init__.py`
- Test: `tests/test_paysim.py`

Background: PaySim's real file is a single CSV whose columns are `step, type, amount,
nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest,
isFraud, isFlaggedFraud`. `step` is a **1-based hour offset**, not a date, so
`event_time = start_date + (step - 1) hours`. The adapter globs for the CSV rather than
hardcoding Kaggle's filename, which contains an unstable numeric suffix.

- [ ] **Step 1: Create the fixture**

`tests/fixtures/paysim/sample.csv`:

```csv
step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0,0,0
1,TRANSFER,181.0,C1305486145,181.0,0.0,C553264065,0.0,0.0,1,0
2,CASH_OUT,181.0,C840083671,181.0,0.0,C38997010,21182.0,0.0,1,0
3,PAYMENT,11668.14,C2048537720,41554.0,29885.86,M1230701703,0.0,0.0,0,0
25,CASH_IN,7817.71,C90045638,53860.0,61677.71,C972765878,0.0,0.0,0,0
```

- [ ] **Step 2: Write the failing tests**

`tests/test_paysim.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.datasets.base import get_adapter
from fraud_benchmark.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "paysim"
OPTIONS = {"start_date": "2023-01-01"}


@pytest.fixture
def frame():
    return get_adapter("paysim").to_canonical(FIXTURE, OPTIONS)


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_step_is_anchored_to_start_date(frame):
    # step 1 is the first hour, so it maps to start_date itself.
    assert frame["event_time"].iloc[0] == pd.Timestamp("2023-01-01 00:00:00")
    # step 25 is 24 hours later.
    assert frame["event_time"].iloc[4] == pd.Timestamp("2023-01-02 00:00:00")


def test_entity_id_is_the_originating_customer(frame):
    assert frame["entity_id"].iloc[0] == "C1231006815"


def test_is_fraud_is_boolean(frame):
    assert frame["is_fraud"].tolist() == [False, True, True, False, False]


def test_source_columns_pass_through(frame):
    assert frame["type"].iloc[0] == "PAYMENT"
    assert "isFlaggedFraud" in frame.columns
    assert "oldbalanceOrg" in frame.columns


def test_original_step_column_is_kept(frame):
    assert frame["step"].tolist() == [1, 1, 2, 3, 25]


def test_column_mapping_documents_provenance():
    mapping = get_adapter("paysim").column_mapping(OPTIONS)
    assert mapping["entity_id"] == "nameOrig"
    assert mapping["is_fraud"] == "isFraud"
    assert "step" in mapping["event_time"]


def test_missing_start_date_is_an_error():
    with pytest.raises(ValueError, match="start_date"):
        get_adapter("paysim").to_canonical(FIXTURE, {})


def test_missing_csv_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="CSV"):
        get_adapter("paysim").to_canonical(tmp_path, OPTIONS)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_paysim.py -v`
Expected: all fail with `UnknownDatasetError: unknown dataset 'paysim'`

- [ ] **Step 4: Write the implementation**

`src/fraud_benchmark/datasets/paysim.py`:

```python
"""PaySim: an agent-based mobile money transaction simulator.

https://www.kaggle.com/datasets/ealaxi/paysim1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.datasets.base import DatasetAdapter, register
from fraud_benchmark.sources import KaggleDataset


def find_single_csv(raw_dir: Path) -> Path:
    """Return the one CSV in `raw_dir`, searched recursively.

    Kaggle filenames carry unstable numeric suffixes, so adapters glob rather than
    hardcode them.
    """
    matches = sorted(raw_dir.rglob("*.csv"))
    if not matches:
        raise FileNotFoundError(f"no CSV file found under {raw_dir}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise FileNotFoundError(f"expected exactly one CSV under {raw_dir}, found: {names}")
    return matches[0]


@register
class PaySimAdapter(DatasetAdapter):
    name = "paysim"
    source = KaggleDataset("ealaxi/paysim1")
    caveats = (
        "PaySim is fully synthetic. Its 'step' column is a 1-based hour offset, not a "
        "real date, so event_time is anchored to a configured start_date and the "
        "absolute dates carry no meaning.",
        "Timestamps have hourly granularity, so many transactions share an event_time.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        start_date = options.get("start_date")
        if not start_date:
            raise ValueError(
                "paysim requires a 'start_date' option to anchor its relative 'step' "
                "column; set datasets.paysim.start_date in the config"
            )

        df = pd.read_csv(find_single_csv(raw_dir))

        # PaySim already has a column literally named `amount`, so cast it in place
        # rather than inserting a second one.
        anchor = pd.Timestamp(start_date)
        df["amount"] = df["amount"].astype("float64")
        df.insert(0, "event_time", anchor + pd.to_timedelta(df["step"] - 1, unit="h"))
        df.insert(1, "entity_id", df["nameOrig"].astype("string"))
        df.insert(2, "is_fraud", df["isFraud"].astype(bool))
        return df

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        return {
            "event_time": f"step (hours since {options.get('start_date')})",
            "entity_id": "nameOrig",
            "amount": "amount",
            "is_fraud": "isFraud",
        }
```

- [ ] **Step 5: Register the adapter**

`src/fraud_benchmark/datasets/__init__.py`:

```python
"""Per-dataset adapters.

Importing this package registers every adapter. Plan 2 adds the remaining six
imports here.
"""

from fraud_benchmark.datasets import paysim  # noqa: F401
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_paysim.py -v`
Expected: 9 passed

- [ ] **Step 7: Run the whole suite**

Run: `.venv/bin/pytest`
Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add src/fraud_benchmark/datasets/paysim.py src/fraud_benchmark/datasets/__init__.py tests/test_paysim.py tests/fixtures/paysim/sample.csv
git commit -m "feat: PaySim adapter"
```

---

### Task 8: Temporal splitting

**Files:**
- Create: `src/fraud_benchmark/splitting.py`
- Test: `tests/test_splitting.py`

Background: the split must cut on **timestamp values, not row positions**, so that rows
sharing an event_time never straddle a boundary. This matters enormously for PaySim and
BankSim, whose timestamps are hourly or daily and therefore heavily tied.

Algorithm: count rows per distinct timestamp, take the cumulative fraction, and find the
first timestamp at which the cumulative fraction reaches each target. That timestamp and
everything before it belong to the earlier split.

- [ ] **Step 1: Write the failing tests**

`tests/test_splitting.py`:

```python
import pandas as pd
import pytest

from fraud_benchmark.splitting import assign_splits, split_boundaries


def frame_with_times(times):
    return pd.DataFrame({"event_time": pd.to_datetime(times)})


def test_splits_ten_distinct_days_80_10_10():
    df = frame_with_times([f"2023-01-{d:02d}" for d in range(1, 11)])
    splits = assign_splits(df, (0.8, 0.1, 0.1))
    assert splits.tolist() == ["train"] * 8 + ["val"] + ["test"]


def test_result_is_categorical_with_all_three_levels():
    df = frame_with_times([f"2023-01-{d:02d}" for d in range(1, 11)])
    splits = assign_splits(df, (0.8, 0.1, 0.1))
    assert str(splits.dtype) == "category"
    assert set(splits.cat.categories) == {"train", "val", "test"}


def test_rows_sharing_a_timestamp_are_never_split():
    # Nine rows on day 1, one row each on days 2 and 3. A naive positional 80/10/10
    # cut would slice through the day-1 block.
    times = ["2023-01-01"] * 9 + ["2023-01-02", "2023-01-03"]
    df = frame_with_times(times)
    splits = assign_splits(df, (0.8, 0.1, 0.1))
    day_one = splits[:9]
    assert day_one.nunique() == 1


def test_split_is_monotonic_in_time():
    df = frame_with_times([f"2023-01-{d:02d}" for d in range(1, 21)])
    splits = assign_splits(df, (0.7, 0.2, 0.1))
    rank = {"train": 0, "val": 1, "test": 2}
    codes = [rank[s] for s in splits]
    assert codes == sorted(codes)


def test_unsorted_input_is_handled():
    df = frame_with_times(
        ["2023-01-05", "2023-01-01", "2023-01-03", "2023-01-02", "2023-01-04"]
    )
    splits = assign_splits(df, (0.6, 0.2, 0.2))
    # The earliest date must be train, the latest must be test.
    assert splits.iloc[1] == "train"
    assert splits.iloc[0] == "test"


def test_single_timestamp_puts_everything_in_train():
    df = frame_with_times(["2023-01-01"] * 5)
    splits = assign_splits(df, (0.8, 0.1, 0.1))
    assert set(splits) == {"train"}


def test_boundaries_are_reported():
    df = frame_with_times([f"2023-01-{d:02d}" for d in range(1, 11)])
    bounds = split_boundaries(df, (0.8, 0.1, 0.1))
    assert bounds["train_end"] == pd.Timestamp("2023-01-08")
    assert bounds["val_end"] == pd.Timestamp("2023-01-09")


def test_ratios_must_sum_to_one():
    df = frame_with_times(["2023-01-01", "2023-01-02"])
    with pytest.raises(ValueError, match="sum to 1"):
        assign_splits(df, (0.5, 0.2, 0.2))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_splitting.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'fraud_benchmark.splitting'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/splitting.py`:

```python
"""Temporal train/validation/test splitting.

Cuts land on timestamp values rather than row positions, so transactions sharing an
event_time always end up in the same split. Datasets with coarse time resolution
(PaySim's hours, BankSim's days) have heavy ties, and slicing through a tied block
would leak same-instant information across the boundary.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


def _cumulative_fraction(df: pd.DataFrame) -> pd.Series:
    """Fraction of rows at or before each distinct timestamp, indexed by timestamp."""
    counts = df["event_time"].value_counts().sort_index()
    return counts.cumsum() / len(df)


def _cut_at(cumulative: pd.Series, target: float) -> pd.Timestamp:
    """The last timestamp belonging to a split ending at `target`.

    Returns the first timestamp whose cumulative fraction reaches `target`, so a split
    is never empty even when a single tied block is larger than its target share.
    """
    index = int(np.searchsorted(cumulative.to_numpy(), target, side="left"))
    index = min(index, len(cumulative) - 1)
    return cumulative.index[index]


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != 3:
        raise ValueError(f"expected three ratios, got {len(ratios)}")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError(f"ratios must sum to 1, got {sum(ratios)}")


def split_boundaries(
    df: pd.DataFrame, ratios: tuple[float, float, float]
) -> dict[str, pd.Timestamp]:
    """The last timestamp in the train and val splits."""
    _validate_ratios(ratios)
    cumulative = _cumulative_fraction(df)
    return {
        "train_end": _cut_at(cumulative, ratios[0]),
        "val_end": _cut_at(cumulative, ratios[0] + ratios[1]),
    }


def assign_splits(
    df: pd.DataFrame, ratios: tuple[float, float, float]
) -> pd.Series:
    """Return a categorical split label per row, aligned to `df`'s index."""
    bounds = split_boundaries(df, ratios)
    times = df["event_time"]

    labels = np.where(
        times <= bounds["train_end"],
        "train",
        np.where(times <= bounds["val_end"], "val", "test"),
    )
    return pd.Series(
        pd.Categorical(labels, categories=SPLIT_NAMES), index=df.index, name="split"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_splitting.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/fraud_benchmark/splitting.py tests/test_splitting.py
git commit -m "feat: tie-safe temporal train/val/test splitting"
```

---

### Task 9: Pipeline orchestration

**Files:**
- Create: `src/fraud_benchmark/pipeline.py`
- Test: `tests/test_pipeline.py`

Background: the pipeline runs fetch → canonicalize → validate → split → order → write.
Output goes to a temp directory that is atomically renamed into place, so an interrupted
run never leaves a half-written dataset that looks valid.

`reported_at` is not produced here; Plan 3 adds it.

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:

```python
import json
from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.config import Config
from fraud_benchmark.pipeline import prepare

FIXTURE = Path(__file__).parent / "fixtures" / "paysim"


@pytest.fixture
def config(tmp_path):
    return Config(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        split_ratios=(0.6, 0.2, 0.2),
        datasets={"paysim": {"start_date": "2023-01-01"}},
    )


@pytest.fixture
def no_download(monkeypatch):
    """Pretend the raw files are already downloaded, using the test fixture."""

    def fake_fetch(source, dest, *, force=False):
        return FIXTURE

    monkeypatch.setattr("fraud_benchmark.pipeline.fetch", fake_fetch)


def test_prepare_writes_parquet(config, no_download):
    out = prepare("paysim", config)
    assert (out / "data.parquet").exists()
    assert (out / "dataset_card.json").exists()


def test_output_has_core_columns_first(config, no_download):
    out = prepare("paysim", config)
    df = pd.read_parquet(out / "data.parquet")
    assert list(df.columns)[:5] == [
        "event_time",
        "entity_id",
        "amount",
        "is_fraud",
        "split",
    ]


def test_output_row_count_matches_fixture(config, no_download):
    out = prepare("paysim", config)
    df = pd.read_parquet(out / "data.parquet")
    assert len(df) == 5


def test_every_row_has_a_split(config, no_download):
    out = prepare("paysim", config)
    df = pd.read_parquet(out / "data.parquet")
    assert df["split"].notna().all()
    assert set(df["split"]) <= {"train", "val", "test"}


def test_dataset_card_records_counts_and_provenance(config, no_download):
    out = prepare("paysim", config)
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["name"] == "paysim"
    assert card["n_rows"] == 5
    assert card["n_fraud"] == 2
    assert card["source"]["handle"] == "ealaxi/paysim1"
    assert card["column_mapping"]["entity_id"] == "nameOrig"
    assert card["caveats"]
    assert card["split"]["ratios"] == [0.6, 0.2, 0.2]
    assert "train" in card["split"]["counts"]


def test_rerun_replaces_previous_output(config, no_download):
    first = prepare("paysim", config)
    stale = first / "stale.txt"
    stale.write_text("left over from a previous run")
    prepare("paysim", config)
    assert not stale.exists()
    assert (first / "data.parquet").exists()


def test_no_temp_directory_is_left_behind(config, no_download):
    prepare("paysim", config)
    leftovers = list(config.processed_dir.glob("*.tmp*"))
    assert leftovers == []


def test_validation_failure_writes_nothing(config, monkeypatch):
    def fake_fetch(source, dest, *, force=False):
        return FIXTURE

    monkeypatch.setattr("fraud_benchmark.pipeline.fetch", fake_fetch)
    monkeypatch.setattr(
        "fraud_benchmark.pipeline.validate_canonical",
        lambda df: (_ for _ in ()).throw(ValueError("boom")),
    )
    with pytest.raises(ValueError, match="boom"):
        prepare("paysim", config)
    assert not (config.processed_dir / "paysim").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'fraud_benchmark.pipeline'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/pipeline.py`:

```python
"""Stage orchestration: fetch, canonicalize, validate, split, write."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import fraud_benchmark.datasets  # noqa: F401  (registers all adapters)
from fraud_benchmark.config import Config
from fraud_benchmark.datasets.base import get_adapter
from fraud_benchmark.schema import order_columns, validate_canonical
from fraud_benchmark.splitting import assign_splits, split_boundaries
from fraud_benchmark.sources import fetch


def prepare(name: str, config: Config, *, force: bool = False) -> Path:
    """Run the full pipeline for one dataset. Returns its output directory."""
    adapter = get_adapter(name)
    options = config.for_dataset(name)

    raw_dir = fetch(adapter.source, config.raw_dir / name, force=force)

    df = adapter.to_canonical(raw_dir, options)
    validate_canonical(df)

    df = df.sort_values("event_time", kind="stable").reset_index(drop=True)
    df["split"] = assign_splits(df, config.split_ratios)
    df = order_columns(df)

    card = _build_card(name, adapter, options, df, config)
    return _write_atomically(config.processed_dir / name, df, card)


def _build_card(name, adapter, options, df, config) -> dict:
    bounds = split_boundaries(df, config.split_ratios)
    return {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": _describe_source(adapter.source),
        "n_rows": int(len(df)),
        "n_fraud": int(df["is_fraud"].sum()),
        "fraud_rate": float(df["is_fraud"].mean()),
        "n_entities": int(df["entity_id"].nunique()),
        "time_range": {
            "start": df["event_time"].min().isoformat(),
            "end": df["event_time"].max().isoformat(),
        },
        "split": {
            "strategy": "temporal, cut on timestamp values",
            "ratios": list(config.split_ratios),
            "train_end": bounds["train_end"].isoformat(),
            "val_end": bounds["val_end"].isoformat(),
            "counts": {
                str(k): int(v) for k, v in df["split"].value_counts().items()
            },
        },
        "column_mapping": adapter.column_mapping(options),
        "options": options,
        "caveats": list(adapter.caveats),
    }


def _describe_source(source) -> dict:
    described = asdict(source) if is_dataclass(source) else {"repr": repr(source)}
    described["type"] = type(source).__name__
    described["url"] = getattr(source, "url", None)
    return described


def _write_atomically(dest: Path, df: pd.DataFrame, card: dict) -> Path:
    """Write into a temp directory, then swap it into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f"{dest.name}.tmp{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    try:
        df.to_parquet(staging / "data.parquet", index=False)
        (staging / "dataset_card.json").write_text(json.dumps(card, indent=2) + "\n")

        if dest.exists():
            shutil.rmtree(dest)
        os.replace(staging, dest)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return dest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/fraud_benchmark/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestration with dataset cards and atomic writes"
```

---

### Task 10: Command-line interface

**Files:**
- Create: `src/fraud_benchmark/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:

```python
import json
from pathlib import Path

import pytest

from fraud_benchmark.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "paysim"


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "test.yaml"
    path.write_text(
        f"paths:\n"
        f"  raw: {tmp_path / 'raw'}\n"
        f"  processed: {tmp_path / 'processed'}\n"
        f"split:\n"
        f"  ratios: [0.6, 0.2, 0.2]\n"
    )
    return path


@pytest.fixture
def no_download(monkeypatch):
    monkeypatch.setattr(
        "fraud_benchmark.pipeline.fetch", lambda source, dest, force=False: FIXTURE
    )


def test_list_prints_registered_datasets(capsys):
    assert main(["list"]) == 0
    assert "paysim" in capsys.readouterr().out


def test_prepare_creates_output(tmp_path, config_file, no_download):
    assert main(["prepare", "paysim", "--config", str(config_file)]) == 0
    assert (tmp_path / "processed" / "paysim" / "data.parquet").exists()


def test_prepare_all_creates_output(tmp_path, config_file, no_download):
    assert main(["prepare", "--all", "--config", str(config_file)]) == 0
    assert (tmp_path / "processed" / "paysim" / "data.parquet").exists()


def test_prepare_requires_a_target(config_file):
    with pytest.raises(SystemExit):
        main(["prepare", "--config", str(config_file)])


def test_info_prints_card(tmp_path, config_file, no_download, capsys):
    main(["prepare", "paysim", "--config", str(config_file)])
    capsys.readouterr()
    assert main(["info", "paysim", "--config", str(config_file)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["name"] == "paysim"


def test_info_on_unprepared_dataset_returns_error(config_file, capsys):
    assert main(["info", "paysim", "--config", str(config_file)]) == 1
    assert "not been prepared" in capsys.readouterr().err


def test_unknown_dataset_returns_error(config_file, capsys):
    assert main(["prepare", "nope", "--config", str(config_file)]) == 1
    assert "unknown dataset" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'fraud_benchmark.cli'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/cli.py`:

```python
"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

import fraud_benchmark.datasets  # noqa: F401  (registers all adapters)
from fraud_benchmark.config import load_config
from fraud_benchmark.datasets.base import UnknownDatasetError, get_adapter, list_datasets
from fraud_benchmark.pipeline import prepare
from fraud_benchmark.sources import FetchError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-benchmark",
        description="Prepare fraud and AML benchmark datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list available datasets")

    prepare_parser = subparsers.add_parser("prepare", help="download and process a dataset")
    prepare_parser.add_argument("dataset", nargs="?", help="dataset name")
    prepare_parser.add_argument("--all", action="store_true", help="prepare every dataset")
    prepare_parser.add_argument("--config", help="path to a config file")
    prepare_parser.add_argument(
        "--force", action="store_true", help="re-download even if raw files exist"
    )

    info_parser = subparsers.add_parser("info", help="print a prepared dataset's card")
    info_parser.add_argument("dataset")
    info_parser.add_argument("--config", help="path to a config file")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return _cmd_list()
    if args.command == "prepare":
        # Validated here rather than with a mutually exclusive group: argparse
        # handles an optional positional inside such a group unreliably.
        if bool(args.dataset) == bool(args.all):
            parser.error("give exactly one of: a dataset name, or --all")
        return _cmd_prepare(args)
    return _cmd_info(args)


def _cmd_list() -> int:
    for name in list_datasets():
        adapter = get_adapter(name)
        print(f"{name:12s} {getattr(adapter.source, 'url', '')}")
    return 0


def _cmd_prepare(args) -> int:
    config = load_config(args.config)
    names = list_datasets() if args.all else [args.dataset]

    for name in names:
        try:
            out = prepare(name, config, force=args.force)
        except UnknownDatasetError as exc:
            print(str(exc).strip("\"'"), file=sys.stderr)
            return 1
        except FetchError as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            return 1
        print(f"{name}: wrote {out}")
    return 0


def _cmd_info(args) -> int:
    config = load_config(args.config)
    card = config.processed_dir / args.dataset / "dataset_card.json"
    if not card.exists():
        print(
            f"{args.dataset} has not been prepared; run: "
            f"fraud-benchmark prepare {args.dataset}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(json.loads(card.read_text()), indent=2))
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: all tests pass, none failed

- [ ] **Step 6: Commit**

```bash
git add src/fraud_benchmark/cli.py tests/test_cli.py
git commit -m "feat: list/prepare/info command-line interface"
```

---

### Task 11: End-to-end verification against real PaySim

**Files:**
- Create: `tests/test_integration.py`

This is the task that proves the spine actually works. It downloads roughly 180 MB.

- [ ] **Step 1: Write the network-marked integration test**

`tests/test_integration.py`:

```python
"""Live download test. Deselected by default; run with: pytest -m network"""

import pandas as pd
import pytest

from fraud_benchmark.config import Config
from fraud_benchmark.pipeline import prepare


@pytest.mark.network
def test_paysim_end_to_end(tmp_path):
    config = Config(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        split_ratios=(0.8, 0.1, 0.1),
        datasets={"paysim": {"start_date": "2023-01-01"}},
    )
    out = prepare("paysim", config)
    df = pd.read_parquet(out / "data.parquet")

    assert len(df) > 6_000_000
    assert df["is_fraud"].sum() > 0
    assert set(df["split"]) == {"train", "val", "test"}
    # Splits must be contiguous in time.
    assert df[df["split"] == "train"]["event_time"].max() <= (
        df[df["split"] == "val"]["event_time"].min()
    )
```

- [ ] **Step 2: Run the real thing via the CLI**

Run:

```bash
.venv/bin/fraud-benchmark prepare paysim
```

Expected: download progress, then `paysim: wrote data/processed/paysim`.

If it fails with a credentials message, return to Task 2.

- [ ] **Step 3: Inspect the result**

Run:

```bash
.venv/bin/fraud-benchmark info paysim
```

Expected: JSON showing roughly 6.36M rows, ~8 213 frauds (a fraud rate near 0.13%),
a time range spanning about 30 days from 2023-01-01, and non-zero train/val/test counts.

Report these actual numbers to the user rather than assuming they match.

- [ ] **Step 4: Run the integration test**

Run: `.venv/bin/pytest -m network -v`
Expected: 1 passed

- [ ] **Step 5: Confirm data/ is not tracked by git**

Run: `git status --short`
Expected: no `data/` entries appear.

- [ ] **Step 6: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: live end-to-end PaySim integration test"
```

---

## Done criteria

- [ ] `.venv/bin/pytest -q` passes with no failures
- [ ] `fraud-benchmark list` shows `paysim`
- [ ] `fraud-benchmark prepare paysim` produces `data/processed/paysim/data.parquet`
      and `dataset_card.json`
- [ ] The parquet's first five columns are `event_time`, `entity_id`, `amount`,
      `is_fraud`, `split`, and every row has a split
- [ ] `data/` is untracked

## Not in this plan

- The six remaining adapters and the git source type (Plan 2)
- The `reported_at` label-delay stage and campaign grouping (Plan 3)
