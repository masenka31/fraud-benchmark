# Adapters Part 1 Implementation Plan (Plan 2 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add shared file-selection helpers, licence-aware batch preparation, and the three straightforward adapters — BankSim, Sparkov, SAML-D — verified against real downloads.

**Architecture:** Unchanged from Plan 1. Each dataset is one adapter class implementing `to_canonical(raw_dir, options)`; all downstream stages are shared. This plan adds no new pipeline stages.

**Tech Stack:** Python 3.13, pandas 3.0.5, pyarrow, kagglehub 1.0.2, pytest.

---

## Plan renumbering

The original spec described three plans. Writing six adapters in one plan would be
unreviewable, so the remaining work is split into four:

| Plan | Contents | Status |
|---|---|---|
| 1 | Pipeline spine + PaySim | done, merged |
| **2 (this)** | Shared helpers, licence gating, BankSim + Sparkov + SAML-D | |
| 3 | IBM CCF, IEEE-CIS, Amaretto — the three with joins, heuristics, or split archives | |
| 4 | Label delay (`reported_at`) and campaign grouping | |

---

## Verified facts

Every schema below was read from the actual downloaded files on 2026-08-01, not from
memory. The raw data is already cached under `data/raw/<name>/`.

### BankSim — `data/raw/banksim/`

- **Two CSVs.** Use `bs140513_032310.csv` (594,643 rows x 10 cols). Ignore
  `bsNET140513_032310.csv`, which is a graph edge list (Source/Target/Weight), not
  transactions.
- **Every string column is wrapped in literal single quotes, in 100% of rows.** Raw values
  are `'C1093826151'`, `'es_transportation'`, `'M'`. All seven string columns are affected:
  `customer, age, gender, zipcodeOri, merchant, zipMerchant, category`.
- `step` is **0-based, in days**, range 0–179 (180 distinct values). Needs a `start_date`.
- Entity: `customer`, 4,098 distinct. Fraud rate 1.211%.
- `zipcodeOri` and `zipMerchant` are constant (`'28007'`) — useless but passed through.
- Licence: **CC BY-NC-SA 4.0 — NonCommercial.**

### Sparkov — `data/raw/sparkov/`

- **Two CSVs, both labelled, temporally consecutive and non-overlapping** (verified):
  - `fraudTrain.csv` 1,296,675 rows, 2019-01-01 00:00:18 → 2020-06-21 12:13:37, fraud 0.579%
  - `fraudTest.csv` 555,719 rows, 2020-06-21 12:14:25 → 2020-12-31 23:59:34, fraud 0.386%
  - Concatenated: 1,852,394 rows. Their split is discarded; ours is applied instead.
- `trans_date_trans_time` is a **real datetime string** — no anchor needed.
- Entity: `cc_num` (int64, 16 digits, fits int64) → cast to string.
- Amount column is `amt`. Label is `is_fraud`.
- `Unnamed: 0` is a per-file row index, meaningless after concatenation — **drop it**.
- Contains `first`, `last`, `street`, `dob`, `job`, `city` — synthetic (Faker-generated) but
  person-shaped; worth a caveat.
- Licence: CC0 1.0.

### SAML-D — `data/raw/saml_d/`

- One CSV, `SAML-D.csv`, 9,504,852 rows x 12 cols.
- `Date` (`2022-10-07`) and `Time` (`10:35:19`) are separate strings. Concatenating with a
  space and calling `pd.to_datetime` yields `datetime64[us]` — verified.
- Entity: `Sender_account` (int64) → cast to string. 30,882 distinct in sample.
- Label `Is_laundering` 0/1, rate 0.118% in sample.
- `Laundering_type` has 27 values and encodes the typology, including `Normal_Fan_Out`,
  `Normal_Cash_Deposits` for non-laundering rows. **Keep it** — Plan 4 will likely use it
  for campaign grouping.
- `Payment_currency` / `Received_currency`: 13 currencies. No FX conversion, per spec.
- Licence: **CC BY-NC-SA 4.0 — NonCommercial.**

---

## File Structure

| File | Responsibility |
|---|---|
| `src/fraud_benchmark/datasets/files.py` | Shared raw-file selection helpers |
| `src/fraud_benchmark/datasets/banksim.py` | BankSim adapter |
| `src/fraud_benchmark/datasets/sparkov.py` | Sparkov adapter |
| `src/fraud_benchmark/datasets/saml_d.py` | SAML-D adapter |
| `src/fraud_benchmark/datasets/base.py` | + `commercial_use` attribute |
| `src/fraud_benchmark/cli.py` | + `--exclude-noncommercial`; `--all` no longer aborts |
| `src/fraud_benchmark/pipeline.py` | + `commercial_use` in the card |
| `configs/default.yaml` | + `start_date` for banksim |

---

### Task 1: Shared file-selection helpers

**Why:** `find_single_csv` lives in `paysim.py` and errors when a directory holds more than
one CSV. Every dataset in this plan has multiple files, so it cannot be reused as-is. Move it
to a shared module and add an explicit by-name lookup.

**Files:**
- Create: `src/fraud_benchmark/datasets/files.py`
- Modify: `src/fraud_benchmark/datasets/paysim.py` (import from the new module)
- Test: `tests/test_files.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_files.py`

```python
import pytest

from fraud_benchmark.datasets.files import find_single_csv, require_file


def test_require_file_finds_a_named_file(tmp_path):
    (tmp_path / "wanted.csv").write_text("a\n1\n")
    (tmp_path / "other.csv").write_text("a\n1\n")
    assert require_file(tmp_path, "wanted.csv").name == "wanted.csv"


def test_require_file_searches_subdirectories(tmp_path):
    nested = tmp_path / "inner"
    nested.mkdir()
    (nested / "wanted.csv").write_text("a\n1\n")
    assert require_file(tmp_path, "wanted.csv") == nested / "wanted.csv"


def test_require_file_error_lists_what_is_present(tmp_path):
    (tmp_path / "actual.csv").write_text("a\n1\n")
    with pytest.raises(FileNotFoundError, match="actual.csv"):
        require_file(tmp_path, "missing.csv")


def test_require_file_error_names_the_file_it_wanted(tmp_path):
    (tmp_path / "actual.csv").write_text("a\n1\n")
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        require_file(tmp_path, "missing.csv")


def test_find_single_csv_still_works(tmp_path):
    (tmp_path / "only.csv").write_text("a\n1\n")
    assert find_single_csv(tmp_path).name == "only.csv"


def test_find_single_csv_rejects_two(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    (tmp_path / "b.csv").write_text("x\n1\n")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        find_single_csv(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_files.py -v`
Expected: `ModuleNotFoundError: No module named 'fraud_benchmark.datasets.files'`

- [ ] **Step 3: Write the implementation** — `src/fraud_benchmark/datasets/files.py`

```python
"""Locating raw files inside a downloaded dataset directory.

Kaggle bundles vary: some ship one CSV, others ship several plus auxiliary tables.
Adapters that need a specific file name it explicitly rather than globbing.
"""

from __future__ import annotations

from pathlib import Path


def _all_files(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.rglob("*") if p.is_file())


def require_file(raw_dir: Path, name: str) -> Path:
    """Return the file called `name` under `raw_dir`, searched recursively."""
    for path in _all_files(raw_dir):
        if path.name == name:
            return path
    present = ", ".join(p.name for p in _all_files(raw_dir)) or "(directory is empty)"
    raise FileNotFoundError(
        f"expected a file named {name!r} under {raw_dir}; found: {present}"
    )


def find_single_csv(raw_dir: Path) -> Path:
    """Return the one CSV under `raw_dir`, erroring if there is not exactly one.

    Only for datasets that genuinely ship a single CSV whose name is unstable.
    Prefer `require_file` when the name is known.
    """
    matches = [p for p in _all_files(raw_dir) if p.suffix.lower() == ".csv"]
    if not matches:
        raise FileNotFoundError(f"no CSV file found under {raw_dir}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise FileNotFoundError(
            f"expected exactly one CSV under {raw_dir}, found: {names}"
        )
    return matches[0]
```

- [ ] **Step 4: Point paysim at the shared module**

In `src/fraud_benchmark/datasets/paysim.py`, delete the local `find_single_csv` definition
and replace it with an import:

```python
from fraud_benchmark.datasets.files import find_single_csv
```

Leave everything else in that file unchanged.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_files.py tests/test_paysim.py -v`
Expected: 6 files tests + 12 paysim tests pass. The paysim tests must still pass unchanged —
if `test_missing_csv_is_an_error` breaks, the move altered behaviour and you should STOP.

- [ ] **Step 6: Full suite and commit**

Run: `.venv/bin/pytest` — expect 92 passed, 1 deselected.

```bash
git add src/fraud_benchmark/datasets/files.py src/fraud_benchmark/datasets/paysim.py tests/test_files.py
git commit -m "refactor: shared raw-file selection helpers"
```

---

### Task 2: Licence-aware batch preparation

**Why:** two of the seven datasets are NonCommercial, and `--all` currently aborts on the
first failure without saying which datasets succeeded — harmless with one adapter, wrong
with seven.

**Files:**
- Modify: `src/fraud_benchmark/datasets/base.py`
- Modify: `src/fraud_benchmark/datasets/paysim.py`
- Modify: `src/fraud_benchmark/pipeline.py`
- Modify: `src/fraud_benchmark/cli.py`
- Test: `tests/test_cli.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline.py`:

```python
def test_dataset_card_records_commercial_use(config, no_download):
    out = prepare("paysim", config)
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["commercial_use"] is True
```

Add to `tests/test_cli.py`:

```python
def test_list_marks_noncommercial_datasets(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    # paysim is commercially usable, so it must not be flagged.
    paysim_line = [ln for ln in out.splitlines() if ln.startswith("paysim")][0]
    assert "noncommercial" not in paysim_line.lower()


def test_prepare_all_continues_past_a_failure(tmp_path, config_file, monkeypatch, capsys):
    """A failing dataset must not prevent the others from being prepared."""
    from fraud_benchmark.sources import FetchError

    calls = []

    def flaky(name, config, *, force=False):
        calls.append(name)
        if name == "always_fails":
            raise FetchError("simulated network failure")
        return config.processed_dir / name

    class _Fake:
        def __init__(self, name):
            self.name = name
            self.commercial_use = True
            self.data_license = "CC0 1.0"
            self.source = None

    monkeypatch.setattr("fraud_benchmark.cli.prepare", flaky)
    monkeypatch.setattr(
        "fraud_benchmark.cli.list_datasets", lambda: ["always_fails", "paysim"]
    )
    # _cmd_prepare resolves the adapter before preparing, so this must be stubbed
    # too — otherwise "always_fails" raises UnknownDatasetError and the loop exits
    # before either dataset is attempted.
    monkeypatch.setattr("fraud_benchmark.cli.get_adapter", _Fake)

    assert main(["prepare", "--all", "--config", str(config_file)]) == 1
    # Both were attempted, not just the first.
    assert calls == ["always_fails", "paysim"]
    captured = capsys.readouterr()
    assert "always_fails" in captured.err
    assert "paysim" in captured.out


def test_exclude_noncommercial_skips_those_datasets(tmp_path, config_file, monkeypatch, capsys):
    prepared = []

    def record(name, config, *, force=False):
        prepared.append(name)
        return config.processed_dir / name

    class _Fake:
        def __init__(self, name, commercial):
            self.name = name
            self.commercial_use = commercial
            self.source = None

    monkeypatch.setattr("fraud_benchmark.cli.prepare", record)
    monkeypatch.setattr("fraud_benchmark.cli.list_datasets", lambda: ["open_one", "nc_one"])
    monkeypatch.setattr(
        "fraud_benchmark.cli.get_adapter",
        lambda n: _Fake(n, commercial=(n == "open_one")),
    )

    assert main(["prepare", "--all", "--exclude-noncommercial",
                 "--config", str(config_file)]) == 0
    assert prepared == ["open_one"]
    assert "nc_one" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py tests/test_pipeline.py -v`
Expected: the four new tests fail; existing tests still pass.

- [ ] **Step 3: Add `commercial_use` to the adapter base**

In `src/fraud_benchmark/datasets/base.py`, inside `DatasetAdapter`, directly after the
`data_license` attribute:

```python
    #: False when the upstream licence forbids commercial use (e.g. CC BY-NC-SA).
    #: Drives `prepare --all --exclude-noncommercial`.
    commercial_use: bool = True
```

In `src/fraud_benchmark/datasets/paysim.py`, after `data_license`, add:

```python
    commercial_use = True
```

- [ ] **Step 4: Record it in the dataset card**

In `src/fraud_benchmark/pipeline.py`, in `_build_card`, immediately after the
`"data_license"` entry:

```python
        "commercial_use": adapter.commercial_use,
```

- [ ] **Step 5: Rework the CLI**

Replace `_cmd_list` and `_cmd_prepare` in `src/fraud_benchmark/cli.py` with:

```python
def _cmd_list() -> int:
    for name in list_datasets():
        adapter = get_adapter(name)
        flag = "" if adapter.commercial_use else "  [noncommercial]"
        url = getattr(adapter.source, "url", "")
        print(f"{name:12s} {adapter.data_license:18s} {url}{flag}")
    return 0


def _cmd_prepare(args) -> int:
    config = load_config(args.config)
    names = list_datasets() if args.all else [args.dataset]

    failures: list[str] = []
    for name in names:
        try:
            adapter = get_adapter(name)
        except UnknownDatasetError as exc:
            # UnknownDatasetError subclasses KeyError, whose __str__ adds repr
            # quotes. Print the raw message instead.
            print(exc.args[0], file=sys.stderr)
            return 1

        if args.exclude_noncommercial and not adapter.commercial_use:
            print(f"{name}: skipped, licence forbids commercial use "
                  f"({adapter.data_license})")
            continue

        try:
            out = prepare(name, config, force=args.force)
        except (FetchError, ValueError, FileNotFoundError) as exc:
            # Keep going: one unavailable dataset must not block the rest.
            print(f"{name}: FAILED {exc}", file=sys.stderr)
            failures.append(name)
            continue
        print(f"{name}: wrote {out}")

    if failures:
        print(f"\n{len(failures)} of {len(names)} failed: {', '.join(failures)}",
              file=sys.stderr)
        return 1
    return 0
```

Add the flag in `build_parser`, alongside `--force`:

```python
    prepare_parser.add_argument(
        "--exclude-noncommercial", action="store_true",
        help="skip datasets whose licence forbids commercial use",
    )
```

- [ ] **Step 6: Run tests and commit**

Run: `.venv/bin/pytest` — expect 96 passed, 1 deselected.

```bash
git add src/fraud_benchmark/datasets/base.py src/fraud_benchmark/datasets/paysim.py src/fraud_benchmark/pipeline.py src/fraud_benchmark/cli.py tests/test_cli.py tests/test_pipeline.py
git commit -m "feat: licence-aware listing and batch preparation"
```

---

### Task 3: BankSim adapter

**Files:**
- Create: `src/fraud_benchmark/datasets/banksim.py`
- Create: `tests/fixtures/banksim/bs140513_032310.csv`
- Create: `tests/fixtures/banksim/bsNET140513_032310.csv`
- Modify: `src/fraud_benchmark/datasets/__init__.py`
- Modify: `configs/default.yaml`
- Test: `tests/test_banksim.py`

- [ ] **Step 1: Create the fixtures**

`tests/fixtures/banksim/bs140513_032310.csv` — note the literal single quotes, which are
present in the real file:

```csv
step,customer,age,gender,zipcodeOri,merchant,zipMerchant,category,amount,fraud
0,'C1093826151','4','M','28007','M348934600','28007','es_transportation',4.55,0
0,'C352968107','2','M','28007','M348934600','28007','es_transportation',39.68,0
1,'C2054744914','4','F','28007','M1823072687','28007','es_health',26.89,1
2,'C1760612790','3','M','28007','M348934600','28007','es_transportation',17.25,0
179,'C1093826151','4','M','28007','M50039827','28007','es_otherservices',112.0,1
```

`tests/fixtures/banksim/bsNET140513_032310.csv` — the decoy graph file, present so the
adapter is proven to pick the right one:

```csv
Source,Target,Weight,typeTrans,fraud
'C1093826151','M348934600',4.55,'es_transportation',0
'C352968107','M348934600',39.68,'es_transportation',0
```

- [ ] **Step 2: Write the failing tests** — `tests/test_banksim.py`

```python
from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.datasets.base import get_adapter
from fraud_benchmark.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "banksim"
OPTIONS = {"start_date": "2023-01-01"}


@pytest.fixture
def frame():
    return get_adapter("banksim").to_canonical(FIXTURE, OPTIONS)


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_picks_the_transaction_file_not_the_graph_file(frame):
    # The graph file has a Source column; the transaction file has customer.
    assert "customer" in frame.columns
    assert "Source" not in frame.columns
    assert len(frame) == 5


def test_single_quotes_are_stripped_from_entity_id(frame):
    assert frame["entity_id"].iloc[0] == "C1093826151"
    assert not frame["entity_id"].str.startswith("'").any()


def test_single_quotes_are_stripped_from_every_string_column(frame):
    for column in ("customer", "age", "gender", "merchant", "category",
                   "zipcodeOri", "zipMerchant"):
        values = frame[column].astype(str)
        assert not values.str.startswith("'").any(), f"{column} still quoted"
        assert not values.str.endswith("'").any(), f"{column} still quoted"


def test_step_is_zero_based_days(frame):
    # step 0 is the first day, so it maps to start_date itself.
    assert frame["event_time"].iloc[0] == pd.Timestamp("2023-01-01")
    # step 179 is 179 days later.
    assert frame["event_time"].iloc[4] == pd.Timestamp("2023-01-01") + pd.Timedelta(days=179)


def test_is_fraud_is_boolean(frame):
    assert frame["is_fraud"].tolist() == [False, False, True, False, True]


def test_amount_is_float(frame):
    assert frame["amount"].iloc[0] == 4.55


def test_banksim_is_noncommercial():
    adapter = get_adapter("banksim")
    assert adapter.commercial_use is False
    assert adapter.data_license == "CC BY-NC-SA 4.0"


def test_column_mapping_documents_provenance():
    mapping = get_adapter("banksim").column_mapping(OPTIONS)
    assert mapping["entity_id"] == "customer"
    assert mapping["is_fraud"] == "fraud"
    assert "step" in mapping["event_time"]


def test_missing_start_date_is_an_error():
    with pytest.raises(ValueError, match="start_date"):
        get_adapter("banksim").to_canonical(FIXTURE, {})


def test_banksim_is_registered():
    from fraud_benchmark.datasets.base import list_datasets
    assert "banksim" in list_datasets()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_banksim.py -v`
Expected: `UnknownDatasetError: unknown dataset 'banksim'`

- [ ] **Step 4: Write the implementation** — `src/fraud_benchmark/datasets/banksim.py`

```python
"""BankSim: an agent-based retail-payment simulator.

https://www.kaggle.com/datasets/ealaxi/banksim1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.datasets.base import DatasetAdapter, register
from fraud_benchmark.datasets.files import require_file
from fraud_benchmark.sources import KaggleDataset

#: The transaction table. The bundle also ships bsNET140513_032310.csv, which is a
#: graph edge list (Source/Target/Weight) rather than transactions.
TRANSACTIONS_FILE = "bs140513_032310.csv"


@register
class BankSimAdapter(DatasetAdapter):
    name = "banksim"
    source = KaggleDataset("ealaxi/banksim1")
    data_license = "CC BY-NC-SA 4.0"
    commercial_use = False
    caveats = (
        "BankSim is fully synthetic. Its 'step' column is a 0-based day offset, not a "
        "real date, so event_time is anchored to a configured start_date and the "
        "absolute dates carry no meaning.",
        "Timestamps have daily granularity, so many transactions share an event_time.",
        "Every string column in the source file is wrapped in literal single quotes; "
        "the adapter strips them.",
        "zipcodeOri and zipMerchant are constant ('28007') in the source data.",
        "Licence is CC BY-NC-SA 4.0: NonCommercial and ShareAlike.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        start_date = options.get("start_date")
        if not start_date:
            raise ValueError(
                "banksim requires a 'start_date' option to anchor its relative 'step' "
                "column; set datasets.banksim.start_date in the config"
            )

        df = pd.read_csv(require_file(raw_dir, TRANSACTIONS_FILE))

        # Every string column arrives quoted, e.g. "'C1093826151'".
        for column in df.columns:
            if pd.api.types.is_string_dtype(df[column]):
                df[column] = df[column].str.strip("'")

        anchor = pd.Timestamp(start_date)
        df["amount"] = df["amount"].astype("float64")
        df.insert(0, "event_time", anchor + pd.to_timedelta(df["step"], unit="D"))
        df.insert(1, "entity_id", df["customer"].astype("string"))
        df.insert(2, "is_fraud", df["fraud"].astype(bool))
        return df

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        return {
            "event_time": f"step (days since {options.get('start_date')})",
            "entity_id": "customer",
            "amount": "amount",
            "is_fraud": "fraud",
        }
```

- [ ] **Step 5: Register it and add config**

`src/fraud_benchmark/datasets/__init__.py`:

```python
"""Per-dataset adapters.

Importing this package registers every adapter.
"""

from fraud_benchmark.datasets import banksim, paysim  # noqa: F401
```

In `configs/default.yaml`, under `datasets:`:

```yaml
  # BankSim's `step` column is a 0-based day offset (0-179), not a real date.
  banksim:
    start_date: "2023-01-01"
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/test_banksim.py -v` — expect 11 passed.
Run: `.venv/bin/pytest` — expect 107 passed, 1 deselected.

- [ ] **Step 7: Commit**

```bash
git add src/fraud_benchmark/datasets/banksim.py src/fraud_benchmark/datasets/__init__.py configs/default.yaml tests/test_banksim.py tests/fixtures/banksim/
git commit -m "feat: BankSim adapter"
```

---

### Task 4: Sparkov adapter

**Files:**
- Create: `src/fraud_benchmark/datasets/sparkov.py`
- Create: `tests/fixtures/sparkov/fraudTrain.csv`, `tests/fixtures/sparkov/fraudTest.csv`
- Modify: `src/fraud_benchmark/datasets/__init__.py`
- Test: `tests/test_sparkov.py`

**Why both files are concatenated:** they are two halves of one timeline, verified
non-overlapping (train ends 2020-06-21 12:13:37, test begins 2020-06-21 12:14:25). Their
split is discarded so that every dataset in the benchmark is split by the same rule.

- [ ] **Step 1: Create the fixtures**

`tests/fixtures/sparkov/fraudTrain.csv` (columns abridged to the ones the adapter touches
plus a few passthroughs — the real file has 23):

```csv
Unnamed: 0,trans_date_trans_time,cc_num,merchant,category,amt,first,last,gender,city,state,zip,lat,long,city_pop,job,dob,trans_num,unix_time,merch_lat,merch_long,is_fraud
0,2019-01-01 00:00:18,2703186189652095,fraud_Rippin and Kub,misc_net,4.97,Jennifer,Banks,F,Moravian Falls,NC,28654,36.0788,-81.1781,3495,Psychologist,1988-03-09,0b242abb623afc578575680df306,1325376018,36.011293,-82.048315,0
1,2019-06-02 12:00:00,630423337322,fraud_Heller Ltd,grocery_pos,107.23,Stephanie,Gill,F,Orient,WA,99160,48.8878,-118.2105,149,Educator,1978-06-21,1f76529f8574734946361c461b02,1325376044,49.159047,-118.186462,1
2,2020-06-21 12:13:37,2703186189652095,fraud_Lind-Buckridge,entertainment,220.11,Edward,Sanchez,M,Malad City,ID,83252,42.1808,-112.262,4154,Officer,1962-01-19,a1a22d70485983eac12b5b88dad1,1325376051,43.150704,-112.154481,0
```

`tests/fixtures/sparkov/fraudTest.csv`:

```csv
Unnamed: 0,trans_date_trans_time,cc_num,merchant,category,amt,first,last,gender,city,state,zip,lat,long,city_pop,job,dob,trans_num,unix_time,merch_lat,merch_long,is_fraud
0,2020-06-21 12:14:25,2291163933867244,fraud_Kirlin and Sons,personal_care,2.86,Jeff,Elliott,M,Columbia,SC,29209,33.9659,-80.9355,333497,Engineer,1968-03-19,2da90c7d74bd46a0caf3777415b3,1371816865,33.986391,-81.200714,0
1,2020-12-31 23:59:34,2291163933867244,fraud_Sporer-Keebler,health_fitness,29.84,Joanne,Williams,F,Altonah,UT,84002,40.3207,-110.436,302,Librarian,1990-01-17,324cc204407e99f51b0d6ca00550,1371816873,39.450498,-109.960431,1
```

- [ ] **Step 2: Write the failing tests** — `tests/test_sparkov.py`

```python
from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.datasets.base import get_adapter
from fraud_benchmark.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "sparkov"


@pytest.fixture
def frame():
    return get_adapter("sparkov").to_canonical(FIXTURE, {})


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_both_source_files_are_concatenated(frame):
    # 3 rows from fraudTrain plus 2 from fraudTest.
    assert len(frame) == 5


def test_rows_are_ordered_in_time(frame):
    assert frame["event_time"].is_monotonic_increasing


def test_event_time_comes_from_the_real_timestamp(frame):
    assert frame["event_time"].iloc[0] == pd.Timestamp("2019-01-01 00:00:18")
    assert frame["event_time"].iloc[-1] == pd.Timestamp("2020-12-31 23:59:34")


def test_entity_id_is_the_card_number_as_string(frame):
    assert frame["entity_id"].iloc[0] == "2703186189652095"
    assert str(frame["entity_id"].dtype) == "string"


def test_amount_comes_from_amt(frame):
    assert frame["amount"].iloc[0] == 4.97


def test_is_fraud_is_boolean(frame):
    assert frame["is_fraud"].sum() == 2


def test_row_index_column_is_dropped(frame):
    # 'Unnamed: 0' is a per-file index, meaningless once the files are concatenated.
    assert "Unnamed: 0" not in frame.columns


def test_source_column_is_recorded(frame):
    # Which file each row came from is worth keeping.
    assert set(frame["source_file"]) == {"fraudTrain.csv", "fraudTest.csv"}


def test_passthrough_columns_survive(frame):
    for column in ("merchant", "category", "city_pop", "merch_lat"):
        assert column in frame.columns


def test_sparkov_is_commercially_usable():
    adapter = get_adapter("sparkov")
    assert adapter.commercial_use is True
    assert adapter.data_license == "CC0 1.0"


def test_column_mapping_documents_provenance():
    mapping = get_adapter("sparkov").column_mapping({})
    assert mapping["entity_id"] == "cc_num"
    assert mapping["amount"] == "amt"
    assert mapping["event_time"] == "trans_date_trans_time"


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fraudTrain.csv"):
        get_adapter("sparkov").to_canonical(tmp_path, {})


def test_sparkov_is_registered():
    from fraud_benchmark.datasets.base import list_datasets
    assert "sparkov" in list_datasets()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sparkov.py -v`
Expected: `UnknownDatasetError: unknown dataset 'sparkov'`

- [ ] **Step 4: Write the implementation** — `src/fraud_benchmark/datasets/sparkov.py`

```python
"""Sparkov (Shenoy): simulated credit-card transactions with real timestamps.

https://www.kaggle.com/datasets/kartik2112/fraud-detection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.datasets.base import DatasetAdapter, register
from fraud_benchmark.datasets.files import require_file
from fraud_benchmark.sources import KaggleDataset

#: The bundle ships a pre-made temporal split. Both halves are labelled and do not
#: overlap (train ends 2020-06-21 12:13:37, test starts 2020-06-21 12:14:25), so they
#: are concatenated and re-split by this project's own rule.
SOURCE_FILES = ("fraudTrain.csv", "fraudTest.csv")


@register
class SparkovAdapter(DatasetAdapter):
    name = "sparkov"
    source = KaggleDataset("kartik2112/fraud-detection")
    data_license = "CC0 1.0"
    commercial_use = True
    caveats = (
        "Simulated with Sparkov/Faker. Customer names, addresses, jobs and dates of "
        "birth are fabricated, not real people.",
        "The upstream bundle ships its own train/test split; this adapter concatenates "
        "both files and applies the project's temporal split instead. The source file "
        "each row came from is kept in 'source_file'.",
        "The per-file row index column ('Unnamed: 0') is dropped as meaningless after "
        "concatenation.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        parts = []
        for filename in SOURCE_FILES:
            part = pd.read_csv(require_file(raw_dir, filename))
            part["source_file"] = filename
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")])

        # The source already has a column called is_fraud, so cast it in place
        # rather than inserting a second one.
        df["is_fraud"] = df["is_fraud"].astype(bool)
        df.insert(0, "event_time", pd.to_datetime(df["trans_date_trans_time"]))
        df.insert(1, "entity_id", df["cc_num"].astype("string"))
        df.insert(2, "amount", df["amt"].astype("float64"))
        return df.sort_values("event_time", kind="stable").reset_index(drop=True)

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        return {
            "event_time": "trans_date_trans_time",
            "entity_id": "cc_num",
            "amount": "amt",
            "is_fraud": "is_fraud",
        }
```

- [ ] **Step 5: Register it**

`src/fraud_benchmark/datasets/__init__.py`:

```python
from fraud_benchmark.datasets import banksim, paysim, sparkov  # noqa: F401
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/test_sparkov.py -v` — expect 14 passed.
Run: `.venv/bin/pytest` — expect 121 passed, 1 deselected.

- [ ] **Step 7: Commit**

```bash
git add src/fraud_benchmark/datasets/sparkov.py src/fraud_benchmark/datasets/__init__.py tests/test_sparkov.py tests/fixtures/sparkov/
git commit -m "feat: Sparkov adapter"
```

---

### Task 5: SAML-D adapter

**Files:**
- Create: `src/fraud_benchmark/datasets/saml_d.py`
- Create: `tests/fixtures/saml_d/SAML-D.csv`
- Modify: `src/fraud_benchmark/datasets/__init__.py`
- Test: `tests/test_saml_d.py`

- [ ] **Step 1: Create the fixture** — `tests/fixtures/saml_d/SAML-D.csv`

```csv
Time,Date,Sender_account,Receiver_account,Amount,Payment_currency,Received_currency,Sender_bank_location,Receiver_bank_location,Payment_type,Is_laundering,Laundering_type
10:35:19,2022-10-07,8724731955,2769355426,1459.15,UK pounds,UK pounds,UK,UK,Cash Deposit,0,Normal_Cash_Deposits
10:35:20,2022-10-07,1491989064,8401255335,6019.64,Indian rupee,Dirham,Albania,UAE,Cross-border,0,Normal_Fan_Out
11:02:44,2022-10-08,287305149,4404767002,14328.44,Albanian lek,Pakistani rupee,Nigeria,Spain,Cheque,1,Smurfing
11:02:45,2022-10-08,287305149,4404767003,14328.44,Albanian lek,Pakistani rupee,Nigeria,Spain,Cheque,1,Smurfing
23:59:59,2022-10-09,8724731955,2769355426,55.10,UK pounds,UK pounds,UK,UK,Cash Deposit,0,Normal_Small_Fan_Out
```

- [ ] **Step 2: Write the failing tests** — `tests/test_saml_d.py`

```python
from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.datasets.base import get_adapter
from fraud_benchmark.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "saml_d"


@pytest.fixture
def frame():
    return get_adapter("saml_d").to_canonical(FIXTURE, {})


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_date_and_time_are_combined(frame):
    assert frame["event_time"].iloc[0] == pd.Timestamp("2022-10-07 10:35:19")
    assert frame["event_time"].iloc[-1] == pd.Timestamp("2022-10-09 23:59:59")


def test_entity_id_is_the_sender_account_as_string(frame):
    assert frame["entity_id"].iloc[0] == "8724731955"
    assert str(frame["entity_id"].dtype) == "string"


def test_is_fraud_comes_from_is_laundering(frame):
    assert frame["is_fraud"].tolist() == [False, False, True, True, False]


def test_laundering_type_is_preserved(frame):
    # Encodes the typology and is likely needed for campaign grouping later.
    assert "Laundering_type" in frame.columns
    assert frame["Laundering_type"].iloc[2] == "Smurfing"


def test_currency_columns_are_preserved_without_conversion(frame):
    assert frame["Payment_currency"].iloc[1] == "Indian rupee"
    assert frame["Received_currency"].iloc[1] == "Dirham"
    # Amount stays in native units.
    assert frame["amount"].iloc[1] == 6019.64


def test_saml_d_is_noncommercial():
    adapter = get_adapter("saml_d")
    assert adapter.commercial_use is False
    assert adapter.data_license == "CC BY-NC-SA 4.0"


def test_column_mapping_documents_provenance():
    mapping = get_adapter("saml_d").column_mapping({})
    assert mapping["entity_id"] == "Sender_account"
    assert mapping["is_fraud"] == "Is_laundering"
    assert "Date" in mapping["event_time"] and "Time" in mapping["event_time"]


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="SAML-D.csv"):
        get_adapter("saml_d").to_canonical(tmp_path, {})


def test_saml_d_is_registered():
    from fraud_benchmark.datasets.base import list_datasets
    assert "saml_d" in list_datasets()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_saml_d.py -v`
Expected: `UnknownDatasetError: unknown dataset 'saml_d'`

- [ ] **Step 4: Write the implementation** — `src/fraud_benchmark/datasets/saml_d.py`

```python
"""SAML-D: a synthetic anti-money-laundering transaction monitoring dataset.

https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.datasets.base import DatasetAdapter, register
from fraud_benchmark.datasets.files import require_file
from fraud_benchmark.sources import KaggleDataset

TRANSACTIONS_FILE = "SAML-D.csv"


@register
class SamlDAdapter(DatasetAdapter):
    name = "saml_d"
    source = KaggleDataset(
        "berkanoztas/synthetic-transaction-monitoring-dataset-aml"
    )
    data_license = "CC BY-NC-SA 4.0"
    commercial_use = False
    caveats = (
        "Synthetic AML data. The label is Is_laundering, and 'fraud' here means a "
        "laundering typology rather than card fraud.",
        "Amounts span 13 currencies (Payment_currency, Received_currency) and are NOT "
        "converted; cross-row amount comparison is not meaningful.",
        "entity_id is the sending account. Laundering is a multi-party phenomenon, so "
        "the receiving account (Receiver_account) matters too and is passed through.",
        "Laundering_type records the typology for both laundering and normal rows "
        "(e.g. Smurfing, Normal_Fan_Out) and is retained.",
        "Licence is CC BY-NC-SA 4.0: NonCommercial and ShareAlike.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        df = pd.read_csv(require_file(raw_dir, TRANSACTIONS_FILE))

        event_time = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str)
        )
        df.insert(0, "event_time", event_time)
        df.insert(1, "entity_id", df["Sender_account"].astype("string"))
        df.insert(2, "amount", df["Amount"].astype("float64"))
        df.insert(3, "is_fraud", df["Is_laundering"].astype(bool))
        return df

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        return {
            "event_time": "Date + Time",
            "entity_id": "Sender_account",
            "amount": "Amount",
            "is_fraud": "Is_laundering",
        }
```

- [ ] **Step 5: Register it**

`src/fraud_benchmark/datasets/__init__.py`:

```python
from fraud_benchmark.datasets import banksim, paysim, saml_d, sparkov  # noqa: F401
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/test_saml_d.py -v` — expect 10 passed.
Run: `.venv/bin/pytest` — expect 131 passed, 1 deselected.

- [ ] **Step 7: Commit**

```bash
git add src/fraud_benchmark/datasets/saml_d.py src/fraud_benchmark/datasets/__init__.py tests/test_saml_d.py tests/fixtures/saml_d/
git commit -m "feat: SAML-D adapter"
```

---

### Task 6: Live verification of all four datasets

The raw data is already cached under `data/raw/`, so no re-download is needed.

- [ ] **Step 1: Prepare each new dataset**

```bash
.venv/bin/fraud-benchmark prepare banksim
.venv/bin/fraud-benchmark prepare sparkov
.venv/bin/fraud-benchmark prepare saml_d
```

Expected: each prints `<name>: wrote data/processed/<name>`.

- [ ] **Step 2: Check the numbers against the verified figures**

```bash
for d in banksim sparkov saml_d; do .venv/bin/fraud-benchmark info $d; done
```

Compare against what was measured from the raw files:

| Dataset | Expected rows | Expected fraud rate |
|---|---|---|
| banksim | 594,643 | 1.211% |
| sparkov | 1,852,394 | ~0.521% (weighted mean of 0.579% and 0.386%) |
| saml_d | 9,504,852 | ~0.118% |

**Report the actual numbers.** If rows differ from these, STOP — it means the adapter is
dropping or duplicating data.

- [ ] **Step 3: Verify splits are sane on real data**

```bash
.venv/bin/python - <<'PY'
import pandas as pd
for name in ("paysim", "banksim", "sparkov", "saml_d"):
    df = pd.read_parquet(f"data/processed/{name}/data.parquet",
                         columns=["event_time", "split", "is_fraud", "entity_id"])
    counts = df["split"].value_counts()
    frac = {k: f"{v/len(df)*100:.2f}%" for k, v in counts.items()}
    tr = df[df.split == "train"]["event_time"].max()
    va = df[df.split == "val"]["event_time"]
    ok = "OK" if va.empty or tr <= va.min() else "LEAK"
    print(f"{name:9s} {len(df):>10,} rows  {frac}  boundary={ok}")
    print(f"          entities={df.entity_id.nunique():,} "
          f"fraud={df.is_fraud.mean()*100:.4f}%")
PY
```

Expect three non-empty splits per dataset and `boundary=OK` for all.

- [ ] **Step 4: Verify the licence gate works end to end**

```bash
.venv/bin/fraud-benchmark list
.venv/bin/fraud-benchmark prepare --all --exclude-noncommercial
```

Expected: `list` marks banksim and saml_d as `[noncommercial]`; the `--all` run skips exactly
those two with a stated reason and prepares paysim and sparkov.

- [ ] **Step 5: Measure campaign groupability for Plan 4**

This directly informs the label-delay design, where PaySim was already measured to have
none.

```bash
.venv/bin/python - <<'PY'
import pandas as pd
print(f"{'dataset':10s} {'frauds':>9s} {'entities':>9s} {'>1 fraud':>9s} {'max':>5s}")
for name in ("paysim", "banksim", "sparkov", "saml_d"):
    df = pd.read_parquet(f"data/processed/{name}/data.parquet",
                         columns=["entity_id", "is_fraud"])
    vc = df[df.is_fraud].entity_id.value_counts()
    print(f"{name:10s} {int(df.is_fraud.sum()):>9,} {vc.size:>9,} "
          f"{int((vc>1).sum()):>9,} {int(vc.max()) if vc.size else 0:>5d}")
PY
```

Record the output — it decides whether entity-keyed campaign grouping is viable per dataset.

- [ ] **Step 6: Commit any fixture or doc updates and report**

```bash
git add -A
git commit -m "test: verify BankSim, Sparkov and SAML-D against real data"
```

---

## Done criteria

- [ ] `.venv/bin/pytest` passes (expect 131 passed, 1 deselected)
- [ ] `fraud-benchmark list` shows 4 datasets with licences, 2 marked `[noncommercial]`
- [ ] `prepare --all --exclude-noncommercial` prepares exactly paysim and sparkov
- [ ] Real row counts match the verified figures above
- [ ] No split boundary leaks any timestamp across splits
- [ ] Campaign groupability measured and recorded for all four

## Not in this plan

- IBM CCF, IEEE-CIS, Amaretto (Plan 3) — these need auxiliary-table joins, the uid
  heuristic, and split-archive extraction respectively
- Label delay (Plan 4)
