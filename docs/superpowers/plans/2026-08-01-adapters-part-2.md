# Adapters Part 2 Implementation Plan (Plan 3 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the three remaining adapters — IBM CCF, IEEE-CIS and Amaretto — plus the git source type and split-archive handling they need, completing all seven datasets.

**Architecture:** Unchanged. Each dataset is one adapter implementing `to_canonical(raw_dir, options)`; all downstream stages are shared. This plan adds one new source type (git), one shared archive helper, and one small pipeline hook for auxiliary output files.

**Tech Stack:** Python 3.13, pandas 3.0.5, pyarrow, kagglehub 1.0.2, pytest.

---

## Verified facts

Every figure below was measured from the actual cached files on 2026-08-01, not recalled.
Raw data is already present under `data/raw/`, so no downloads are needed.

### IBM CCF — `data/raw/ibm_ccf/`

- `credit_card_transactions-ibm_v2.csv`: **24,386,900 rows**, 15 columns, 2.35 GB.
  Fraud rate **0.1220%** (29,757 frauds). Years **1991–2020**. 2,000 users, 6,139 user+card pairs.
- Columns: `User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name,
  Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?`
- `Amount` is a **string with a `$` prefix** (`"$134.09"`). `Is Fraud?` is **`"Yes"`/`"No"`**.
  `Time` is `HH:MM` with **no seconds**.
- `sd254_cards.csv`: 6,146 rows, keyed by `User` + `CARD INDEX`. Columns: `User, CARD INDEX,
  Card Brand, Card Type, Card Number, Expires, CVV, Has Chip, Cards Issued, Credit Limit,
  Acct Open Date, Year PIN last Changed, Card on Dark Web`.
- `sd254_users.csv`: 2,000 rows. **It has no ID column** — the first column is `Person`, a
  name. The join is **positional: row index == `User`**. Verified: 2,000 user rows and
  `cards.User` spans 0–1999 with 2,000 distinct values.
- `Credit Limit`, `Per Capita Income - Zipcode`, `Yearly Income - Person` and `Total Debt`
  are also `$`-prefixed strings.
- **Decision (user, 2026-08-01): join everything.** All 13 card columns and all 18 user
  columns are left-joined onto the transactions. The person-shaped fields (names, addresses,
  card numbers, CVVs) are synthetic, not real people, and a caveat records that.
- Entity: `User` by default, switchable to user+card via `datasets.ibm_ccf.entity_key`.
- Licence: CC BY 4.0 (per the Kaggle page). Commercially usable.

### IEEE-CIS — `data/raw/ieee_cis/`

- `train_transaction.csv`: **590,540 rows x 394 columns**. Fraud rate **3.4990%**.
- `train_identity.csv`: 144,233 rows x 41 columns — covers only **24.4%** of transactions.
  Left-joined on `TransactionID`, leaving nulls for the rest.
- `test_transaction.csv`: 506,691 rows x **393** columns — **`isFraud` is absent**. This is
  the unlabelled competition test set. `test_identity.csv`: 141,907 rows.
- `TransactionDT` is a **seconds offset**, spanning days 1.00–183.00. Anchored to a
  configured `start_date`.
- **Decision (user, 2026-08-01):** the benchmark uses the 590,540 labelled train rows with
  the standard temporal split, AND the unlabelled test rows are written alongside as a
  separate `unlabelled_test.parquet` for semi-supervised work.

#### The uid heuristic, and a real trap

IEEE-CIS has no card identifier. The community constructs a pseudo-id from
`card1` + `addr1` + a `D1`-derived account-start day. Measured:

- `card1` has 0 nulls, but **`addr1` is null in 65,706 rows and `D1` in 1,269** —
  **66,794 rows (11.3%)** are missing at least one component.
- Under **pandas 3, `astype(str)` on NaN yields `<NA>`, and string concatenation propagates
  it.** So the naive heuristic produces **66,794 NULL `entity_id` values**, which
  `validate_canonical` rejects outright. This is not a theoretical risk; it was reproduced.
- Fix: where any component is missing, fall back to a **per-row unique** id
  (`"txn_" + TransactionID`). Verified result: **290,492 distinct entities, 0 nulls,
  largest entity 1,175 rows, 66.0% of rows in multi-row entities.**
- A per-row-unique fallback is deliberate: it is better for 11% of rows to have no
  campaign than for unrelated transactions to be fused into fabricated ones, which is what
  a shared `"nan"` bucket would do to Plan 4.

### Amaretto — `data/raw/amaretto/`

- Git repository, **not Kaggle**: `https://github.com/necst/amaretto_dataset`. Licence
  **MIT** (verified in the repo's LICENSE), so commercially usable.
- `Data/` holds **34 split-zip parts** (`amaretto_dataset_anon.zip.001` … `.034`, 1.2 GB
  total). Plain concatenation in name order produces a valid zip — verified — containing a
  single member, `amaretto_dataset_anon.csv`, **3,577,592,424 bytes**.
- README states 29,704,090 transactions, 81,262 anomalous (0.27%), 400 clients, 60 days.
- Columns: `Transaction ID, Originator, Originator_ID, EntryDate, InputOutput, Market,
  Product ISIN, Product Type, Product Class, Normalized Amount, Currency, Anomaly`
- `EntryDate` is a **real datetime string** (`2019-01-01 17:55:33`) — no anchor needed.
- `Originator` is the client (`Client_087`), 400 distinct — the entity.
- **`Anomaly` is NOT binary.** It has 6 values: `0` plus classes `1`–`5`, matching the five
  FATF typologies in the README. Measured on a 2M-row sample: 0.2542% anomalous, with class
  counts 1:352, 2:178, 3:1006, 4:2269, 5:1279. `is_fraud` is `Anomaly > 0`; the class is
  retained, exactly as `Laundering_type` is for SAML-D.
- `Originator_ID` is the constant `"_XID"` in every row — useless, passed through with a caveat.
- This is **capital-market trading** (buy/sell of securities), not payments. Amounts are
  already normalised; there are 2 currency labels.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/fraud_benchmark/sources.py` | + `GitRepo` source type and its fetcher |
| `src/fraud_benchmark/datasets/files.py` | + split-archive reassembly and member extraction |
| `src/fraud_benchmark/datasets/base.py` | + `auxiliary_frames` hook |
| `src/fraud_benchmark/pipeline.py` | + write auxiliary frames, record them in the card |
| `src/fraud_benchmark/datasets/ibm_ccf.py` | IBM CCF adapter |
| `src/fraud_benchmark/datasets/ieee_cis.py` | IEEE-CIS adapter |
| `src/fraud_benchmark/datasets/amaretto.py` | Amaretto adapter |
| `configs/default.yaml` | + `ibm_ccf`, `ieee_cis` options |

---

### Task 1: Git source type and split-archive handling

**Why:** Amaretto is the only non-Kaggle dataset, and the only one shipping a multi-part zip.

**Files:**
- Modify: `src/fraud_benchmark/sources.py`
- Modify: `src/fraud_benchmark/datasets/files.py`
- Test: `tests/test_sources.py`, `tests/test_files.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_sources.py`:

```python
def test_git_repo_exposes_its_url():
    from fraud_benchmark.sources import GitRepo

    repo = GitRepo("https://github.com/necst/amaretto_dataset")
    assert repo.url == "https://github.com/necst/amaretto_dataset"
    assert repo.ref == "main"


def test_git_clone_is_invoked_with_the_ref(tmp_path, monkeypatch):
    from fraud_benchmark.sources import GitRepo

    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        dest = Path(cmd[-1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text("cloned\n")

        class _Result:
            returncode = 0
            stderr = ""

        return _Result()

    monkeypatch.setattr("fraud_benchmark.sources.subprocess.run", fake_run)

    fetch(GitRepo("https://example.com/repo", ref="v1"), tmp_path / "dest")

    assert "clone" in calls["cmd"]
    assert "v1" in calls["cmd"]
    assert "https://example.com/repo" in calls["cmd"]


def test_git_clone_failure_is_a_fetch_error(tmp_path, monkeypatch):
    from fraud_benchmark.sources import GitRepo

    def fake_run(cmd, **kwargs):
        class _Result:
            returncode = 128
            stderr = "fatal: repository not found"

        return _Result()

    monkeypatch.setattr("fraud_benchmark.sources.subprocess.run", fake_run)

    with pytest.raises(FetchError, match="repository not found"):
        fetch(GitRepo("https://example.com/nope"), tmp_path / "dest")
```

Add to `tests/test_files.py`:

```python
import zipfile


def _make_split_zip(tmp_path, member_name, member_bytes, part_size):
    """Build a real zip, then chop it into numbered parts like the Amaretto release."""
    whole = tmp_path / "whole.zip"
    with zipfile.ZipFile(whole, "w") as zf:
        zf.writestr(member_name, member_bytes)
    data = whole.read_bytes()
    whole.unlink()
    parts_dir = tmp_path / "raw" / "Data"
    parts_dir.mkdir(parents=True)
    index = 1
    for offset in range(0, len(data), part_size):
        (parts_dir / f"archive.zip.{index:03d}").write_bytes(
            data[offset : offset + part_size]
        )
        index += 1
    return tmp_path / "raw"


def test_split_zip_member_is_reassembled_and_extracted(tmp_path):
    from fraud_benchmark.datasets.files import require_split_zip_member

    raw = _make_split_zip(tmp_path, "data.csv", b"a,b\n1,2\n" * 500, part_size=97)
    out = require_split_zip_member(raw, "archive.zip.*", "data.csv", tmp_path / "cache")
    assert out.read_bytes() == b"a,b\n1,2\n" * 500


def test_split_zip_extraction_is_cached(tmp_path):
    from fraud_benchmark.datasets.files import require_split_zip_member

    raw = _make_split_zip(tmp_path, "data.csv", b"x\n" * 100, part_size=64)
    cache = tmp_path / "cache"
    first = require_split_zip_member(raw, "archive.zip.*", "data.csv", cache)
    # Removing the parts must not break a second call: the result is cached.
    for part in (raw / "Data").iterdir():
        part.unlink()
    second = require_split_zip_member(raw, "archive.zip.*", "data.csv", cache)
    assert first == second
    assert second.exists()


def test_split_zip_leaves_no_reassembled_archive(tmp_path):
    from fraud_benchmark.datasets.files import require_split_zip_member

    raw = _make_split_zip(tmp_path, "data.csv", b"y\n" * 100, part_size=64)
    cache = tmp_path / "cache"
    require_split_zip_member(raw, "archive.zip.*", "data.csv", cache)
    assert not list(cache.glob("*.zip"))


def test_missing_split_parts_is_an_error(tmp_path):
    from fraud_benchmark.datasets.files import require_split_zip_member

    empty = tmp_path / "raw"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="archive.zip"):
        require_split_zip_member(empty, "archive.zip.*", "data.csv", tmp_path / "cache")
```

`tests/test_files.py` needs `import pytest` and `from pathlib import Path` at the top if absent.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sources.py tests/test_files.py -v`
Expected: `ImportError: cannot import name 'GitRepo'` and
`cannot import name 'require_split_zip_member'`.

- [ ] **Step 3: Add `GitRepo` to `src/fraud_benchmark/sources.py`**

Add `import subprocess` at the top, then after `KaggleCompetition`:

```python
@dataclass(frozen=True)
class GitRepo:
    """A dataset distributed as a git repository rather than via Kaggle."""

    url: str
    ref: str = "main"
```

Widen the union:

```python
Source = KaggleDataset | KaggleCompetition | GitRepo
```

Add a branch in `fetch`, before the `else` that raises on unsupported types:

```python
        elif isinstance(source, GitRepo):
            _git_clone(source, dest)
```

And the helper, beside `_http_error_message`:

```python
def _git_clone(source: GitRepo, dest: Path) -> None:
    """Shallow-clone `source` into `dest`, which must be empty."""
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", source.ref, source.url, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FetchError(
            f"git clone of {source.url} (ref {source.ref}) failed: {result.stderr.strip()}"
        )
```

- [ ] **Step 4: Add archive handling to `src/fraud_benchmark/datasets/files.py`**

Add `import shutil` and `import zipfile` at the top, then:

```python
def require_split_zip_member(
    raw_dir: Path, part_glob: str, member: str, cache_dir: Path
) -> Path:
    """Reassemble a multi-part zip under `raw_dir` and extract one member.

    Some datasets ship as numbered fragments (`x.zip.001`, `x.zip.002`, ...) because
    of file-size limits. Concatenating them in name order reproduces the original
    archive. The extracted member is cached in `cache_dir`, so the cost is paid once.
    """
    extracted = cache_dir / member
    if extracted.exists():
        return extracted

    parts = sorted(raw_dir.rglob(part_glob))
    if not parts:
        raise FileNotFoundError(
            f"no archive parts matching {part_glob!r} under {raw_dir}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "_reassembled.zip"
    try:
        with open(archive, "wb") as combined:
            for part in parts:
                with open(part, "rb") as fragment:
                    shutil.copyfileobj(fragment, combined)
        with zipfile.ZipFile(archive) as zf:
            zf.extract(member, cache_dir)
    finally:
        # The reassembled archive is a large temporary; never leave it behind.
        archive.unlink(missing_ok=True)

    return extracted
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_sources.py tests/test_files.py -v` — all pass.
Run: `.venv/bin/pytest` — expect 146 passed, 1 deselected.

- [ ] **Step 6: Commit**

```bash
git add src/fraud_benchmark/sources.py src/fraud_benchmark/datasets/files.py \
        tests/test_sources.py tests/test_files.py
git commit -m "feat: git source type and split-archive extraction"
```

---

### Task 2: Auxiliary output frames

**Why:** IEEE-CIS must also write its unlabelled competition test set. That is not a canonical
frame — it has no `is_fraud` — so it cannot go through `validate_canonical` or the splitter.
It needs a separate output file.

**Files:**
- Modify: `src/fraud_benchmark/datasets/base.py`
- Modify: `src/fraud_benchmark/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline.py`:

```python
def test_auxiliary_frames_are_written_alongside(config, no_download, monkeypatch):
    import pandas as pd
    from fraud_benchmark.datasets.base import get_adapter

    adapter = get_adapter("paysim")
    monkeypatch.setattr(
        type(adapter),
        "auxiliary_frames",
        lambda self, raw_dir, options: {"extra": pd.DataFrame({"a": [1, 2, 3]})},
    )
    out = prepare("paysim", config)
    assert (out / "extra.parquet").exists()
    assert len(pd.read_parquet(out / "extra.parquet")) == 3


def test_auxiliary_frames_are_recorded_in_the_card(config, no_download, monkeypatch):
    import pandas as pd
    from fraud_benchmark.datasets.base import get_adapter

    adapter = get_adapter("paysim")
    monkeypatch.setattr(
        type(adapter),
        "auxiliary_frames",
        lambda self, raw_dir, options: {"extra": pd.DataFrame({"a": [1, 2, 3]})},
    )
    out = prepare("paysim", config)
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["auxiliary"] == {"extra": 3}


def test_no_auxiliary_key_when_there_are_none(config, no_download):
    out = prepare("paysim", config)
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["auxiliary"] == {}
    assert not list(out.glob("extra*.parquet"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: the three new tests fail — `auxiliary_frames` does not exist and the card has no
`auxiliary` key.

- [ ] **Step 3: Add the hook to `src/fraud_benchmark/datasets/base.py`**

Inside `DatasetAdapter`, after `custom_splits`:

```python
    def auxiliary_frames(
        self, raw_dir: Path, options: dict[str, Any]
    ) -> dict[str, pd.DataFrame]:
        """Extra frames to write beside the canonical one, keyed by file stem.

        Default: none. Override for data that belongs with the dataset but is not
        canonical — e.g. an unlabelled competition test set, which has no is_fraud
        column and so cannot be validated or split.
        """
        return {}
```

- [ ] **Step 4: Write them in `src/fraud_benchmark/pipeline.py`**

In `prepare`, after the card is built and before the write:

```python
    aux = adapter.auxiliary_frames(raw_dir, options)
    card = _build_card(
        name, adapter, options, df, config, custom_split=supplied is not None
    )
    card["auxiliary"] = {key: int(len(frame)) for key, frame in aux.items()}
    return _write_atomically(config.processed_dir / name, df, card, aux)
```

Change `_write_atomically`'s signature and body to accept them:

```python
def _write_atomically(
    dest: Path,
    df: pd.DataFrame,
    card: dict,
    aux: dict[str, pd.DataFrame] | None = None,
) -> Path:
```

and inside the `try`, immediately after `data.parquet` is written:

```python
        for key, frame in (aux or {}).items():
            frame.to_parquet(staging / f"{key}.parquet", index=False)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_pipeline.py -v` — all pass.
Run: `.venv/bin/pytest` — expect 149 passed, 1 deselected.

- [ ] **Step 6: Commit**

```bash
git add src/fraud_benchmark/datasets/base.py src/fraud_benchmark/pipeline.py \
        tests/test_pipeline.py
git commit -m "feat: adapters may write auxiliary output frames"
```

---

### Task 3: IBM CCF adapter

**Files:**
- Create: `src/fraud_benchmark/datasets/ibm_ccf.py`
- Create: `tests/fixtures/ibm_ccf/credit_card_transactions-ibm_v2.csv`
- Create: `tests/fixtures/ibm_ccf/sd254_cards.csv`
- Create: `tests/fixtures/ibm_ccf/sd254_users.csv`
- Create: `tests/test_ibm_ccf.py`
- Modify: `src/fraud_benchmark/datasets/__init__.py`, `configs/default.yaml`

- [ ] **Step 1: Create the fixtures**

`tests/fixtures/ibm_ccf/credit_card_transactions-ibm_v2.csv`:

```csv
User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,Merchant State,Zip,MCC,Errors?,Is Fraud?
0,0,2002,9,1,06:21,$134.09,Swipe Transaction,3527213246127876953,La Verne,CA,91750.0,5300,,No
0,0,2002,9,1,06:42,$38.48,Swipe Transaction,-727612092139916043,Monterey Park,CA,91754.0,5411,,No
0,1,2002,9,2,17:05,$-25.00,Online Transaction,123456789,Anywhere,CA,91750.0,5411,Bad PIN,Yes
1,0,2010,12,31,23:59,$1200.55,Chip Transaction,987654321,Boston,MA,2101.0,5812,,No
1,0,2011,1,1,00:01,$3.99,Swipe Transaction,987654321,Boston,MA,2101.0,5812,,Yes
```

`tests/fixtures/ibm_ccf/sd254_cards.csv`:

```csv
User,CARD INDEX,Card Brand,Card Type,Card Number,Expires,CVV,Has Chip,Cards Issued,Credit Limit,Acct Open Date,Year PIN last Changed,Card on Dark Web
0,0,Visa,Debit,4344676511950444,12/2022,623,YES,2,$24295,09/2002,2008,No
0,1,Visa,Credit,4956965974959986,12/2020,393,YES,1,$21968,04/2014,2014,No
1,0,Amex,Credit,340071811951482,02/2024,693,NO,1,$12400,01/2009,2009,No
```

`tests/fixtures/ibm_ccf/sd254_users.csv` — note there is no ID column; row order is the User id:

```csv
Person,Current Age,Retirement Age,Birth Year,Birth Month,Gender,Address,Apartment,City,State,Zipcode,Latitude,Longitude,Per Capita Income - Zipcode,Yearly Income - Person,Total Debt,FICO Score,Num Credit Cards
Hazel Robinson,53,66,1966,11,Female,462 Rose Lane,,La Verne,CA,91750,34.15,-117.76,$29278,$59696,$127613,787,5
Sasha Sadr,53,68,1966,12,Female,3606 Federal Boulevard,,Boston,MA,2101,42.34,-71.09,$37891,$77254,$191349,701,5
```

- [ ] **Step 2: Write the failing tests** — `tests/test_ibm_ccf.py`

```python
from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.datasets.base import get_adapter
from fraud_benchmark.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "ibm_ccf"


@pytest.fixture
def frame():
    return get_adapter("ibm_ccf").to_canonical(FIXTURE, {"entity_key": "user"})


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_all_rows_survive(frame):
    assert len(frame) == 5


def test_event_time_is_built_from_the_date_parts(frame):
    assert frame["event_time"].iloc[0] == pd.Timestamp("2002-09-01 06:21")
    assert frame["event_time"].iloc[4] == pd.Timestamp("2011-01-01 00:01")


def test_dollar_amounts_are_parsed(frame):
    assert frame["amount"].iloc[0] == 134.09
    # Negative amounts appear as "$-25.00".
    assert frame["amount"].iloc[2] == -25.00


def test_is_fraud_comes_from_the_yes_no_column(frame):
    assert frame["is_fraud"].tolist() == [False, False, True, False, True]


def test_entity_id_defaults_to_the_user(frame):
    assert frame["entity_id"].tolist() == ["0", "0", "0", "1", "1"]


def test_entity_key_can_be_switched_to_card():
    frame = get_adapter("ibm_ccf").to_canonical(FIXTURE, {"entity_key": "card"})
    # User 0 has two cards, so its rows must now split into two entities.
    assert frame["entity_id"].tolist() == ["0-0", "0-0", "0-1", "1-0", "1-0"]


def test_unknown_entity_key_is_an_error():
    with pytest.raises(ValueError, match="entity_key"):
        get_adapter("ibm_ccf").to_canonical(FIXTURE, {"entity_key": "nonsense"})


def test_card_attributes_are_joined(frame):
    assert frame["Card Brand"].tolist()[:3] == ["Visa", "Visa", "Visa"]
    assert frame["Card Type"].iloc[2] == "Credit"
    assert frame["Card Type"].iloc[0] == "Debit"


def test_user_attributes_are_joined_positionally(frame):
    # sd254_users.csv has no ID column; row 0 is User 0.
    assert frame["Person"].iloc[0] == "Hazel Robinson"
    assert frame["Person"].iloc[3] == "Sasha Sadr"
    assert frame["FICO Score"].iloc[0] == 787


def test_dollar_columns_in_the_joined_tables_are_parsed(frame):
    assert frame["Credit Limit"].iloc[0] == 24295.0
    assert frame["Total Debt"].iloc[0] == 127613.0
    assert frame["Yearly Income - Person"].iloc[0] == 59696.0
    assert frame["Per Capita Income - Zipcode"].iloc[0] == 29278.0


def test_the_card_index_join_key_is_not_duplicated(frame):
    assert "CARD INDEX" not in frame.columns


def test_ibm_ccf_is_commercially_usable():
    adapter = get_adapter("ibm_ccf")
    assert adapter.commercial_use is True


def test_column_mapping_documents_provenance():
    mapping = get_adapter("ibm_ccf").column_mapping({"entity_key": "user"})
    assert mapping["amount"] == "Amount"
    assert mapping["is_fraud"] == "Is Fraud?"
    assert "Year" in mapping["event_time"]


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="credit_card_transactions"):
        get_adapter("ibm_ccf").to_canonical(tmp_path, {"entity_key": "user"})


def test_ibm_ccf_is_registered():
    from fraud_benchmark.datasets.base import list_datasets

    assert "ibm_ccf" in list_datasets()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ibm_ccf.py -v`
Expected: `UnknownDatasetError: unknown dataset 'ibm_ccf'`.

- [ ] **Step 4: Write the implementation** — `src/fraud_benchmark/datasets/ibm_ccf.py`

```python
"""IBM CCF (Altman): a large synthetic credit-card transaction log.

https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.datasets.base import DatasetAdapter, register
from fraud_benchmark.datasets.files import require_file
from fraud_benchmark.sources import KaggleDataset

TRANSACTIONS_FILE = "credit_card_transactions-ibm_v2.csv"
CARDS_FILE = "sd254_cards.csv"
USERS_FILE = "sd254_users.csv"

#: Columns in the joined tables that are money strings such as "$24295".
MONEY_COLUMNS = (
    "Credit Limit",
    "Per Capita Income - Zipcode",
    "Yearly Income - Person",
    "Total Debt",
)


def _parse_money(series: pd.Series) -> pd.Series:
    """Turn "$134.09" / "$-25.00" into a float."""
    return series.astype(str).str.replace("$", "", regex=False).astype("float64")


@register
class IbmCcfAdapter(DatasetAdapter):
    name = "ibm_ccf"
    source = KaggleDataset("ealtman2019/credit-card-transactions")
    data_license = "CC BY 4.0"
    commercial_use = True
    caveats = (
        "Fully synthetic. The bundled cardholder details — names, addresses, card "
        "numbers, CVVs — are fabricated and do not describe real people.",
        "All 13 card columns and all 18 user columns are left-joined onto every "
        "transaction, so static attributes repeat across a user's rows.",
        "sd254_users.csv has no identifier column; it is joined positionally, its row "
        "index being the User id.",
        "Money columns in the joined tables ($-prefixed strings) are parsed to float.",
        "Timestamps have minute granularity; the source has no seconds.",
        "entity_key defaults to 'user'. A user may hold several cards, so set it to "
        "'card' to key on the individual card instead.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        entity_key = options.get("entity_key", "user")
        if entity_key not in ("user", "card"):
            raise ValueError(
                f"ibm_ccf entity_key must be 'user' or 'card', got {entity_key!r}"
            )

        df = pd.read_csv(require_file(raw_dir, TRANSACTIONS_FILE))
        cards = pd.read_csv(require_file(raw_dir, CARDS_FILE))
        users = pd.read_csv(require_file(raw_dir, USERS_FILE))

        # sd254_users.csv carries no id; its row order is the User id.
        users = users.copy()
        users.insert(0, "User", range(len(users)))

        df = df.merge(
            cards, how="left", left_on=["User", "Card"], right_on=["User", "CARD INDEX"]
        ).drop(columns=["CARD INDEX"])
        df = df.merge(users, how="left", on="User")

        for column in MONEY_COLUMNS:
            if column in df.columns:
                df[column] = _parse_money(df[column])

        event_time = pd.to_datetime(
            df["Year"].astype(str)
            + "-"
            + df["Month"].astype(str).str.zfill(2)
            + "-"
            + df["Day"].astype(str).str.zfill(2)
            + " "
            + df["Time"].astype(str),
            format="%Y-%m-%d %H:%M",
        )

        if entity_key == "user":
            entity = df["User"].astype("string")
        else:
            entity = (df["User"].astype(str) + "-" + df["Card"].astype(str)).astype(
                "string"
            )

        amount = _parse_money(df["Amount"])
        is_fraud = df["Is Fraud?"].astype(str).str.strip().eq("Yes")

        df.insert(0, "event_time", event_time)
        df.insert(1, "entity_id", entity)
        df.insert(2, "amount", amount)
        df.insert(3, "is_fraud", is_fraud)
        return df

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        entity_key = options.get("entity_key", "user")
        return {
            "event_time": "Year + Month + Day + Time",
            "entity_id": "User" if entity_key == "user" else "User + Card",
            "amount": "Amount",
            "is_fraud": "Is Fraud?",
        }
```

- [ ] **Step 5: Register it and add config**

`src/fraud_benchmark/datasets/__init__.py`:

```python
from fraud_benchmark.datasets import (  # noqa: F401
    banksim,
    ibm_ccf,
    paysim,
    saml_d,
    sparkov,
)
```

In `configs/default.yaml`, under `datasets:`:

```yaml
  # A user may hold several cards; 'user' groups them, 'card' keeps them separate.
  ibm_ccf:
    entity_key: user
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/test_ibm_ccf.py -v` — expect 16 passed.
Run: `.venv/bin/pytest` — expect 165 passed, 1 deselected.

Note: `tests/test_cli.py`'s `no_download` fixture resolves `tests/fixtures/<dataset name>`,
so `prepare --all` there will now also prepare ibm_ccf from your fixture. Its five rows span
2002–2011 with 4 distinct timestamps, enough for a three-way split.

- [ ] **Step 7: Commit**

```bash
git add src/fraud_benchmark/datasets/ibm_ccf.py src/fraud_benchmark/datasets/__init__.py \
        configs/default.yaml tests/test_ibm_ccf.py tests/fixtures/ibm_ccf/
git commit -m "feat: IBM CCF adapter"
```

---

### Task 4: IEEE-CIS adapter

**Files:**
- Create: `src/fraud_benchmark/datasets/ieee_cis.py`
- Create: `tests/fixtures/ieee_cis/train_transaction.csv`, `train_identity.csv`,
  `test_transaction.csv`, `test_identity.csv`
- Create: `tests/test_ieee_cis.py`
- Modify: `src/fraud_benchmark/datasets/__init__.py`, `configs/default.yaml`

- [ ] **Step 1: Create the fixtures**

`tests/fixtures/ieee_cis/train_transaction.csv` — row 3 deliberately has a missing `addr1`
and row 4 a missing `D1`, to exercise the null-safe uid fallback:

```csv
TransactionID,isFraud,TransactionDT,TransactionAmt,ProductCD,card1,card2,addr1,D1,C1
2987000,0,86400,68.5,W,13926,,315.0,14.0,1.0
2987001,0,86401,29.0,W,13926,404.0,315.0,14.0,1.0
2987002,1,172800,59.0,W,2755,404.0,,0.0,1.0
2987003,0,259200,50.0,H,4663,490.0,330.0,,1.0
2987004,1,345600,117.0,W,13926,404.0,315.0,17.0,2.0
```

Note row 4's `D1` is 17, not 14: `D1` counts days since the card was first seen, so for the
same card it must grow in step with `TransactionDT`. Day 4 minus 17 gives the same account
start (-13) as day 1 minus 14, which is exactly what makes those three rows one entity. A
fixture with a constant `D1` would silently fail to group and make the test meaningless.

`tests/fixtures/ieee_cis/train_identity.csv` — covers only two of the five transactions:

```csv
TransactionID,id_01,id_02,id_12,id_31
2987001,0.0,70787.0,NotFound,samsung browser 6.2
2987004,-5.0,98945.0,NotFound,mobile safari 11.0
```

`tests/fixtures/ieee_cis/test_transaction.csv` — note there is no `isFraud` column:

```csv
TransactionID,TransactionDT,TransactionAmt,ProductCD,card1,card2,addr1,D1,C1
3663549,18403224,31.95,W,10409,111.0,325.0,0.0,1.0
3663550,18403263,49.0,W,4272,111.0,325.0,0.0,1.0
```

`tests/fixtures/ieee_cis/test_identity.csv`:

```csv
TransactionID,id_01,id_02,id_12,id_31
3663549,0.0,70787.0,NotFound,chrome 62.0
```

- [ ] **Step 2: Write the failing tests** — `tests/test_ieee_cis.py`

```python
from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.datasets.base import get_adapter
from fraud_benchmark.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "ieee_cis"
OPTIONS = {"start_date": "2017-12-01"}


@pytest.fixture
def frame():
    return get_adapter("ieee_cis").to_canonical(FIXTURE, OPTIONS)


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_only_labelled_train_rows_are_used(frame):
    # The 2 unlabelled test rows must not appear in the canonical frame.
    assert len(frame) == 5
    assert 3663549 not in set(frame["TransactionID"])


def test_transaction_dt_is_anchored_to_start_date(frame):
    # TransactionDT is a seconds offset; 86400 is one day past the anchor.
    assert frame["event_time"].iloc[0] == pd.Timestamp("2017-12-02 00:00:00")
    assert frame["event_time"].iloc[4] == pd.Timestamp("2017-12-05 00:00:00")


def test_identity_columns_are_left_joined(frame):
    assert "id_31" in frame.columns
    joined = frame.set_index("TransactionID")["id_31"]
    assert joined.loc[2987001] == "samsung browser 6.2"
    # Transactions without identity data keep nulls rather than being dropped.
    assert pd.isna(joined.loc[2987000])


def test_uid_groups_repeat_customers(frame):
    # Rows 0, 1 and 4 share card1 and addr1, and their D1 values track TransactionDT
    # so all three resolve to the same account start — one entity.
    uids = frame.set_index("TransactionID")["entity_id"]
    assert uids.loc[2987000] == uids.loc[2987001] == uids.loc[2987004]
    # A different card must not collide with them.
    assert uids.loc[2987000] != uids.loc[2987003]


def test_rows_missing_a_uid_component_get_a_unique_identity(frame):
    """addr1 or D1 being null must not fuse unrelated rows into one entity.

    Under pandas 3, astype(str) on NaN yields <NA> and propagates through
    concatenation, so the naive heuristic would emit NULL entity_id and fail
    validation outright.
    """
    uids = frame.set_index("TransactionID")["entity_id"]
    assert uids.loc[2987002] == "txn_2987002"  # missing addr1
    assert uids.loc[2987003] == "txn_2987003"  # missing D1


def test_no_entity_id_is_null(frame):
    assert frame["entity_id"].notna().all()


def test_amount_comes_from_transaction_amt(frame):
    assert frame["amount"].iloc[0] == 68.5


def test_is_fraud_is_boolean(frame):
    assert frame["is_fraud"].tolist() == [False, False, True, False, True]


def test_unlabelled_test_set_is_offered_as_an_auxiliary_frame():
    aux = get_adapter("ieee_cis").auxiliary_frames(FIXTURE, OPTIONS)
    assert set(aux) == {"unlabelled_test"}
    test = aux["unlabelled_test"]
    assert len(test) == 2
    assert "isFraud" not in test.columns
    # It gets a usable timestamp even though it is not canonical.
    assert "event_time" in test.columns
    # And its identity table is joined too.
    assert "id_31" in test.columns


def test_missing_start_date_is_an_error():
    with pytest.raises(ValueError, match="start_date"):
        get_adapter("ieee_cis").to_canonical(FIXTURE, {})


def test_ieee_cis_is_registered():
    from fraud_benchmark.datasets.base import list_datasets

    assert "ieee_cis" in list_datasets()


def test_column_mapping_records_the_uid_heuristic():
    mapping = get_adapter("ieee_cis").column_mapping(OPTIONS)
    assert "card1" in mapping["entity_id"]
    assert mapping["is_fraud"] == "isFraud"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ieee_cis.py -v`
Expected: `UnknownDatasetError: unknown dataset 'ieee_cis'`.

- [ ] **Step 4: Write the implementation** — `src/fraud_benchmark/datasets/ieee_cis.py`

```python
"""IEEE-CIS / Vesta: real e-commerce transactions from a Kaggle competition.

https://www.kaggle.com/c/ieee-fraud-detection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.datasets.base import DatasetAdapter, register
from fraud_benchmark.datasets.files import require_file
from fraud_benchmark.sources import KaggleCompetition

TRAIN_TRANSACTION = "train_transaction.csv"
TRAIN_IDENTITY = "train_identity.csv"
TEST_TRANSACTION = "test_transaction.csv"
TEST_IDENTITY = "test_identity.csv"

SECONDS_PER_DAY = 86_400


def _require_start_date(options: dict[str, Any]) -> pd.Timestamp:
    start_date = options.get("start_date")
    if not start_date:
        raise ValueError(
            "ieee_cis requires a 'start_date' option to anchor its relative "
            "'TransactionDT' offsets; set datasets.ieee_cis.start_date in the config"
        )
    return pd.Timestamp(start_date)


def _load_with_identity(raw_dir: Path, transactions: str, identity: str) -> pd.DataFrame:
    df = pd.read_csv(require_file(raw_dir, transactions))
    ids = pd.read_csv(require_file(raw_dir, identity))
    return df.merge(ids, how="left", on="TransactionID")


def build_uid(df: pd.DataFrame) -> pd.Series:
    """Derive a pseudo card identifier, falling back to a per-row unique id.

    IEEE-CIS has no card identifier. The community heuristic combines card1, addr1
    and a D1-derived account start day. Two things matter here:

    1. addr1 is null in ~11% of rows and D1 in a few thousand. Under pandas 3,
       astype(str) on NaN yields <NA> and concatenation propagates it, so the naive
       heuristic produces NULL entity_id values that fail schema validation.
    2. Bucketing those rows together under a shared "nan" key would be worse than
       useless — it would fabricate campaigns out of unrelated transactions.

    So rows missing any component get their own identity instead.
    """
    day = df["TransactionDT"] / SECONDS_PER_DAY
    account_start = (day - df["D1"]).round()
    uid = (
        df["card1"].astype(str)
        + "_"
        + df["addr1"].astype(str)
        + "_"
        + account_start.astype(str)
    )
    incomplete = df["addr1"].isna() | df["D1"].isna()
    return uid.where(~incomplete, "txn_" + df["TransactionID"].astype(str))


@register
class IeeeCisAdapter(DatasetAdapter):
    name = "ieee_cis"
    source = KaggleCompetition("ieee-fraud-detection")
    data_license = "Competition rules (research use)"
    commercial_use = False
    caveats = (
        "The only non-synthetic dataset here: real Vesta e-commerce transactions, "
        "heavily anonymised.",
        "entity_id is a DERIVED pseudo-identifier, not a real card id. It combines "
        "card1, addr1 and a D1-derived account start day — the well-known community "
        "'uid' heuristic. Rows missing addr1 or D1 (~11%) get a per-row unique id "
        "rather than being fused into a shared bucket.",
        "TransactionDT is a seconds offset with no stated origin, so event_time is "
        "anchored to a configured start_date and absolute dates carry no meaning.",
        "The competition test set has no labels and is therefore excluded from the "
        "benchmark; it is written separately as unlabelled_test.parquet.",
        "Identity data covers only about a quarter of transactions; the rest are null.",
        "Use is governed by the Kaggle competition rules, which must be accepted "
        "before download.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        anchor = _require_start_date(options)
        df = _load_with_identity(raw_dir, TRAIN_TRANSACTION, TRAIN_IDENTITY)

        event_time = anchor + pd.to_timedelta(df["TransactionDT"], unit="s")
        df.insert(0, "event_time", event_time)
        df.insert(1, "entity_id", build_uid(df).astype("string"))
        df.insert(2, "amount", df["TransactionAmt"].astype("float64"))
        df.insert(3, "is_fraud", df["isFraud"].astype(bool))
        return df

    def auxiliary_frames(
        self, raw_dir: Path, options: dict[str, Any]
    ) -> dict[str, pd.DataFrame]:
        """The unlabelled competition test set, timestamped the same way."""
        anchor = _require_start_date(options)
        test = _load_with_identity(raw_dir, TEST_TRANSACTION, TEST_IDENTITY)
        test.insert(0, "event_time", anchor + pd.to_timedelta(test["TransactionDT"], unit="s"))
        test.insert(1, "entity_id", build_uid(test).astype("string"))
        return {"unlabelled_test": test}

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        return {
            "event_time": f"TransactionDT (seconds since {options.get('start_date')})",
            "entity_id": "derived uid: card1 + addr1 + (day - D1)",
            "amount": "TransactionAmt",
            "is_fraud": "isFraud",
        }
```

- [ ] **Step 5: Register it and add config**

`src/fraud_benchmark/datasets/__init__.py`:

```python
from fraud_benchmark.datasets import (  # noqa: F401
    banksim,
    ibm_ccf,
    ieee_cis,
    paysim,
    saml_d,
    sparkov,
)
```

In `configs/default.yaml`, under `datasets:`:

```yaml
  # TransactionDT is a seconds offset with no stated origin.
  ieee_cis:
    start_date: "2017-12-01"
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/test_ieee_cis.py -v` — expect 13 passed.
Run: `.venv/bin/pytest` — expect 178 passed, 1 deselected.

- [ ] **Step 7: Commit**

```bash
git add src/fraud_benchmark/datasets/ieee_cis.py src/fraud_benchmark/datasets/__init__.py \
        configs/default.yaml tests/test_ieee_cis.py tests/fixtures/ieee_cis/
git commit -m "feat: IEEE-CIS adapter with a null-safe uid heuristic"
```

---

### Task 5: Amaretto adapter

**Files:**
- Create: `src/fraud_benchmark/datasets/amaretto.py`
- Create: `tests/fixtures/amaretto/Data/` (a real split zip, built by the test)
- Create: `tests/test_amaretto.py`
- Modify: `src/fraud_benchmark/datasets/__init__.py`

**Note on the fixture:** a split zip is binary, but it must still be COMMITTED under
`tests/fixtures/amaretto/Data/`. `tests/test_cli.py`'s `no_download` fixture resolves each
dataset to `tests/fixtures/<name>`, so if amaretto has no fixture directory,
`test_prepare_all_creates_output` will fail the moment the adapter is registered. The parts
are only a few hundred bytes, so committing them is cheap and keeps `--all` honest.

- [ ] **Step 1: Generate and commit the fixture**

Run this once to build a genuine multi-part archive, mirroring how Amaretto ships:

```bash
.venv/bin/python - <<'PY'
import io, zipfile
from pathlib import Path

CSV = """Transaction ID,Originator,Originator_ID,EntryDate,InputOutput,Market,Product ISIN,Product Type,Product Class,Normalized Amount,Currency,Anomaly
I9Q3S5YYLCQX,Client_087,_XID,2019-01-01 17:55:33,Sell,Market2,ISN-X01-LRBXBXN,FutureCommodity,Trade,10317357.93,Currency1,0
VFZ6INAXVYJV,Client_019,_XID,2019-01-01 18:52:34,Sell,Market1,ISN-X01-856Z8OC,FX,Trade,31042.04,Currency2,0
HXOA8DNPX1OE,Client_385,_XID,2019-01-02 15:21:09,Sell,Market1,ISN-X01-5CW2HUF,SimpleTransfer,Trade,35910.53,Currency1,3
O9W5IR932XT1,Client_276,_XID,2019-01-03 20:02:36,Buy,Market1,ISN-X01-AQ848H7,FutureEquity,Trade,79630.92,Currency2,5
84KF31TPK1LU,Client_049,_XID,2019-01-04 09:06:58,Buy,Market1,ISN-X01-7JWB2C7,FX,Trade,434370.69,Currency1,0
"""

buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("amaretto_dataset_anon.csv", CSV)
data = buffer.getvalue()

out = Path("tests/fixtures/amaretto/Data")
out.mkdir(parents=True, exist_ok=True)
size = max(1, len(data) // 3 + 1)
for index, offset in enumerate(range(0, len(data), size), start=1):
    (out / f"amaretto_dataset_anon.zip.{index:03d}").write_bytes(data[offset:offset + size])
print("wrote", sorted(p.name for p in out.iterdir()))
PY
```

Expected: it prints three or four part filenames. Verify they reassemble:

```bash
cat tests/fixtures/amaretto/Data/amaretto_dataset_anon.zip.??? > /tmp/check.zip && unzip -l /tmp/check.zip && rm /tmp/check.zip
```

Expected: one member, `amaretto_dataset_anon.csv`.

- [ ] **Step 2: Write the failing tests** — `tests/test_amaretto.py`

```python
import shutil
from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.datasets.base import get_adapter
from fraud_benchmark.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "amaretto"


@pytest.fixture
def raw_dir(tmp_path):
    """A disposable copy of the committed fixture.

    Copied rather than used in place because extraction writes a cache directory
    beside the archive parts, and one test deletes the parts outright.
    """
    target = tmp_path / "amaretto"
    shutil.copytree(FIXTURE, target)
    return target


@pytest.fixture
def frame(raw_dir):
    return get_adapter("amaretto").to_canonical(raw_dir, {})


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_all_rows_are_read_from_the_split_archive(frame):
    assert len(frame) == 5


def test_event_time_is_the_real_timestamp(frame):
    assert frame["event_time"].iloc[0] == pd.Timestamp("2019-01-01 17:55:33")


def test_entity_id_is_the_originator(frame):
    assert frame["entity_id"].iloc[0] == "Client_087"
    assert str(frame["entity_id"].dtype) == "string"


def test_amount_comes_from_normalized_amount(frame):
    assert frame["amount"].iloc[0] == 10317357.93


def test_any_nonzero_anomaly_class_counts_as_fraud(frame):
    # Anomaly is 0 plus five FATF typologies, not a boolean.
    assert frame["is_fraud"].tolist() == [False, False, True, True, False]


def test_the_anomaly_class_is_preserved(frame):
    # Plan 4 may key campaign grouping on the typology, as with SAML-D.
    assert frame["Anomaly"].tolist() == [0, 0, 3, 5, 0]


def test_passthrough_columns_survive(frame):
    for column in ("Market", "Product Type", "Currency", "InputOutput"):
        assert column in frame.columns


def test_extraction_is_cached_between_calls(raw_dir):
    adapter = get_adapter("amaretto")
    first = adapter.to_canonical(raw_dir, {})
    for part in (raw_dir / "Data").iterdir():
        part.unlink()
    # The extracted CSV is cached under raw_dir, so a second call still works.
    second = adapter.to_canonical(raw_dir, {})
    assert len(first) == len(second)


def test_amaretto_is_mit_licensed_and_commercial():
    adapter = get_adapter("amaretto")
    assert adapter.commercial_use is True
    assert adapter.data_license == "MIT"


def test_source_is_a_git_repo():
    from fraud_benchmark.sources import GitRepo

    assert isinstance(get_adapter("amaretto").source, GitRepo)


def test_missing_archive_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="amaretto_dataset_anon.zip"):
        get_adapter("amaretto").to_canonical(tmp_path, {})


def test_column_mapping_documents_provenance():
    mapping = get_adapter("amaretto").column_mapping({})
    assert mapping["entity_id"] == "Originator"
    assert mapping["amount"] == "Normalized Amount"
    assert mapping["is_fraud"] == "Anomaly > 0"


def test_amaretto_is_registered():
    from fraud_benchmark.datasets.base import list_datasets

    assert "amaretto" in list_datasets()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_amaretto.py -v`
Expected: `UnknownDatasetError: unknown dataset 'amaretto'`.

- [ ] **Step 4: Write the implementation** — `src/fraud_benchmark/datasets/amaretto.py`

```python
"""Amaretto: a synthetic capital-market dataset for money-laundering detection.

https://github.com/necst/amaretto_dataset

Unlike the other datasets here this is securities trading, not payments: clients
buy and sell instruments on a market.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.datasets.base import DatasetAdapter, register
from fraud_benchmark.datasets.files import require_split_zip_member
from fraud_benchmark.sources import GitRepo

ARCHIVE_PARTS = "amaretto_dataset_anon.zip.*"
ARCHIVE_MEMBER = "amaretto_dataset_anon.csv"

#: Where the extracted CSV is cached, relative to the raw directory. Kept inside
#: data/raw so it is gitignored and survives between runs.
EXTRACT_DIR = "_extracted"


@register
class AmarettoAdapter(DatasetAdapter):
    name = "amaretto"
    source = GitRepo("https://github.com/necst/amaretto_dataset")
    data_license = "MIT"
    commercial_use = True
    caveats = (
        "Capital-market trading data, not payments: rows are securities buy/sell "
        "orders, so 'amount' is a normalised trade value rather than a transfer.",
        "The label column Anomaly is NOT binary. It is 0 plus five classes matching "
        "the FATF typologies described upstream; is_fraud is Anomaly > 0 and the "
        "class itself is retained.",
        "Originator_ID is the constant '_XID' in every row and carries no information.",
        "Distributed as a 34-part split zip inside a git repository; the adapter "
        "reassembles and extracts it once, caching the result under data/raw.",
        "Fully synthetic, built from aggregate real market parameters.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        csv_path = require_split_zip_member(
            raw_dir, ARCHIVE_PARTS, ARCHIVE_MEMBER, raw_dir / EXTRACT_DIR
        )
        df = pd.read_csv(csv_path)

        df.insert(0, "event_time", pd.to_datetime(df["EntryDate"]))
        df.insert(1, "entity_id", df["Originator"].astype("string"))
        df.insert(2, "amount", df["Normalized Amount"].astype("float64"))
        df.insert(3, "is_fraud", df["Anomaly"].gt(0))
        return df

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        return {
            "event_time": "EntryDate",
            "entity_id": "Originator",
            "amount": "Normalized Amount",
            "is_fraud": "Anomaly > 0",
        }
```

- [ ] **Step 5: Register it**

`src/fraud_benchmark/datasets/__init__.py`:

```python
from fraud_benchmark.datasets import (  # noqa: F401
    amaretto,
    banksim,
    ibm_ccf,
    ieee_cis,
    paysim,
    saml_d,
    sparkov,
)
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/test_amaretto.py -v` — expect 14 passed.
Run: `.venv/bin/pytest` — expect 192 passed, 1 deselected.

`tests/test_cli.py::test_prepare_all_creates_output` exercises `prepare --all`, which now
includes amaretto. It works because Step 1 committed a real fixture at
`tests/fixtures/amaretto/`. If it fails anyway, STOP and report — do not weaken that test.

- [ ] **Step 7: Commit**

```bash
git add src/fraud_benchmark/datasets/amaretto.py src/fraud_benchmark/datasets/__init__.py \
        tests/test_amaretto.py tests/fixtures/amaretto/
git commit -m "feat: Amaretto adapter"
```

---

### Task 6: Live verification of all seven datasets

Raw data for every dataset is already cached under `data/raw/`.

- [ ] **Step 1: Prepare the three new datasets**

```bash
.venv/bin/fraud-benchmark prepare ibm_ccf
.venv/bin/fraud-benchmark prepare ieee_cis
.venv/bin/fraud-benchmark prepare amaretto
```

Amaretto extracts a 3.58 GB CSV on first run, so expect it to take several minutes.
IBM CCF reads 24.4M rows and joins two tables.

- [ ] **Step 2: Check against the measured figures**

```bash
.venv/bin/python - <<'PY'
import json
EXPECTED = {
    "ibm_ccf":  (24_386_900, 0.1220),
    "ieee_cis": (   590_540, 3.4990),
    "amaretto": (29_704_090, 0.2700),   # README figure; report the real one
}
for name, (rows, rate) in EXPECTED.items():
    c = json.load(open(f"data/processed/{name}/dataset_card.json"))
    print(f"{name:9s} rows={c['n_rows']:>12,} (exp {rows:>12,})  "
          f"fraud={c['fraud_rate']*100:.4f}% (exp {rate:.4f}%)  "
          f"entities={c['n_entities']:,}")
    print(f"{'':9s} splits={c['split']['counts']}  aux={c.get('auxiliary')}")
PY
```

IBM CCF and IEEE-CIS row counts must match exactly. Amaretto's is a README claim rather than
something measured here — **report the actual number** and note any difference rather than
assuming the README is right.

- [ ] **Step 3: Confirm the IEEE-CIS auxiliary file**

```bash
.venv/bin/python -c "
import pandas as pd
t = pd.read_parquet('data/processed/ieee_cis/unlabelled_test.parquet')
print('unlabelled test rows:', f'{len(t):,}', '(expected 506,691)')
print('has isFraud?', 'isFraud' in t.columns, '(must be False)')
print('has event_time?', 'event_time' in t.columns)
"
```

- [ ] **Step 4: Verify splits and null-free entities on real data**

```bash
.venv/bin/python - <<'PY'
import pandas as pd
for name in ("paysim","banksim","sparkov","saml_d","ibm_ccf","ieee_cis","amaretto"):
    df = pd.read_parquet(f"data/processed/{name}/data.parquet",
                         columns=["event_time","split","is_fraud","entity_id"])
    frac = {k: f"{v/len(df)*100:.1f}%" for k, v in df["split"].value_counts().items()}
    straddle = int((df.groupby("event_time")["split"].nunique() > 1).sum())
    print(f"{name:9s} {len(df):>11,}  {frac}")
    print(f"{'':9s} entities={df.entity_id.nunique():>9,} "
          f"null_ids={int(df.entity_id.isna().sum())} straddling={straddle}")
PY
```

Every dataset must show `null_ids=0` and `straddling=0`.

- [ ] **Step 5: Measure campaign groupability for all seven**

```bash
.venv/bin/python - <<'PY'
import pandas as pd
print(f"{'dataset':10s} {'frauds':>9s} {'fraud ents':>11s} {'>1 fraud':>9s} {'%grouped':>9s} {'max':>6s}")
for name in ("paysim","banksim","sparkov","saml_d","ibm_ccf","ieee_cis","amaretto"):
    df = pd.read_parquet(f"data/processed/{name}/data.parquet",
                         columns=["entity_id","is_fraud"])
    f = df[df.is_fraud]
    vc = f.entity_id.value_counts()
    grouped = int(vc[vc > 1].sum())
    pct = grouped / len(f) * 100 if len(f) else 0.0
    print(f"{name:10s} {len(f):>9,} {vc.size:>11,} {int((vc>1).sum()):>9,} "
          f"{pct:>8.1f}% {int(vc.max()) if vc.size else 0:>6,}")
PY
```

This completes the per-dataset picture that Plan 4 needs. Record it.

- [ ] **Step 6: Check the licence gate with all seven registered**

```bash
.venv/bin/fraud-benchmark list
```

Expect seven rows. banksim, saml_d and ieee_cis are non-commercial; paysim, sparkov,
ibm_ccf and amaretto are not flagged.

- [ ] **Step 7: Commit any corrections and report**

```bash
git add -u
git commit -m "test: verify IBM CCF, IEEE-CIS and Amaretto against real data"
```

Report actual numbers. If a row count differs from the figures above, say so plainly rather
than rounding to agreement.

---

## Done criteria

- [ ] `.venv/bin/pytest` passes (expect 192 passed, 1 deselected)
- [ ] `fraud-benchmark list` shows all seven datasets with licences
- [ ] IBM CCF and IEEE-CIS real row counts match exactly; Amaretto's is reported
- [ ] `data/processed/ieee_cis/unlabelled_test.parquet` has 506,691 rows and no `isFraud`
- [ ] No dataset has a null `entity_id` or a timestamp straddling a split boundary
- [ ] Campaign groupability recorded for all seven

## Not in this plan

- Label delay, `reported_at`, and campaign grouping (Plan 4)
- Any FX conversion of amounts — explicitly out of scope per the design spec
