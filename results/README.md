# Results

`paper/` holds the records behind the paper's tables. Each readable table is generated
from its records; do not edit it by hand.

| Paper table | Readable table | Records | Runner |
|---|---|---|---|
| Table 3: IBM CCF temporal vs iid splits | [`ibm_lstm_split_comparison.md`](paper/ibm_lstm_split_comparison.md) | `paper/ibm_lstm_*.json` | `scripts/lstm_protocol.py` |
| Table 4: Sparkov synthetic label delay | [`sparkov_label_delay.md`](paper/sparkov_label_delay.md) | `paper/sparkov_lstm_delay/` (seeds 0–4, then `seed5/`–`seed9/`) | `scripts/lstm_protocol.py` |

How each table is produced is in [`docs/experiments.md`](../docs/experiments.md).
`archive/` holds other experiments that are not used in the paper
([`docs/archive.md`](../docs/archive.md)). `figures/` holds the dataset plots used in the
documentation.
