# Results

`paper/` holds the records behind the paper's tables. Each readable table is generated
from its records; do not edit it by hand.

| Paper table | Readable table | Records | Runner |
|---|---|---|---|
| IBM CCF temporal vs iid splits | [`ibm_lstm_split_comparison.md`](paper/ibm_lstm_split_comparison.md) | `paper/ibm_lstm_*.json` | `scripts/lstm_protocol.py` |
| Sparkov synthetic label delay | [`sparkov_label_delay.md`](paper/sparkov_label_delay.md) | `paper/sparkov_delay_*.json` | `scripts/sparkov_delay_protocol.py` |

How each table is produced is in [`docs/experiments.md`](../docs/experiments.md).
`archive/` holds other experiments that are not used in the paper
([`docs/archive.md`](../docs/archive.md)). `figures/` holds the dataset plots used in the
documentation.
