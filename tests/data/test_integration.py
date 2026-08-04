"""Live download test. Deselected by default; run with: pytest -m network"""

import pandas as pd
import pytest

from fraud_benchmark.data.config import Config
from fraud_benchmark.data.delay import DelayParams
from fraud_benchmark.data.pipeline import prepare


@pytest.mark.network
def test_paysim_end_to_end(tmp_path):
    config = Config(
        raw_dir=tmp_path / 'raw',
        processed_dir=tmp_path / 'processed',
        split_ratios=(0.8, 0.1, 0.1),
        datasets={'paysim': {'start_date': '2023-01-01'}},
        delay=DelayParams(median_days=7.0, sigma=1.0, seed=0),
        campaign_gap=pd.Timedelta(days=1),
    )
    out = prepare('paysim', config)
    df = pd.read_parquet(out / 'data.parquet')

    assert len(df) > 6_000_000
    assert df['is_fraud'].sum() > 0
    assert set(df['split']) == {'train', 'val', 'test'}
    # Splits must be contiguous in time.
    assert df[df['split'] == 'train']['event_time'].max() <= (
        df[df['split'] == 'val']['event_time'].min()
    )
