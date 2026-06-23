from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest


CODING_ROOT = Path(__file__).resolve().parents[1]
VOLUME_ROOT = CODING_ROOT / "volume"

for path in (CODING_ROOT, VOLUME_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    """Repo-local scratch dir for tests; avoids Windows tmp_path permissions."""
    name = uuid.uuid4().hex[:12]
    path = CODING_ROOT / "artifacts" / "tests" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(autouse=True)
def _clear_trade_qc_policy_cache():
    """Per-date memos (QC policy, parquet layout) must not leak across tests
    that reuse the same dates with different roots or monkeypatched files."""
    from analysis.data import taq_loader

    taq_loader.clear_trade_qc_policy_cache()
    taq_loader.clear_layout_cache()
    yield
    taq_loader.clear_trade_qc_policy_cache()
    taq_loader.clear_layout_cache()
