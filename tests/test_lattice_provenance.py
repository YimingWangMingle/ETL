from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "third_party" / "lattice"
EXPECTED = {
    "distributions.py": "D6CC339D73765D588E00E4A07E23265F1333163E866A87BC7A3644061A2A230D",
    "ppo_policies.py": "46C826E616BB1614D18C2748F832D194A5251692EBCFAB54BBB94EF66D2C6416",
    "sac_policies.py": "D046C8AAB013B32DFC1F431F3C49BEF9B0372751464410AC99F10D813A2FA59B",
    "main_reorient.py": "A9D4A992B360828E5ED8A68A37B1961581717C0292919670E96090637CFE4CF1",
    "main_walker.py": "B043C22B1E6B882CF97F4B48D08A5F4DD6E01F7F63BD383AE72F5B69D6990986",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_vendored_lattice_files_match_pinned_upstream() -> None:
    for filename, expected_hash in EXPECTED.items():
        assert _sha256(VENDORED / filename) == expected_hash


def test_upstream_document_records_all_vendored_hashes() -> None:
    text = (VENDORED / "UPSTREAM.md").read_text(encoding="utf-8")
    assert "846d02fa993b9b80ce5ecb806463e0a05711bad3" in text
    for filename, expected_hash in EXPECTED.items():
        assert filename in text
        assert expected_hash in text


def test_upstream_license_is_preserved() -> None:
    text = (VENDORED / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Copyright (c) 2023 Mathis Group" in text
