from etl_sar.lattice.runtime import checkpoint_boundaries


def test_checkpoint_boundaries_resume_without_reset_or_overshoot() -> None:
    assert checkpoint_boundaries(0, 1_000_000, 250_000) == (
        250_000,
        500_000,
        750_000,
        1_000_000,
    )
    assert checkpoint_boundaries(500_000, 1_100_000, 250_000) == (
        750_000,
        1_000_000,
        1_100_000,
    )
