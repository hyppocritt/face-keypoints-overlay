def test_overrides_parsing():
    from src.utils.settings import Settings

    overrides = ["model.lr=0.01", "training.use_amp=false"]

    parsed = Settings._parse_overrides(overrides)

    assert parsed["model"]["lr"] == 0.01
    assert parsed["training"]["use_amp"] is False


def test_deep_update():
    from src.utils.settings import Settings

    base = {"model": {"lr": 0.001}}
    new = {"model": {"lr": 0.01}}

    result = Settings._deep_update(base, new)

    assert result["model"]["lr"] == 0.01
