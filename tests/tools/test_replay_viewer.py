from pathlib import Path


def test_orbit_wars_replay_viewer_uses_kaggle_renderer_style() -> None:
    viewer = Path(__file__).parents[2] / "tools" / "orbit_wars_replay_viewer.html"
    html = viewer.read_text(encoding="utf-8")

    assert 'const PLAYER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#F0E442"]' in html
    assert "rgba(255, 200, 50, 0.6)" in html
    assert "drawCometTrails" in html
    assert "drawFleets" in html
    assert "Fleet #" in html
    assert "Production" in html
