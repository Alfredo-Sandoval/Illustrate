from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from illustrate import cli

DATA = Path(__file__).resolve().parents[1] / "examples"
PDB = str(DATA / "2hhb.pdb")


def _run(*args: str) -> int:
    return cli.main([*args, "--backend", "numpy"])


def test_structure_file_renders_a_png_named_after_it(tmp_path: Path) -> None:
    assert _run(PDB, "--outdir", str(tmp_path / "figures")) == 0

    with Image.open(tmp_path / "figures" / "2hhb.png") as image:
        assert image.mode == "RGB"
        assert min(image.size) > 100


def test_width_scales_the_structure_to_the_requested_pixels(tmp_path: Path) -> None:
    target = tmp_path / "wide.png"

    assert _run(PDB, "-o", str(target), "--width", "600", "--transparent") == 0

    with Image.open(target) as image:
        assert 590 <= image.width <= 600
        assert image.mode == "RGBA"
        assert image.getchannel("A").getpixel((0, 0)) == 0


def test_saved_settings_reproduce_the_image(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    again = tmp_path / "again.png"
    assert _run(
        PDB, "-o", str(first), "--preset", "cool blues", "--rotate", "x=30,y=-45",
        "--height", "300", "--save-settings",
    ) == 0

    assert _run("--settings", str(first.with_suffix(".json")), "-o", str(again)) == 0

    with Image.open(first) as a, Image.open(again) as b:
        assert a.tobytes() == b.tobytes()


def test_batch_renders_the_rest_when_one_structure_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(PDB, "missing.pdb", "--outdir", str(tmp_path), "--format", "svg") == 1

    assert (tmp_path / "2hhb.svg").read_text(encoding="utf-8").startswith("<?xml")
    assert "missing.pdb: no such file" in capsys.readouterr().err


def test_command_file_renders_the_original_input_format(tmp_path: Path) -> None:
    target = tmp_path / "legacy.png"

    assert _run("--command-file", str(DATA / "2hhb.inp"), "-o", str(target)) == 0

    # The file's WOR card sets a white background and autosizing.
    with Image.open(target) as image:
        assert image.mode == "RGB"
        assert image.width % 2 == 0 and image.height % 2 == 0
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.getpixel((image.width // 2, image.height // 2)) != (255, 255, 255)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "name at least one PDB file or PDB ID"),
        ([PDB, "--scale", "5", "--width", "500"], "either --scale or --width/--height"),
        ([PDB, "--preset", "nope"], "unknown preset 'nope'"),
        ([PDB, "--preset", "Default", "--settings", "s.json"], "either --preset or --settings"),
        ([PDB, "--rotate", "q=3"], "cannot read rotation 'q=3'"),
        ([PDB, "-o", "figure.jpg"], "output must be a .png or .svg file"),
        ([PDB, PDB], "two structures would write the same image"),
        ([PDB, "--command-file", "x.inp"], "--command-file takes its structure and look from the file"),
        (["zzzz9"], "no such file, and not a four-character PDB ID"),
    ],
)
def test_invalid_requests_exit_2_with_a_reason(
    args: list[str], message: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    assert _run(*args) == 2

    assert message in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
