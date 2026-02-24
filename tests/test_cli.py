from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image

from illustrate import cli


def _write_minimal_pdb(path: Path) -> None:
    line = (
        f"{'ATOM':<6}{1:5d} {'CA':<4}{' ':1}{'GLY':>3} {'A':1}{1:4d}    "
        f"{0.000:8.3f}{0.000:8.3f}{-5.000:8.3f}"
    )
    path.write_text(line + "\nEND\n", encoding="utf-8")


def _write_render_command(pdb_name: str, output_name: str, width: int = 40, height: int = 40) -> str:
    return "\n".join(
        [
            "read",
            pdb_name,
            "ATOM  ---------- 0,9999, 1.0,1.0,1.0, 1.5",
            "END",
            "center",
            "aut",
            "scale",
            "10.0",
            "wor",
            "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
            "0.0,0.0023,2.0,1.0,0.2",
            f"{width},{height}",
            "calculate",
            output_name,
        ],
    )


def test_main_errors_without_stdin(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    code = cli.main([])

    out = capsys.readouterr()
    assert code == 2
    assert "no command stream on stdin" in out.err


def test_main_strict_mode_rejects_unknown_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("foo\n"))

    code = cli.main(["--strict-input"])

    out = capsys.readouterr()
    assert code == 2
    assert "parse error" in out.err.lower()
    assert "unknown command card" in out.err.lower()


def test_main_normalizes_netpbm_output(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_minimal_pdb(tmp_path / "mini.pdb")

    stream = _write_render_command("mini.pdb", "out.pnm")
    monkeypatch.setattr(sys, "stdin", io.StringIO(stream))

    code = cli.main([])

    assert code == 0
    assert (tmp_path / "out.png").is_file()
    assert (tmp_path / "opacity.png").is_file()

    image = Image.open(tmp_path / "out.png")
    assert image.size == (40, 40)


def test_main_writes_requested_non_square_image_size(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_minimal_pdb(tmp_path / "mini.pdb")

    stream = _write_render_command("mini.pdb", "rect.png", width=30, height=20)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stream))

    code = cli.main([])

    assert code == 0
    image = Image.open(tmp_path / "rect.png")
    assert image.size == (30, 20)


def test_main_writes_opacity_next_to_output_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_minimal_pdb(tmp_path / "mini.pdb")
    (tmp_path / "out").mkdir()

    stream = _write_render_command("mini.pdb", "out/render.png")
    monkeypatch.setattr(sys, "stdin", io.StringIO(stream))

    code = cli.main([])

    assert code == 0
    assert (tmp_path / "out" / "render.png").is_file()
    assert (tmp_path / "out" / "opacity.png").is_file()
    assert not (tmp_path / "opacity.png").exists()
