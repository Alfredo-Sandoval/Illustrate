from __future__ import annotations

from illustrate.presets import default_rules, preset_library, render_params_from_preset


def test_preset_library_returns_independent_rule_instances() -> None:
    presets = preset_library()
    before = presets[1].rules[0].radius

    presets[0].rules[0].radius = 9.9

    assert presets[1].rules[0].radius == before


def test_render_params_from_preset_returns_independent_nested_objects() -> None:
    a = render_params_from_preset("Default", "one.pdb")
    b = render_params_from_preset("Default", "two.pdb")

    a.transform.rotations[0] = ("z", 123.0)
    a.rules[0].radius = 7.7
    a.world.background = (0.0, 0.0, 0.0)

    assert b.transform.rotations[0] != ("z", 123.0)
    assert b.rules[0].radius != 7.7
    assert b.world.background != (0.0, 0.0, 0.0)


def test_default_rules_match_original_fortran_stack() -> None:
    rules = default_rules()

    assert len(rules) == 15

    assert rules[0].record_name == "HETATM"
    assert rules[0].descriptor == "-----HOH--"
    assert rules[0].radius == 0.0

    assert rules[11].record_name == "ATOM  "
    assert rules[11].descriptor == "----------"
    assert rules[11].color == (1.0, 0.7, 0.5)
    assert rules[11].radius == 1.5

    assert rules[-1].record_name == "HETATM"
    assert rules[-1].descriptor == "-----HEM--"
