from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from illustrate import RenderParams, Transform, params_from_json, params_to_json
from illustrate.gui.render_params import render_params_from_panels, transform_panel_values
from illustrate.rendering.api import rotation_matrix


def test_single_axis_and_empty_rotations_keep_their_meaning() -> None:
    values = transform_panel_values(Transform(rotations=[("x", 30.0)]))
    np.testing.assert_allclose(
        (values["xrot"], values["yrot"], values["zrot"]), (30.0, 0.0, 0.0), atol=1e-6, rtol=0,
    )
    values = transform_panel_values(Transform(rotations=[]))
    assert (values["xrot"], values["yrot"], values["zrot"]) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "transform, message",
    [
        (Transform(autocenter="none"), "autocenter='auto'"),
        (Transform(autocenter="center"), "autocenter='auto'"),
    ],
)
def test_unrepresentable_gui_transforms_raise(transform: Transform, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        transform_panel_values(transform)


@settings(derandomize=True, max_examples=100)
@given(
    rotations=st.lists(st.tuples(st.sampled_from(["z", "y", "x"]), st.integers(-720, 720)), max_size=12),
)
def test_settings_gui_round_trip_preserves_rotation_matrix(
    rotations: list[tuple[str, int]],
) -> None:
    rotation_values = [(axis, float(angle)) for axis, angle in rotations]
    params = RenderParams(pdb_path="mini.pdb", rules=[], transform=Transform(rotations=rotation_values))
    loaded = params_from_json(params_to_json(params))
    rebuilt = render_params_from_panels(
        loaded.pdb_path, loaded.rules, transform_panel_values(loaded.transform), {}, {},
        render_width=64, render_height=64,
    )
    np.testing.assert_allclose(
        rotation_matrix(rebuilt.transform.rotations), rotation_matrix(rotation_values), atol=1e-6, rtol=0,
    )


@pytest.mark.parametrize("y_angle", [-90.0, 90.0])
def test_settings_rotation_at_gimbal_lock_preserves_orientation(y_angle: float) -> None:
    transform = Transform(rotations=[("z", 45.0), ("y", y_angle), ("x", 33.0)])
    rebuilt = render_params_from_panels("mini.pdb", [], transform_panel_values(transform), {}, {}, render_width=64, render_height=64)
    np.testing.assert_allclose(
        rotation_matrix(rebuilt.transform.rotations), rotation_matrix(transform.rotations), atol=1e-6, rtol=0,
    )


@settings(derandomize=True, max_examples=100)
@given(
    rotations=st.lists(st.tuples(st.sampled_from(["x", "y", "z", "X", "q", ""]), st.integers(-720, 720)), max_size=8),
    centering=st.sampled_from(["auto", "aut", "center", "none", "invalid"]),
)
def test_gui_transform_projection_preserves_rotation_or_rejects(
    rotations: list[tuple[str, int]], centering: str,
) -> None:
    transform = Transform(rotations=[(axis, float(angle)) for axis, angle in rotations], autocenter=centering)
    try:
        values = transform_panel_values(transform)
    except ValueError as exc:
        assert str(exc).startswith(("GUI settings require", "unsupported rotation axis"))
        return
    rebuilt = render_params_from_panels("mini.pdb", [], values, {}, {}, render_width=64, render_height=64)
    np.testing.assert_allclose(
        rotation_matrix(rebuilt.transform.rotations), rotation_matrix(transform.rotations), atol=1e-6, rtol=0,
    )
