"""spinw_studio tests: the data contract (pure) and, below, the studio rendered offscreen (GPU).

Everything runs on a hand-made SystemView, so spinwaves is not needed.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from spinw_studio import adapter
from spinw_studio.model import (
    IDENTITY,
    AtomSite,
    SymOp,
    SystemView,
    apply_op,
    cell_edges,
    default_radius,
    expand,
    lattice_matrix,
    lattice_params,
    moment_xyz,
    orientation,
    parse_color,
    to_xyz,
    validate_options,
    with_lattice,
)

A_LFO = np.diag([5.3, 5.6, 7.5])
FE_UVW = [(0, 0.5, 0), (0.5, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)]  # Pbnm 4b, like LFO
FE_M = [(-1, 0, 0.1), (1, 0, -0.1), (1, 0, 0.1), (-1, 0, -0.1)]  # G-type, canted


def op(R, t, theta=1, label=""):
    """A SymOp from its crystal-frame rotation R, translation t and time reversal theta (moments: θ·det(R)·R)."""
    R = np.asarray(R, float)
    return SymOp(R, t, theta * np.linalg.det(R) * R, label)


# The generators of lfo_spinw.py's Pbnm (the first one with time reversal) and the products that place the 4 Fe of 4b.
G1 = op(np.diag([1, -1, -1]), (0.5, 0.5, 0), -1, "x+1/2,-y+1/2,-z, -1")
G3 = op(np.diag([-1, -1, 1]), (0, 0, 0.5), 1, "-x,-y,z+1/2, +1")
G13 = op(np.diag([-1, 1, -1]), (0.5, 0.5, -0.5), -1, "-x+1/2,y+1/2,-z-1/2, -1")  # G1 after G3
INVERSION = op(-np.eye(3), (0, 0, 0), 1, "-x,-y,-z, +1")
PBNM_OPS = (IDENTITY, G1, G3, G13, INVERSION)


def pbnm_fe() -> SystemView:
    """The 4 Fe of LFO made from one site (0, 1/2, 0) with moment (-1, 0, 0.1) by the operations above."""
    site_uvw, site_m = np.array([0.0, 0.5, 0.0]), np.array([-1.0, 0.0, 0.1])
    atoms = tuple(AtomSite("Fe", i, tuple(apply_op(PBNM_OPS[g], site_uvw)[0]),
                           tuple(moment_xyz(A_LFO, PBNM_OPS[g], site_m)), 2.5, op=g, site=0)
                  for i, g in enumerate((0, 1, 2, 3)))
    return SystemView(A_LFO, atoms, "LFO Fe", PBNM_OPS)


def test_symmetry_operations_place_atoms_and_turn_moments_as_axial_vectors():
    fe = pbnm_fe()
    assert np.allclose([a.uvw for a in fe.atoms], FE_UVW)  # 4b, in spinwaves' order
    m = np.array([a.moment for a in fe.atoms])
    assert np.allclose(m[:, 0], [-1, 1, 1, -1]) and np.allclose(m[:, 2], 0.1)  # G-type along x, weak FM along z
    assert np.allclose(apply_op(INVERSION, [[0, 0.5, 0]]), [[0, 0.5, 0]])  # 4b sits on an inversion centre
    assert np.allclose(moment_xyz(A_LFO, INVERSION, [1, 2, 3]), [1, 2, 3])  # axial: inversion keeps a moment
    assert np.allclose(moment_xyz(A_LFO, op(np.eye(3), (0, 0, 0), -1), [1, 2, 3]), [-1, -2, -3])  # time reversal
    assert np.linalg.det(orientation(A_LFO, INVERSION)) == pytest.approx(-1)  # meshes are mirrored by it
    A = lattice_matrix(5, 6, 7, 90, 90, 120)  # hexagonal-like: the Cartesian matrix depends on A
    turn = orientation(A, op([[0, -1, 0], [1, -1, 0], [0, 0, 1]], (0, 0, 0)))  # 3-fold about c
    assert np.allclose(np.linalg.matrix_power(turn, 3), np.eye(3)) and not np.allclose(turn, np.round(turn))
    with pytest.raises(ValueError, match="det"):
        SymOp(np.diag([2, 1, 1]), (0, 0, 0), np.eye(3))


def test_expand_carries_each_atoms_orientation():
    out = expand(pbnm_fe(), [[0, 0.99]] * 3)["Fe"]  # one copy of each
    assert np.allclose(out.orientations, [orientation(A_LFO, PBNM_OPS[g]) for g in (0, 1, 2, 3)])
    assert np.allclose(expand(lfo_like(), [[0, 0.99]] * 3)["Fe"].orientations, np.eye(3))  # no ops: identity
    with pytest.raises(ValueError, match="symmetry operation outside ops"):
        SystemView(A_LFO, (AtomSite("Fe", 0, (0, 0, 0), (0, 0, 0), op=3),))


def lfo_like(**kw) -> SystemView:
    atoms = tuple(AtomSite("Fe", i, u, m, spin=2.5) for i, (u, m) in enumerate(zip(FE_UVW, FE_M, strict=True)))
    return SystemView(kw.get("A", A_LFO), kw.get("atoms", atoms), "LFO-like")


def test_uvw_to_xyz_uses_the_columns_of_A():
    A = np.array([[5.0, 1.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 3.0]])  # b leans toward +x
    assert np.allclose(to_xyz(A, [[0, 1, 0]]), [[1.0, 4.0, 0.0]])
    assert np.allclose(to_xyz(A, [[0.5, 0.5, 1]]), [[3.0, 2.0, 3.0]])


def test_expand_finds_boundary_atoms_and_their_translations():
    out = expand(lfo_like(), [[-0.1, 1.1]] * 3)["Fe"]
    assert len(out.xyz) == 4 + 4 + 2 + 2  # coordinates 0 also appear at 1
    assert ((out.uvw >= -0.1) & (out.uvw <= 1.1)).all()
    assert np.allclose(out.xyz, to_xyz(A_LFO, out.uvw))
    i = np.flatnonzero((out.indices == 0) & (out.translations == [1, 0, 1]).all(1))
    assert len(i) == 1 and np.allclose(out.uvw[i[0]], [1, 0.5, 1]) and np.allclose(out.moments[i[0]], FE_M[0])
    assert len(expand(lfo_like(), [[0, 2], [0, 1], [0, 1]])["Fe"].xyz) == 6 + 8 + 3 + 4  # a 2x1x1 box


def test_cell_edges_are_the_12_edges_of_the_cell():
    edges = cell_edges(A_LFO)
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    assert edges.shape == (12, 2, 3) and sorted(lengths) == pytest.approx([5.3] * 4 + [5.6] * 4 + [7.5] * 4)


def test_plot_options_are_validated_and_merged_with_defaults():
    opts = validate_options({"boundaries": ([-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.02]), "atom_colors": {"Fe": "Orange"}})
    assert opts["boundaries"][2] == [-0.1, 1.02] and opts["show_moments"] is True and opts["cell_radius"] == 0.03
    with pytest.raises(KeyError, match="did you mean 'boundaries'"):
        validate_options({"boundary": [[0, 1]] * 3})
    for bad in ({"boundaries": [[1, 0], [0, 1], [0, 1]]}, {"atom_colors": {"Fe": "Orangey"}},
                {"atom_radii": {"Fe": -1}}, {"show_cell": "yes"}):
        with pytest.raises(ValueError):
            validate_options(bad)


def test_colours_and_radii():
    assert parse_color("Orange") == pytest.approx((1.0, 165 / 255, 0.0))
    assert parse_color("#ff0000") == (1.0, 0.0, 0.0) and parse_color((255, 0, 0)) == (1.0, 0.0, 0.0)
    assert default_radius("Fe3+") == default_radius("Fe") == 0.6 and default_radius("Xx") == 0.5


def test_system_view_rejects_a_bad_lattice_matrix():
    with pytest.raises(ValueError, match="invertible 3x3"):
        lfo_like(A=np.zeros((3, 3)))


def fake_sw(A=A_LFO):
    """A duck-typed stand-in for a SpinW object: LFO's Fe site, the Pbnm operations that copy it, and spinwaves' own
    cell positions (what the adapter's position check reads)."""
    atoms_all = [SimpleNamespace(label="Fe", r=u) for u in FE_UVW]
    return SimpleNamespace(crystal=SimpleNamespace(A=np.array(A, float), atoms_all=atoms_all),
                           sites=[("Fe", (0, 0.5, 0), (-1, 0, 0.1), 2.5)], ops=list(PBNM_OPS[:4]),
                           generators=[[0, 0], [0, 1], [0, 2], [0, 3]])


@pytest.fixture
def filled(monkeypatch):
    """The adapter's required stubs filled for ``fake_sw`` objects, as you fill them for spinwaves."""
    stubs = {
        "site_positions_uvw": lambda sw: [s[1] for s in sw.sites],
        "site_moments_xyz": lambda sw: [s[2] for s in sw.sites],
        "site_labels": lambda sw: [s[0] for s in sw.sites],
        "site_spins": lambda sw: [s[3] for s in sw.sites],
        "symmetry_operations": lambda sw: sw.ops,
        "atom_operation": lambda sw, op: np.block([[op.R, op.t[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]),
        "moment_operation": lambda sw, op: op.M,
        "operation_label": lambda sw, op: op.label,
        "atom_generators": lambda sw: np.array(sw.generators),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(adapter, name, stub)
    return monkeypatch


def test_unfilled_stubs_say_what_to_fill():
    with pytest.raises(NotImplementedError, match="fill in from spinwaves: site_labels"):
        adapter.system_view(fake_sw())


def test_the_adapter_builds_the_cell_from_sites_and_operations(filled):
    view = adapter.system_view(fake_sw())
    assert view.labels == ["Fe"] and len(view.ops) == 4 and np.allclose(view.A, A_LFO)
    assert np.allclose([a.uvw for a in view.atoms], FE_UVW)  # spinwaves' order
    assert [(a.site, a.op) for a in view.atoms] == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert np.allclose([a.moment[0] for a in view.atoms], [-1, 1, 1, -1]) and view.atoms[3].spin == 2.5
    assert view.ops[1].label == "x+1/2,-y+1/2,-z, -1"


@pytest.mark.parametrize(("stub", "value", "message"), [
    ("atom_operation", lambda sw, op: np.block([[op.R, op.t[:, None] + 0.25], [np.zeros((1, 3)), np.ones((1, 1))]]),
     "check adapter.atom_operation and adapter.atom_generators"),  # caught by spinwaves' own positions
    ("atom_operation", lambda sw, op: np.eye(4) * 2, "must return \\[\\[R, t\\], \\[0, 0, 0, 1\\]\\]"),
    ("moment_operation", lambda sw, op: np.eye(4), "moment_operation must return"),
    ("atom_generators", lambda sw: np.array([[0, 0], [0, 7]]), "operation index outside"),
    ("atom_generators", lambda sw: np.array([[0, 0], [0, 0]]), "land on the same position"),
])
def test_the_adapter_names_the_stub_that_breaks_the_contract(filled, stub, value, message):
    filled.setattr(adapter, stub, value)
    with pytest.raises(ValueError, match=message):
        adapter.system_view(fake_sw())


def test_lattice_parameters_and_matrix_agree():
    params = (5.0, 6.0, 7.0, 80.0, 95.0, 110.0)
    A = lattice_matrix(*params)
    assert lattice_params(A) == pytest.approx(params) and np.allclose(A[1:, 0], 0) and A[2, 1] == 0
    assert lattice_params(A_LFO) == pytest.approx((5.3, 5.6, 7.5, 90, 90, 90))
    with pytest.raises(ValueError, match="no unit cell"):
        lattice_matrix(5, 5, 5, 10, 10, 170)
    turned = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]) @ A_LFO  # a along +y
    view = with_lattice(lfo_like(A=turned), 6.0, 5.6, 7.5, 90, 90, 90)
    assert np.allclose(view.A[:, 0], [0, 6.0, 0]) and view.atoms == lfo_like().atoms  # orientation and uvw kept
    assert np.allclose(with_lattice(lfo_like(A=turned), *lattice_params(turned)).A, turned)


# -- the studio, offscreen (GPU) ------------------------------------------------------------------


@pytest.fixture
def studio(viewer):
    from spinw_studio import show

    editor = show(viewer, lfo_like(), {"atom_colors": {"Fe": "Orange"}})
    return viewer, editor.scene, editor


def test_scene_draws_atoms_moments_and_cell(studio):
    viewer, scene, editor = studio
    assert [n.name for n in scene.walk()] == ["Structure", "Atoms", "Fe", "Moments", "Fe moments", "Cell"]
    assert all(n.locked for n in scene.walk()) and editor.selected is scene.find("Structure")
    assert scene.find("Fe").instances.count == 12 and scene.find("Fe moments").instances.count == 12
    assert scene.find("Cell").instances.count == 12 and viewer.snapshot().std() > 5
    m = scene.find("Fe moments").instances.matrices[0]
    assert np.linalg.norm(m[:3, 2]) == pytest.approx(2.0)  # arrow length (moment_length, Å) along its z axis
    assert scene.find("Structure").params["b"] == pytest.approx(5.6)


def test_non_magnetic_labels_get_no_moments_object(viewer):
    from spinw_studio import show

    o = AtomSite("O", 4, (0.25, 0.25, 0.25), (0, 0, 0))
    editor = show(viewer, lfo_like(atoms=(*lfo_like().atoms, o)))
    assert [n.name for n in editor.scene.find("Moments").children] == ["Fe moments"]
    assert editor.scene.find("O").instances.count == 1


def structure_param(key, value):
    return lambda ed, s: ed.set_procedural_param(s.find("Structure"), key, value)


SPINW_ACTIONS = {  # name → (action(editor, scene), changes the image)
    "lattice a": (structure_param("a", 6.5), True),
    "lattice gamma": (structure_param("gamma", 105.0), True),
    "extent": (structure_param("umax", 2.1), True),
    "moment length": (structure_param("moment_length", 3.0), True),
    "radius": (lambda ed, s: ed.set_shape_param(s.find("Fe"), "r", 1.0), True),
    "atom shape": (lambda ed, s: ed.set_shape(s.find("Fe"), "valo.meshes.primitives:cube"), True),
    "material parameter": (lambda ed, s: ed.set_param(s.materials["Fe"], "base_color", (0.1, 0.2, 0.9)), True),
    "hide moments": (lambda ed, s: ed.set_visible(s.find("Moments"), False), True),
    "cell radius": (lambda ed, s: ed.set_shape_param(s.find("Cell"), "r", 0.15), True),
    "hide cell": (lambda ed, s: ed.set_visible(s.find("Cell"), False), True),
    "add object": (lambda ed, s: ed.add_shape("valo.meshes.primitives:torus"), True),
}


@pytest.mark.parametrize("name", list(SPINW_ACTIONS))
def test_studio_actions_undo_exactly(studio, name):
    viewer, scene, editor = studio
    action, visible = SPINW_ACTIONS[name]
    def state():  # plot_options, plus the lattice (it lives in sw, not in plot_options)
        return editor.plot_options(), dict(scene.find("Structure").params)

    before, image = state(), viewer.snapshot()
    action(editor, scene)
    assert len(editor.undo_stack.done) == 1 and state() != before
    assert (np.abs(viewer.snapshot().astype(int) - image).max() > 10) == visible
    editor.undo_stack.undo()
    assert state() == before and np.array_equal(viewer.snapshot(), image)


def test_a_lattice_change_moves_everything_and_keeps_your_edits(studio):
    _, scene, editor = studio
    fe, structure = scene.find("Fe"), scene.find("Structure")
    editor.set_shape_param(fe, "r", 0.8)
    editor.set_visible(scene.find("Moments"), False)
    editor.set_procedural_param(structure, "c", 9.0)
    assert scene.find("Fe") is fe and fe.shape.params["r"] == 0.8 and not scene.find("Moments").visible
    assert editor.system.A[2, 2] == pytest.approx(9.0) and fe.instances.matrices[:, 2, 3].max() == pytest.approx(9.0)
    assert sorted(np.linalg.norm(scene.find("Cell").instances.matrices[:, :3, 2], axis=1))[-1] == pytest.approx(9.0)
    editor.set_procedural_param(structure, "beta", 100.0)
    axes = {label: d for label, d, *_ in editor.compass_axes()}
    assert np.dot(axes["a"], axes["c"]) == pytest.approx(np.cos(np.radians(100.0)))  # the compass follows
    with pytest.raises(ValueError, match="not deleted"):
        editor.delete(fe)
    editor.set_procedural_param(structure, "umin", 1.5)  # min > max: refused, nothing changes
    assert structure.params["umin"] == -0.1 and "min < max" in editor.errors["Structure"]


def test_lattice_edits_go_through_the_adapter(viewer, monkeypatch, filled):
    from spinw_studio import adapter, show
    from spinw_studio.model import lattice_matrix

    sw = fake_sw()
    editor = show(viewer, sw)
    structure = editor.scene.find("Structure")

    def unfilled(sw, *p):
        raise NotImplementedError("fill in from spinwaves: set_lattice(...)")

    monkeypatch.setattr(adapter, "set_lattice", unfilled)
    editor.set_procedural_param(structure, "a", 6.0)  # the stub isn't filled: an error, and nothing changes
    assert "set_lattice" in editor.errors["Structure"] and structure.params["a"] == pytest.approx(5.3)
    assert np.allclose(sw.crystal.A, A_LFO)
    monkeypatch.setattr(adapter, "set_lattice", lambda sw, *p: setattr(sw.crystal, "a", p[0]))  # A unchanged
    editor.set_procedural_param(structure, "a", 6.0)
    assert "still gives the old cell" in editor.errors["Structure"] and structure.params["a"] == pytest.approx(5.3)
    monkeypatch.setattr(adapter, "set_lattice", lambda sw, *p: setattr(sw.crystal, "A", lattice_matrix(*p)))
    editor.set_procedural_param(structure, "a", 6.0)
    assert sw.crystal.A[0, 0] == pytest.approx(6.0) and editor.system.A[0, 0] == pytest.approx(6.0)
    editor.undo_stack.undo()
    assert sw.crystal.A[0, 0] == pytest.approx(5.3)


def test_info_rows_and_compass_axes(studio):
    _, scene, editor = studio
    fe = scene.find("Fe")
    k = int(np.flatnonzero(np.all(np.isclose(fe.instances.matrices[:, :3, 3], [2.65, 0, 0]), axis=1))[0])
    rows = fe.info(k)
    assert rows["Object"] == "Fe" and rows["Cell index"] == "1" and rows["uvw"] == "(0.500, 0.000, 0.000)"
    assert rows["xyz (Å)"] == "(2.650, 0.000, 0.000)"
    assert rows["m (xyz)"] == "(1.000, 0.000, -0.100)" and rows["S"] == "2.5"
    assert scene.find("Fe moments").info(k)["Cell index"] == "1"
    axes = {label: d for label, d, *_ in editor.compass_axes()}
    assert set(axes) == {"x", "y", "z", "a", "b", "c"} and axes["b"] == pytest.approx((0.0, 1.0, 0.0))


def test_plot_options_round_trip_and_export(studio, tmp_path):
    import runpy

    import spinw_studio

    import valo

    viewer, scene, editor = studio
    editor.set_procedural_param(scene.find("Structure"), "wmax", 1.6)
    editor.set_shape_param(scene.find("Fe"), "r", 0.45)
    editor.set_param(scene.materials["Fe"], "roughness", 0.8)
    editor.set_object_material(scene.find("Fe"), editor.duplicate_material("Fe").name)
    gold = editor.new_material("Gold", "pbr", metallic=1.0).name
    editor.add_shape("valo.meshes.primitives:torus", material=gold)
    editor.set_visible(scene.find("Cell"), False)
    editor.set_shape(scene.find("Fe moments"), "valo.meshes.primitives:cone")
    editor.render_options.update(size=[240, 160], samples=1)
    opts = editor.plot_options()
    assert opts["boundaries"][2] == [-0.1, 1.6] and opts["atom_radii"] == {"Fe": 0.45} and not opts["show_cell"]
    assert [o["name"] for o in opts["objects"]] == ["Torus"] and "Gold" in opts["object_materials"]
    assert opts["shapes"] == {"Fe moments": {"func": "valo.meshes.primitives:cone",
                                             "params": opts["shapes"]["Fe moments"]["params"]}}
    assert opts["camera"]["azimuth"] == viewer.camera.azimuth and opts["image"]["size"] == [240, 160]
    shown = valo.render(scene, editor.render_settings())  # also: Render works for a scene that can't be saved
    assert np.array_equal(spinw_studio.render(lfo_like(), opts), shown)  # the options rebuild the same picture
    namespace = runpy.run_path(str(editor.export_plot_options(tmp_path / "opts.py")))
    assert namespace["PLOT_OPTIONS"] == opts and not editor.can_export()


def test_the_file_menu_opens_without_errors(studio):

    from valo.testing import open_every_menu

    viewer, _, editor = studio
    assert open_every_menu(viewer) >= 3 and viewer.gui_error is None  # File holds Export plot options


def test_plot_structure_blocks_then_returns_the_final_options(viewer, monkeypatch):
    from spinw_studio import api

    calls = []
    monkeypatch.setattr(api, "_window", lambda title: calls.append(title) or viewer)
    monkeypatch.setattr(viewer, "run",  # "the user" edits, then closes
                        lambda: viewer.editor.set_shape_param(viewer.scene.find("Fe"), "r", 0.9))
    opts = api.plot_structure(lfo_like(), {"atom_colors": {"Fe": "Orange"}})
    assert calls == ["SpinW studio: LFO-like"] and opts["atom_radii"] == {"Fe": 0.9}
    monkeypatch.undo()
