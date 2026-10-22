"""valo.gpu.renderer: what gets drawn: instanced meshes, viewport rects, shading in each light mode, one renderer."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu
VIEWER_SIZE = (640, 400)


def test_empty_viewer_draws_the_background(viewer):
    viewer.show(SimpleNamespace(batches=list))
    img = viewer.snapshot()
    assert (img == img[0, 0]).all()


def test_any_mesh_draws_instanced_with_its_own_transforms(viewer):
    from valo import instance_matrices, meshes

    m = viewer.renderer.material("Matte", "lambert", base_color=(0.8, 0.8, 0.8))
    cubes = viewer.renderer.instances(meshes.cube(), m)
    arrows = viewer.renderer.instances(meshes.arrow(), m)
    assert viewer.renderer.instances(meshes.cube(), m).mesh is not cubes.mesh  # cache is per MeshData object
    viewer.show(SimpleNamespace(batches=lambda: [cubes.batch(), arrows.batch()]))
    viewer.camera.fit(3.0)
    empty = viewer.snapshot()

    cubes.set(instance_matrices([[-1, 0, 0], [1, 0, 0]], [[0.5], [1.0]]))
    with_cubes = viewer.snapshot()
    assert (with_cubes != empty).any(-1).sum() > 500
    arrows.set(instance_matrices([[0, 0, 0]], (0.3, 0.3, 2.0), z_axis=(0, 1, 1)))
    assert (viewer.snapshot() != with_cubes).any(-1).sum() > 200


def test_scene_renders_into_an_offset_viewport_rect(viewer):
    import wgpu

    from valo import Camera, Lighting

    m = viewer.renderer.material("Matte", "lambert", base_color=(0.9, 0.9, 0.9), ambient=1.0)
    ball = viewer.renderer.spheres(m)
    ball.set([[0, 0, 0]], 1.6)  # seen from distance 3, it fills the whole viewport
    scene = SimpleNamespace(batches=lambda: [ball.batch()])
    fmt = viewer.context._config["format"]
    tex = viewer.device.create_texture(size=(400, 300, 1), format=fmt,
                                       usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    viewer.renderer.render(tex, (100, 50, 200, 150), scene, Camera(distance=3.0), Lighting())
    data = viewer.device.queue.read_texture({"texture": tex}, {"bytes_per_row": 400 * 4}, (400, 300, 1))
    img = np.frombuffer(data, np.uint8).reshape(300, 400, 4)[:, :, :3]
    inside = np.zeros((300, 400), bool)
    inside[50:200, 100:300] = True
    assert img[inside].min() > 150  # the bright sphere covers the rect
    assert img[~inside].max() < 60  # the window background everywhere else


def test_orthographic_shading_is_the_same_for_identical_spheres(viewer):
    """In orthographic mode the view vector is the same everywhere, so two identical spheres shade identically."""
    from valo import Camera

    m = viewer.renderer.material("Shiny", "pbr", base_color=(0.9, 0.9, 0.9), metallic=1.0, roughness=0.2)
    balls = viewer.renderer.spheres(m)
    balls.set([[-1.2, 0, 0], [1.2, 0, 0]], 0.5)
    viewer.show(SimpleNamespace(batches=lambda: [balls.batch()]))
    viewer.camera = cam = Camera(azimuth=90.0, elevation=0.0, distance=3.0)

    centres = np.array([[-1.2, 0, 0, 1], [1.2, 0, 0, 1]])

    def patches_differ(projection):  # mean difference of the two spheres' pixels, each centred on its projection
        cam.set_projection(projection)
        img = viewer.snapshot().astype(int)
        h, w = img.shape[:2]
        clip = centres @ cam.matrices(w / h)[0].T
        px = ((clip[:, 0] / clip[:, 3] + 1) / 2 * w).round().astype(int)
        py = ((1 - clip[:, 1] / clip[:, 3]) / 2 * h).round().astype(int)
        a, b = (img[y - 25 : y + 25, x - 25 : x + 25] for x, y in zip(px, py, strict=True))
        return np.abs(a - b).mean()

    persp, ortho = patches_differ("perspective"), patches_differ("orthographic")
    assert ortho < 0.5 and persp > 3 * ortho


def test_global_ambient_lights_the_unlit_side(viewer):
    from valo import Camera
    from valo.scene.lighting import Lighting

    m = viewer.renderer.material("Matte", "lambert", base_color=(0.8, 0.8, 0.8))
    ball = viewer.renderer.spheres(m)
    ball.set([[0, 0, 0]], 0.8)
    viewer.show(SimpleNamespace(batches=lambda: [ball.batch()]))
    viewer.camera = Camera(azimuth=0.0, elevation=0.0, distance=4.0)
    viewer.lighting = Lighting("headlight", azimuth=180.0, elevation=0.0)  # from behind: the visible side is unlit
    lit = viewer.snapshot()
    viewer.lighting.ambient = 0.0
    dark = viewer.snapshot()
    background = lit[0, 0]
    assert lit.max() > 60 and (dark <= background).all()  # only ambient lit the front: without it, black on bg
    viewer.lighting = Lighting()


@pytest.mark.parametrize(("mode", "changes"), [("headlight", False), ("fixed", True)])
def test_orbiting_changes_the_shading_only_with_a_fixed_light(viewer, mode, changes):
    from valo import Camera
    from valo.scene.lighting import Lighting

    m = viewer.renderer.material("Matte", "lambert", base_color=(0.8, 0.8, 0.8))
    ball = viewer.renderer.spheres(m)
    ball.set([[0, 0, 0]], 0.8)
    viewer.show(SimpleNamespace(batches=lambda: [ball.batch()]))
    viewer.camera, viewer.lighting = Camera(elevation=0.0, distance=4.0), Lighting(mode, -40.0, 20.0)
    first = viewer.snapshot()
    viewer.camera.azimuth += 180  # a sphere looks the same from behind, unless the light stayed put
    differs = np.abs(viewer.snapshot().astype(int) - first).max() > 30
    assert differs == changes
    viewer.lighting = Lighting()


def test_batches_carry_their_instances_and_matrices(viewer):
    from valo import instance_matrices, meshes

    m = viewer.renderer.material("M", "lambert")
    cubes = viewer.renderer.instances(meshes.cube(), m, label="cubes", info=lambda i: {"index": i})
    matrices = instance_matrices([[0, 0, 0], [2, 0, 0]])
    cubes.set(matrices)
    b = cubes.batch()
    assert b.source is cubes and b[:4] == (m, cubes.mesh, cubes.bind_group, 2)
    assert np.array_equal(cubes.matrices, matrices) and cubes.label == "cubes" and cubes.info(1) == {"index": 1}


def test_viewers_share_one_renderer_so_scenes_move_between_them(viewer):
    from rendercanvas.offscreen import RenderCanvas

    from valo import Viewer, renderer

    m = renderer().material("M", "lambert")
    balls = renderer().spheres(m)  # built without any window
    balls.set([[0, 0, 0]], 1.0)
    other = Viewer(RenderCanvas(size=(300, 200), pixel_ratio=1))
    assert other.renderer is viewer.renderer is renderer() and other.device is viewer.device
    other.show(SimpleNamespace(batches=lambda: [balls.batch()], bounding_radius=lambda: 1.0))
    assert other.snapshot().std() > 5


def test_background_colour_comes_from_rc(viewer):
    from valo import rc_context

    viewer.show(SimpleNamespace(batches=list))
    with rc_context({"render.background": (1.0, 0.0, 0.0)}):
        img = viewer.snapshot()
    assert (img[0, 0] == (255, 0, 0)).all()  # linear 1.0 → sRGB 255


def test_a_mirrored_instance_is_drawn_outside_out(viewer):
    """An improper instance matrix (det < 0, a mirror) flips the triangles' winding: valo draws those instances with
    the opposite front face, so a mirror-symmetric mesh looks exactly the same mirrored."""
    from valo import Camera, meshes

    m = viewer.renderer.material("M", "pbr", base_color=(0.9, 0.6, 0.3))
    cube = viewer.renderer.instances(meshes.cube(), m)
    viewer.show(SimpleNamespace(batches=lambda: [cube.batch()]))
    viewer.camera = Camera(azimuth=-35.0, elevation=25.0, distance=5.0)
    cube.set([np.eye(4)])
    proper = viewer.snapshot()
    cube.set([np.diag([-1.0, 1.0, 1.0, 1.0])])  # the cube is symmetric under x → -x
    assert cube.batch().mirrored == 1
    assert np.array_equal(viewer.snapshot(), proper)
