"""GPU side: shared WGSL, meshes on the GPU, the orbit camera, and the Renderer.

Frame = scene pass (linear HDR into an offscreen rgba16float texture) → post pass (encode to sRGB
into the window region left of the panel) → ImGui pass (drawn by the app). sRGB encoding happens
only in POST_WGSL, so the window surface format doesn't matter.

Bind groups: 0 = frame (camera, light), 1 = material, 2 = per-instance model matrices.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import wgpu

from .geometry import instance_matrices, look_at, orthographic_reversed_z, pack_mat4, perspective_reversed_z
from .lighting import Lighting
from .materials import FRAGMENT_WGSL, Material
from .meshes.data import MeshData
from .meshes.primitives import sphere

DEPTH_FORMAT = "depth32float"
SCENE_FORMAT = "rgba16float"  # linear HDR scene colour; encoded to sRGB in POST_WGSL

VERTEX_WGSL = """
const PI: f32 = 3.14159265;

struct Frame {
    view_proj: mat4x4f,
    eye: vec4f,        // xyz camera position
    view_dir: vec4f,   // xyz unit forward; w = 1 perspective, 0 orthographic
    light_dir: vec4f,  // xyz unit vector toward the light
    light_color: vec4f,  // rgb = colour × intensity (linear)
    sky: vec4f,        // rgb = hemisphere colour above; w = ambient intensity
    ground: vec4f,     // rgb = hemisphere colour below
};
@group(0) @binding(0) var<uniform> frame: Frame;

// Per-instance model matrix (math convention, packed by pack_mat4). One draw call per batch.
@group(2) @binding(0) var<storage, read> instances: array<mat4x4f>;

struct VsOut {
    @builtin(position) @invariant pos: vec4f,   // invariant: the highlight pass reproduces the exact depth
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) @interpolate(flat) instance: u32,
};

@vertex
fn vs_main(@location(0) position: vec3f, @location(1) normal: vec3f,
           @builtin(instance_index) i: u32) -> VsOut {
    let m = instances[i];
    var out: VsOut;
    let world = (m * vec4f(position, 1.0)).xyz;
    out.pos = frame.view_proj * vec4f(world, 1.0);
    out.world = world;
    // Normals by the cofactor matrix (= det · inverse-transpose): right for non-uniform scale, no inverse.
    let a = m[0].xyz;
    let b = m[1].xyz;
    let c = m[2].xyz;
    out.normal = mat3x3f(cross(b, c), cross(c, a), cross(a, b)) * normal * sign(dot(a, cross(b, c)));
    out.instance = i;
    return out;
}
"""

SHADING_WGSL = """
struct Material {
    base_color: vec4f,   // rgb (linear)
    specular: vec4f,     // rgb = specular colour, w = shininess
    pbr: vec4f,          // x = metallic, y = roughness, z = ambient, w = toon bands
    emissive: vec4f,     // rgb
};
@group(1) @binding(0) var<uniform> material: Material;

fn environment(dir: vec3f) -> vec3f {   // the scene's sky/ground hemisphere, for reflections
    return mix(frame.ground.rgb, frame.sky.rgb, clamp(dir.z * 0.5 + 0.5, 0.0, 1.0));
}

fn ambient_light(n: vec3f) -> vec3f {   // global hemisphere ambient; materials scale it by their `ambient`
    return environment(n) * frame.sky.w;
}

struct Surface { n: vec3f, v: vec3f, l: vec3f };
fn surface(in: VsOut) -> Surface {
    // Toward the viewer: per pixel in perspective, the same everywhere (−forward) in orthographic.
    let v = select(-frame.view_dir.xyz, normalize(frame.eye.xyz - in.world), frame.view_dir.w > 0.5);
    return Surface(normalize(in.normal), v, frame.light_dir.xyz);
}
"""
COMMON_WGSL = VERTEX_WGSL + SHADING_WGSL  # what every shading model's fragment shader is appended to

PICK_WGSL = """
struct PickId { batch: vec4u };   // x = batch index + 1 (0 = nothing)
@group(1) @binding(0) var<uniform> pick: PickId;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4u {
    return vec4u(pick.batch.x, in.instance, bitcast<u32>(in.pos.z), 1u);
}
"""
PICK_FORMAT = "rgba32uint"  # (batch + 1, instance, depth bits, 1)


class Pick(NamedTuple):
    """What is under a pixel: the batch (index into ``scene.batches()``), the instance, and the world point hit."""

    batch_index: int
    instance_index: int
    source: object  # the Instances (or None for a plain tuple batch)
    world_point: np.ndarray

POST_WGSL = """
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> origin: vec4f;   // xy = viewport origin in framebuffer pixels

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));        // one triangle covering the viewport
    return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn linear_to_srgb(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

@fragment
fn fs_main(@builtin(position) p: vec4f) -> @location(0) vec4f {
    // Framebuffer coords → scene texel; the scene texture is exactly viewport-sized.
    let c = textureLoad(scene_tex, vec2i(p.xy - origin.xy), 0).rgb;
    return vec4f(linear_to_srgb(clamp(c, vec3f(0.0), vec3f(1.0))), 1.0);
}
"""


VERTEX_LAYOUT = {
    "array_stride": 24, "step_mode": "vertex",
    "attributes": [{"format": "float32x3", "offset": 0, "shader_location": 0},
                   {"format": "float32x3", "offset": 12, "shader_location": 1}],
}


class GpuMesh:
    """A MeshData uploaded to vertex and index buffers."""
    def __init__(self, device, data: MeshData):
        self.vb = device.create_buffer_with_data(data=data.vertices, usage=wgpu.BufferUsage.VERTEX)
        self.ib = device.create_buffer_with_data(data=data.indices, usage=wgpu.BufferUsage.INDEX)
        self.count = len(data.indices)

    def draw(self, rpass, instances: int = 1):
        """Draw the mesh ``instances`` times in the current render pass."""
        rpass.set_vertex_buffer(0, self.vb)
        rpass.set_index_buffer(self.ib, wgpu.IndexFormat.uint32)
        rpass.draw_indexed(self.count, instances)


PROJECTIONS = ("perspective", "orthographic")
VIEWS = {  # name → (azimuth, elevation): looking from that side toward the target
    "+X": (0.0, 0.0), "-X": (180.0, 0.0), "+Y": (90.0, 0.0), "-Y": (-90.0, 0.0),
    "+Z": (-90.0, 89.99), "-Z": (-90.0, -89.99), "Iso": (-55.0, 25.0),
}


@dataclass
class OrbitCamera:
    """A camera orbiting ``target`` at ``distance``, with a perspective or orthographic projection."""
    azimuth: float = -55.0
    elevation: float = 25.0
    distance: float = 14.0
    target: tuple = (0.0, 0.0, 0.0)
    fovy: float = 35.0
    projection: str = "perspective"  # or "orthographic"
    ortho_height: float = 10.0  # world units visible vertically in orthographic mode

    def eye(self) -> np.ndarray:
        """Camera position in world space."""
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        return np.asarray(self.target, float) + self.distance * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])

    def forward(self) -> np.ndarray:
        """Unit view direction, from the eye toward the target."""
        f = np.asarray(self.target, float) - self.eye()
        return f / np.linalg.norm(f)

    def projection_matrix(self, aspect: float) -> np.ndarray:
        """Projection matrix for the current projection mode (reversed-Z)."""
        if self.projection == "orthographic":  # the whole scene, even behind the eye, fits in depth
            return orthographic_reversed_z(self.ortho_height, aspect, -3 * self.distance, 5 * self.distance)
        return perspective_reversed_z(self.fovy, aspect)

    def matrices(self, aspect: float):
        """(view-projection matrix, eye position) for a viewport of the given aspect ratio."""
        eye = self.eye()
        return self.projection_matrix(aspect) @ look_at(eye, self.target), eye

    def _height_at_target(self) -> float:
        return 2 * self.distance * math.tan(math.radians(self.fovy / 2))

    def set_projection(self, projection: str, radius: float | None = None) -> None:
        """Switch projection without an apparent zoom.

        With the scene's bounding ``radius`` the outline of that sphere keeps its size on screen (perspective
        enlarges the near half, so matching only the target's depth looks like zooming out); without it, the
        plane through the target keeps its size.
        """
        if projection not in PROJECTIONS:
            raise ValueError(f"unknown projection {projection!r}")
        if projection == self.projection:
            return
        t = math.tan(math.radians(self.fovy / 2))
        use_sphere = radius is not None and 0 < radius < self.distance
        if projection == "orthographic":
            self.ortho_height = 2 * radius * t / self._outline(radius) if use_sphere else self._height_at_target()
        elif radius and radius > 0:  # the distance at which the sphere's outline fills the same share of the view
            self.distance = radius / math.sin(math.atan(2 * radius / self.ortho_height * t))
        else:
            self.distance = self.ortho_height / (2 * t)
        self.projection = projection

    def _outline(self, radius: float) -> float:
        """tan of the angle from the view axis to the outline of a sphere of ``radius`` around the target."""
        return math.tan(math.asin(radius / self.distance))

    def set_fov(self, degrees: float, radius: float | None = None) -> None:
        """Change the perspective field of view and move the camera so the scene keeps its size (a dolly zoom).

        With the scene's bounding ``radius`` its outline keeps its size; otherwise the plane through the target.
        """
        old = math.tan(math.radians(self.fovy / 2))
        new = math.tan(math.radians(float(degrees) / 2))
        if radius is not None and 0 < radius < self.distance:
            share = self._outline(radius) / old  # outline height as a share of the view
            self.distance = radius / math.sin(math.atan(share * new))
        else:
            self.distance *= old / new
        self.fovy = float(degrees)

    def zoom(self, factor: float) -> None:
        """Zoom out by ``factor`` (> 1) or in (< 1)."""
        if self.projection == "orthographic":
            self.ortho_height = float(np.clip(self.ortho_height * factor, 0.05, 500.0))
        else:
            self.distance = float(np.clip(self.distance * factor, 2.0, 120.0))

    def fit(self, radius: float) -> None:
        """Move back far enough that a sphere of ``radius`` around the target fits vertically (both projections)."""
        self.distance = max(3.0, radius / math.sin(math.radians(self.fovy / 2)) * 0.95)
        self.ortho_height = self._height_at_target()


class Batch(NamedTuple):
    """What a scene's ``batches()`` yields: one instanced draw. ``source`` is the Instances (for picking/info)."""

    material: Material
    mesh: GpuMesh
    bind_group: object
    count: int
    source: object = None


class Instances:
    """Instances of one mesh sharing one material: a grow-only storage buffer of model matrices.

    ``batch()`` is the Batch the renderer draws. ``label`` names the object in the Info panel, and
    ``info(i) -> dict`` (optional) adds scene-specific lines for instance ``i``.
    """

    def __init__(self, device: wgpu.GPUDevice, instance_layout, mesh: GpuMesh, material: Material,
                 label: str | None = None, info: Callable[[int], dict] | None = None):
        self.device, self.layout, self.mesh, self.material = device, instance_layout, mesh, material
        self.label, self.info = label, info
        self.count = self.capacity = 0  # capacity: instances the buffer can hold
        self.buffer = self.bind_group = None
        self.set(np.zeros((0, 4, 4)))

    def set(self, matrices) -> None:
        """Upload (N, 4, 4) math-convention model matrices, e.g. from ``geometry.instance_matrices``."""
        matrices = np.asarray(matrices, np.float32).reshape(-1, 4, 4)
        self.matrices = matrices.copy()  # CPU copy: picking, info and centring read it
        self.count = len(matrices)
        data = np.zeros((max(self.count, 1), 4, 4), np.float32)  # buffers can't be empty
        data[: self.count] = pack_mat4(matrices)
        if self.buffer is None or self.capacity < len(data):  # grow only; shrinking reuses the buffer
            self.capacity = max(len(data), 2 * self.capacity)
            self.buffer = self.device.create_buffer(
                size=self.capacity * 64, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
            self.bind_group = self.device.create_bind_group(
                layout=self.layout, entries=[{"binding": 0, "resource": {"buffer": self.buffer}}])
        self.device.queue.write_buffer(self.buffer, 0, data)

    def batch(self) -> Batch:
        """The Batch a scene's ``batches()`` returns for these instances."""
        return Batch(self.material, self.mesh, self.bind_group, self.count, self)


class Spheres(Instances):
    """Instances of the unit-diameter sphere, placed by centre and radius."""

    def set(self, centers, radii=0.5) -> None:
        """Upload centres (N, 3) and radii (one value or N values)."""
        centers = np.asarray(centers, np.float32).reshape(-1, 3)
        diameters = 2 * np.broadcast_to(np.asarray(radii, np.float32), len(centers))[:, None]
        super().set(instance_matrices(centers, diameters))


class Renderer:
    """Owns pipelines, layouts and render targets. Knows how to draw anything with ``batches()``."""

    def __init__(self, device: wgpu.GPUDevice, surface_format: str):
        self.device = device

        def layout(binding_type, visibility):
            return device.create_bind_group_layout(entries=[{
                "binding": 0, "visibility": visibility, "buffer": {"type": binding_type}}])

        vf = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        self.frame_layout = layout(wgpu.BufferBindingType.uniform, vf)
        self.material_layout = layout(wgpu.BufferBindingType.uniform, vf)
        self.instance_layout = layout(wgpu.BufferBindingType.read_only_storage, wgpu.ShaderStage.VERTEX)
        self.frame_buffer = device.create_buffer(size=160, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.frame_bg = device.create_bind_group(
            layout=self.frame_layout, entries=[{"binding": 0, "resource": {"buffer": self.frame_buffer}}])

        pl_layout = device.create_pipeline_layout(
            bind_group_layouts=[self.frame_layout, self.material_layout, self.instance_layout])
        self.pipelines = {}
        for name, frag in FRAGMENT_WGSL.items():  # one pipeline per shading model
            module = device.create_shader_module(code=COMMON_WGSL + frag)
            self.pipelines[name] = device.create_render_pipeline(
                layout=pl_layout,
                vertex={"module": module, "entry_point": "vs_main", "buffers": [VERTEX_LAYOUT]},
                primitive={"topology": "triangle-list", "cull_mode": "back"},
                depth_stencil={"format": DEPTH_FORMAT, "depth_write_enabled": True, "depth_compare": "greater"},
                fragment={"module": module, "entry_point": "fs_main", "targets": [{"format": SCENE_FORMAT}]},
            )

        post = device.create_shader_module(code=POST_WGSL)
        self.post_pipeline = device.create_render_pipeline(
            layout="auto",
            vertex={"module": post, "entry_point": "vs_main"},
            primitive={"topology": "triangle-list"},
            fragment={"module": post, "entry_point": "fs_main", "targets": [{"format": surface_format}]},
        )
        self.post_origin = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._meshes: weakref.WeakKeyDictionary[MeshData, GpuMesh] = weakref.WeakKeyDictionary()
        self.sphere = self.mesh(sphere(20, 40))  # unit diameter; scaled per instance
        self._targets = None

    def material(self, name: str, shading: str, **params) -> Material:
        """A new material bound with this renderer's layout."""
        return Material(self.device, self.material_layout, name, shading, **params)

    def mesh(self, data: MeshData) -> GpuMesh:
        """``data`` on the GPU, uploaded once per MeshData object (and released when that object is gone)."""
        if data not in self._meshes:
            self._meshes[data] = GpuMesh(self.device, data)
        return self._meshes[data]

    def instances(self, mesh: MeshData, material: Material, label: str | None = None, info=None) -> Instances:
        """New instanced batch of ``mesh`` with ``material`` (call ``.set(matrices)`` to place them)."""
        return Instances(self.device, self.instance_layout, self.mesh(mesh), material, label, info)

    def spheres(self, material: Material, label: str | None = None, info=None) -> Spheres:
        """New instanced batch of unit spheres with ``material`` (call ``.set(centers, radii)``)."""
        return Spheres(self.device, self.instance_layout, self.sphere, material, label, info)

    def _pick_setup(self):
        """The picking pipeline and its layouts, made on first use."""
        if getattr(self, "_pick_pipeline", None) is None:
            d = self.device
            self._pick_layout = d.create_bind_group_layout(entries=[{
                "binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}}])
            module = d.create_shader_module(code=VERTEX_WGSL + PICK_WGSL)
            self._pick_pipeline = d.create_render_pipeline(
                layout=d.create_pipeline_layout(
                    bind_group_layouts=[self.frame_layout, self._pick_layout, self.instance_layout]),
                vertex={"module": module, "entry_point": "vs_main", "buffers": [VERTEX_LAYOUT]},
                primitive={"topology": "triangle-list", "cull_mode": "back"},
                depth_stencil={"format": DEPTH_FORMAT, "depth_write_enabled": True, "depth_compare": "greater"},
                fragment={"module": module, "entry_point": "fs_main", "targets": [{"format": PICK_FORMAT}]},
            )
            self._pick_ids: list = []  # bind group per batch index, holding that index + 1
            self._pick_targets = None

    def _pick_id(self, i: int):
        while len(self._pick_ids) <= i:
            n = len(self._pick_ids)
            buf = self.device.create_buffer_with_data(data=np.array([n + 1, 0, 0, 0], np.uint32),
                                                      usage=wgpu.BufferUsage.UNIFORM)
            self._pick_ids.append(self.device.create_bind_group(
                layout=self._pick_layout, entries=[{"binding": 0, "resource": {"buffer": buf}}]))
        return self._pick_ids[i]

    def pick(self, viewport: tuple[int, int, int, int], scene, camera: OrbitCamera, x: int, y: int) -> Pick | None:
        """What ``scene`` shows at framebuffer pixel (x, y) inside ``viewport``, or None for the background.

        Renders object ids into a 1-pixel scissor (cheap: vertex work only) and reads that pixel back.
        Call after ``render`` with the same viewport and camera (it reuses the frame uniform).
        """
        vx, vy, vw, vh = viewport
        px, py = int(x) - vx, int(y) - vy
        if not (0 <= px < vw and 0 <= py < vh):
            return None
        self._pick_setup()
        if self._pick_targets is None or self._pick_targets[0] != (vw, vh):
            usage = wgpu.TextureUsage.RENDER_ATTACHMENT
            ids = self.device.create_texture(size=(vw, vh, 1), format=PICK_FORMAT,
                                             usage=usage | wgpu.TextureUsage.COPY_SRC)
            depth = self.device.create_texture(size=(vw, vh, 1), format=DEPTH_FORMAT, usage=usage)
            self._pick_targets = ((vw, vh), ids, depth)
        _, ids, depth = self._pick_targets
        batches = list(scene.batches())
        encoder = self.device.create_command_encoder()
        rpass = encoder.begin_render_pass(
            color_attachments=[{"view": ids.create_view(), "clear_value": (0, 0, 0, 0),
                                "load_op": "clear", "store_op": "store"}],
            depth_stencil_attachment={"view": depth.create_view(), "depth_clear_value": 0.0,
                                      "depth_load_op": "clear", "depth_store_op": "store"},
        )
        rpass.set_scissor_rect(px, py, 1, 1)
        rpass.set_pipeline(self._pick_pipeline)
        rpass.set_bind_group(0, self.frame_bg)
        for i, b in enumerate(batches):
            if b[3] > 0:
                rpass.set_bind_group(1, self._pick_id(i))
                rpass.set_bind_group(2, b[2])
                b[1].draw(rpass, instances=b[3])
        rpass.end()
        self.device.queue.submit([encoder.finish()])
        data = self.device.queue.read_texture({"texture": ids, "origin": (px, py, 0)},
                                              {"bytes_per_row": 16}, (1, 1, 1))
        batch1, instance, depth_bits, _ = np.frombuffer(data, np.uint32)[:4]
        if batch1 == 0:
            return None
        z = float(np.array([depth_bits], np.uint32).view(np.float32)[0])
        ndc = np.array([(px + 0.5) / vw * 2 - 1, 1 - (py + 0.5) / vh * 2, z, 1.0])
        world = np.linalg.inv(camera.matrices(vw / vh)[0]) @ ndc
        b = batches[batch1 - 1]
        return Pick(int(batch1) - 1, int(instance), getattr(b, "source", None), world[:3] / world[3])

    def _scene_targets(self, w: int, h: int):
        if self._targets is None or self._targets[0] != (w, h):
            usage = wgpu.TextureUsage.RENDER_ATTACHMENT
            color = self.device.create_texture(size=(w, h, 1), format=SCENE_FORMAT,
                                               usage=usage | wgpu.TextureUsage.TEXTURE_BINDING)
            depth = self.device.create_texture(size=(w, h, 1), format=DEPTH_FORMAT, usage=usage)
            bg = self.device.create_bind_group(layout=self.post_pipeline.get_bind_group_layout(0),
                                               entries=[{"binding": 0, "resource": color.create_view()},
                                                        {"binding": 1, "resource": {"buffer": self.post_origin}}])
            self._targets = ((w, h), color, depth, bg)
        return self._targets[1:]

    def render(self, surface_texture, viewport: tuple[int, int, int, int], scene, camera: OrbitCamera,
               lighting: Lighting | None = None) -> None:
        """Draw ``scene`` into the ``viewport`` = (x, y, w, h) rectangle of ``surface_texture`` (framebuffer
        pixels, origin top-left); the rest of the texture is cleared to the window background.

        ``scene.batches()`` yields (material, gpu_mesh, instance_bind_group, instance_count) tuples.
        """
        tw, th = surface_texture.size[:2]
        x, y = (int(np.clip(c, 0, lim - 1)) for c, lim in zip(viewport[:2], (tw, th), strict=True))
        vw, h = (max(1, min(int(c), lim - o)) for c, lim, o in zip(viewport[2:], (tw, th), (x, y), strict=True))
        color, depth, post_bg = self._scene_targets(vw, h)
        self.device.queue.write_buffer(self.post_origin, 0, np.array([x, y, 0, 0], np.float32))

        view_proj, eye = camera.matrices(vw / h)
        lighting = lighting or Lighting()
        frame = np.zeros(40, np.float32)
        frame[:16] = pack_mat4(view_proj).ravel()
        frame[16:19] = eye
        frame[20:23] = camera.forward()
        frame[23] = camera.projection == "perspective"
        frame[24:27] = lighting.direction(camera)
        frame[28:31] = lighting.radiance()
        frame[32:35], frame[35], frame[36:39] = lighting.sky, lighting.ambient, lighting.ground
        self.device.queue.write_buffer(self.frame_buffer, 0, frame)

        batches = [b for b in scene.batches() if b[3] > 0]
        for material, *_ in batches:
            material.flush()  # no-op unless edited since last frame

        encoder = self.device.create_command_encoder()
        rpass = encoder.begin_render_pass(
            color_attachments=[{"view": color.create_view(), "clear_value": (0.02, 0.022, 0.026, 1),
                                "load_op": "clear", "store_op": "store"}],
            depth_stencil_attachment={"view": depth.create_view(), "depth_clear_value": 0.0,
                                      "depth_load_op": "clear", "depth_store_op": "store"},
        )
        rpass.set_bind_group(0, self.frame_bg)
        for material, mesh, instances_bg, count, *_ in batches:  # one instanced draw call per batch
            rpass.set_pipeline(self.pipelines[material.shading])
            rpass.set_bind_group(1, material.bind_group)
            rpass.set_bind_group(2, instances_bg)
            mesh.draw(rpass, instances=count)
        rpass.end()

        rpass = encoder.begin_render_pass(color_attachments=[{
            "view": surface_texture.create_view(), "clear_value": (0.12, 0.12, 0.13, 1),
            "load_op": "clear", "store_op": "store"}])
        rpass.set_viewport(x, y, vw, h, 0.0, 1.0)
        rpass.set_pipeline(self.post_pipeline)
        rpass.set_bind_group(0, post_bg)
        rpass.draw(3)
        rpass.end()
        self.device.queue.submit([encoder.finish()])
