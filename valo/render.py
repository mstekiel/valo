"""GPU side: shared WGSL, meshes on the GPU, the orbit camera, and the Renderer.

Frame = scene pass (linear HDR into an offscreen rgba16float texture) → post pass (encode to sRGB
into the window region left of the panel) → ImGui pass (drawn by the app). sRGB encoding happens
only in POST_WGSL, so the window surface format doesn't matter.

Bind groups: 0 = frame (camera, light), 1 = material, 2 = per-instance model matrices.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass

import numpy as np
import wgpu

from .geometry import instance_matrices, look_at, pack_mat4, perspective_reversed_z
from .materials import FRAGMENT_WGSL, Material
from .meshes.data import MeshData
from .meshes.primitives import sphere

DEPTH_FORMAT = "depth32float"
SCENE_FORMAT = "rgba16float"  # linear HDR scene colour; encoded to sRGB in POST_WGSL

COMMON_WGSL = """
const PI: f32 = 3.14159265;

struct Frame { view_proj: mat4x4f, eye: vec4f, light_dir: vec4f };
@group(0) @binding(0) var<uniform> frame: Frame;

struct Material {
    base_color: vec4f,   // rgb (linear)
    specular: vec4f,     // rgb = specular colour, w = shininess
    pbr: vec4f,          // x = metallic, y = roughness, z = ambient, w = toon bands
    emissive: vec4f,     // rgb
};
@group(1) @binding(0) var<uniform> material: Material;

// Per-instance model matrix (math convention, packed by pack_mat4). One draw call per batch.
@group(2) @binding(0) var<storage, read> instances: array<mat4x4f>;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
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
    return out;
}

fn environment(dir: vec3f) -> vec3f {   // stand-in sky/ground for reflections
    return mix(vec3f(0.10, 0.09, 0.08), vec3f(0.55, 0.62, 0.72), clamp(dir.z * 0.5 + 0.5, 0.0, 1.0));
}

struct Surface { n: vec3f, v: vec3f, l: vec3f };
fn surface(in: VsOut) -> Surface {
    return Surface(normalize(in.normal), normalize(frame.eye.xyz - in.world), frame.light_dir.xyz);
}
"""

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
    def __init__(self, device, data: MeshData):
        self.vb = device.create_buffer_with_data(data=data.vertices, usage=wgpu.BufferUsage.VERTEX)
        self.ib = device.create_buffer_with_data(data=data.indices, usage=wgpu.BufferUsage.INDEX)
        self.count = len(data.indices)

    def draw(self, rpass, instances: int = 1):
        rpass.set_vertex_buffer(0, self.vb)
        rpass.set_index_buffer(self.ib, wgpu.IndexFormat.uint32)
        rpass.draw_indexed(self.count, instances)


@dataclass
class OrbitCamera:
    azimuth: float = -55.0
    elevation: float = 25.0
    distance: float = 14.0
    target: tuple = (0.0, 0.0, 0.0)
    fovy: float = 35.0

    def matrices(self, aspect: float):
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        eye = np.asarray(self.target) + self.distance * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        return perspective_reversed_z(self.fovy, aspect) @ look_at(eye, self.target), eye

    def fit(self, radius: float) -> None:
        """Move back far enough that a sphere of ``radius`` around the target fits vertically."""
        self.distance = max(3.0, radius / math.sin(math.radians(self.fovy / 2)) * 0.95)


class Instances:
    """Instances of one mesh sharing one material: a grow-only storage buffer of model matrices.

    ``batch()`` is the (material, mesh, instance bind group, count) tuple the renderer draws.
    """

    def __init__(self, device: wgpu.GPUDevice, instance_layout, mesh: GpuMesh, material: Material):
        self.device, self.layout, self.mesh, self.material = device, instance_layout, mesh, material
        self.count = self.capacity = 0  # capacity: instances the buffer can hold
        self.buffer = self.bind_group = None
        self.set(np.zeros((0, 4, 4)))

    def set(self, matrices) -> None:
        """Upload (N, 4, 4) math-convention model matrices, e.g. from ``geometry.instance_matrices``."""
        matrices = np.asarray(matrices, np.float32).reshape(-1, 4, 4)
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

    def batch(self):
        return self.material, self.mesh, self.bind_group, self.count


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
        self.frame_buffer = device.create_buffer(size=96, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
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

    def instances(self, mesh: MeshData, material: Material) -> Instances:
        return Instances(self.device, self.instance_layout, self.mesh(mesh), material)

    def spheres(self, material: Material) -> Spheres:
        return Spheres(self.device, self.instance_layout, self.sphere, material)

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

    def render(self, surface_texture, viewport: tuple[int, int, int, int], scene, camera: OrbitCamera) -> None:
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
        frame = np.zeros(24, np.float32)
        frame[:16] = pack_mat4(view_proj).ravel()
        frame[16:19] = eye
        light = np.array([-0.4, -0.6, 0.7])
        frame[20:23] = light / np.linalg.norm(light)
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
        for material, mesh, instances_bg, count in batches:  # one instanced draw call per batch
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
