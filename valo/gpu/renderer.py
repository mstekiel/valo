"""The Renderer: pipelines, render targets, the scene / highlight / post passes, and picking.

Frame = scene pass (linear HDR into an offscreen rgba16float texture) → post pass (encode to sRGB
into the window region left of the panel) → ImGui pass (drawn by the app). sRGB encoding happens
only in POST_WGSL, so the window surface format doesn't matter.

Bind groups: 0 = frame (camera, light), 1 = material, 2 = per-instance model matrices.
"""

from __future__ import annotations

import functools
import weakref
from typing import NamedTuple

import numpy as np
import wgpu

from ..config import rc
from ..math import pack_mat4
from ..meshes.data import MeshData
from ..meshes.primitives import sphere
from .instances import Instances, Spheres
from .material import Material
from .mesh import VERTEX_LAYOUT, GpuMesh
from .shaders import COMMON_WGSL, FRAGMENT_WGSL, HIGHLIGHT_WGSL, PICK_WGSL, POST_WGSL, VERTEX_WGSL

__all__ = ["DEPTH_FORMAT", "PICK_FORMAT", "SCENE_FORMAT", "Pick", "Renderer", "get_device", "renderer"]


DEPTH_FORMAT = "depth32float"
SCENE_FORMAT = "rgba16float"  # linear HDR scene colour; encoded to sRGB in POST_WGSL


PICK_FORMAT = "rgba32uint"  # (batch + 1, instance, depth bits, 1)


class Pick(NamedTuple):
    """What is under a pixel: the batch (index into ``scene.batches()``), the instance, and the world point hit."""

    batch_index: int
    instance_index: int
    source: object  # the Instances (or None for a plain tuple batch)
    world_point: np.ndarray


@functools.cache
def get_device() -> tuple[wgpu.GPUAdapter, wgpu.GPUDevice]:
    """The process-wide high-performance GPU adapter and device (created once, shared by every renderer)."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    return adapter, adapter.request_device_sync()


@functools.cache
def renderer() -> Renderer:
    """The shared Renderer on the shared device: build scenes with it, then show them in a Viewer or
    render them offline with ``valo.render``."""
    return Renderer(get_device()[1])


class Renderer:
    """Owns pipelines, layouts and render targets. Knows how to draw anything with ``batches()``."""

    def __init__(self, device: wgpu.GPUDevice, surface_format: str | None = None):
        """``surface_format`` is optional: the post pipeline for each output format is made on first use."""
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

        self._post_module = device.create_shader_module(code=POST_WGSL)
        self.post_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}},
        ])
        self._post_pipelines: dict[str, wgpu.GPURenderPipeline] = {}
        hl_layout = layout(wgpu.BufferBindingType.uniform, wgpu.ShaderStage.FRAGMENT)
        hl_module = device.create_shader_module(code=VERTEX_WGSL + HIGHLIGHT_WGSL)
        add = {"color": {"src_factor": "one", "dst_factor": "one", "operation": "add"},
               "alpha": {"src_factor": "zero", "dst_factor": "one", "operation": "add"}}
        self.highlight_pipeline = device.create_render_pipeline(  # drawn over the scene, same depth
            layout=device.create_pipeline_layout(
                bind_group_layouts=[self.frame_layout, hl_layout, self.instance_layout]),
            vertex={"module": hl_module, "entry_point": "vs_main", "buffers": [VERTEX_LAYOUT]},
            primitive={"topology": "triangle-list", "cull_mode": "back"},
            depth_stencil={"format": DEPTH_FORMAT, "depth_write_enabled": False, "depth_compare": "greater-equal"},
            fragment={"module": hl_module, "entry_point": "fs_main",
                      "targets": [{"format": SCENE_FORMAT, "blend": add}]},
        )
        self._hl_layout, self._hl_buffers = hl_layout, []  # one tint buffer per highlight, made on demand
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

    def _hl_buffer(self, i: int):
        while len(self._hl_buffers) <= i:
            buf = self.device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            self._hl_buffers.append((buf, self.device.create_bind_group(
                layout=self._hl_layout, entries=[{"binding": 0, "resource": {"buffer": buf}}])))
        return self._hl_buffers[i]

    def _pick_id(self, i: int):
        while len(self._pick_ids) <= i:
            n = len(self._pick_ids)
            buf = self.device.create_buffer_with_data(data=np.array([n + 1, 0, 0, 0], np.uint32),
                                                      usage=wgpu.BufferUsage.UNIFORM)
            self._pick_ids.append(self.device.create_bind_group(
                layout=self._pick_layout, entries=[{"binding": 0, "resource": {"buffer": buf}}]))
        return self._pick_ids[i]

    def pick(self, viewport: tuple[int, int, int, int], scene, camera, x: int, y: int) -> Pick | None:
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

    def post_pipeline(self, output_format: str):
        """The post pass (linear scene → sRGB) writing to ``output_format`` (made on first use)."""
        if output_format not in self._post_pipelines:
            self._post_pipelines[output_format] = self.device.create_render_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[self.post_layout]),
                vertex={"module": self._post_module, "entry_point": "vs_main"},
                primitive={"topology": "triangle-list"},
                fragment={"module": self._post_module, "entry_point": "fs_main",
                          "targets": [{"format": output_format}]},
            )
        return self._post_pipelines[output_format]

    def _scene_targets(self, w: int, h: int):
        if self._targets is None or self._targets[0] != (w, h):
            usage = wgpu.TextureUsage.RENDER_ATTACHMENT
            color = self.device.create_texture(size=(w, h, 1), format=SCENE_FORMAT,
                                               usage=usage | wgpu.TextureUsage.TEXTURE_BINDING)
            depth = self.device.create_texture(size=(w, h, 1), format=DEPTH_FORMAT, usage=usage)
            bg = self.device.create_bind_group(layout=self.post_layout,
                                               entries=[{"binding": 0, "resource": color.create_view()},
                                                        {"binding": 1, "resource": {"buffer": self.post_origin}}])
            self._targets = ((w, h), color, depth, bg)
        return self._targets[1:]

    def render(self, surface_texture, viewport: tuple[int, int, int, int], scene, camera,
               lighting, highlights=(), samples: int = 1, transparent: bool = False, background=None) -> None:
        """Draw ``scene`` into the ``viewport`` = (x, y, w, h) rectangle of ``surface_texture`` (framebuffer
        pixels, origin top-left); the rest of the texture is cleared to the window background.

        ``scene.batches()`` yields Batch (material, gpu_mesh, instance_bind_group, instance_count, source)
        tuples. ``lighting`` is a ``valo.Lighting`` (anything with ``direction(camera)``, ``radiance()``, ``sky``,
        ``ground`` and ``ambient``). ``highlights``: (batch_index, instance_index, rgb) tints drawn over those
        instances; an instance_index of None tints every instance of the batch (a whole object).
        ``samples`` k renders the scene k×k larger and averages it down; ``transparent`` gives the background
        alpha 0; ``background`` (linear RGB) overrides ``rc.render.background``.
        """
        tw, th = surface_texture.size[:2]
        x, y = (int(np.clip(c, 0, lim - 1)) for c, lim in zip(viewport[:2], (tw, th), strict=True))
        vw, h = (max(1, min(int(c), lim - o)) for c, lim, o in zip(viewport[2:], (tw, th), (x, y), strict=True))
        k = max(1, int(samples))
        color, depth, post_bg = self._scene_targets(vw * k, h * k)
        self.device.queue.write_buffer(self.post_origin, 0, np.array([x, y, k, 0], np.float32))

        view_proj, eye = camera.matrices(vw / h)
        frame = np.zeros(40, np.float32)
        frame[:16] = pack_mat4(view_proj).ravel()
        frame[16:19] = eye
        frame[20:23] = camera.forward()
        frame[23] = camera.projection == "perspective"
        frame[24:27] = lighting.direction(camera)
        frame[28:31] = lighting.radiance()
        frame[32:35], frame[35], frame[36:39] = lighting.sky, lighting.ambient, lighting.ground
        self.device.queue.write_buffer(self.frame_buffer, 0, frame)

        all_batches = list(scene.batches())
        batches = [b for b in all_batches if b[3] > 0]
        for material, *_ in batches:
            material.flush()  # no-op unless edited since last frame

        encoder = self.device.create_command_encoder()
        rpass = encoder.begin_render_pass(
            color_attachments=[{"view": color.create_view(),
                                "clear_value": (*(rc.render.background if background is None else background),
                                                0.0 if transparent else 1.0),
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
        if highlights:
            rpass.set_pipeline(self.highlight_pipeline)
        for i, (bi, ii, rgb) in enumerate(highlights):
            if bi < len(all_batches) and (ii is None or ii < all_batches[bi][3]):
                buf, bg = self._hl_buffer(i)
                self.device.queue.write_buffer(buf, 0, np.array([*rgb, 0.0], np.float32))
                rpass.set_bind_group(1, bg)
                rpass.set_bind_group(2, all_batches[bi][2])
                first, n = (0, all_batches[bi][3]) if ii is None else (ii, 1)
                all_batches[bi][1].draw(rpass, instances=n, first_instance=first)
        rpass.end()

        rpass = encoder.begin_render_pass(color_attachments=[{
            "view": surface_texture.create_view(), "clear_value": (*rc.ui.window_background, 1),
            "load_op": "clear", "store_op": "store"}])
        rpass.set_viewport(x, y, vw, h, 0.0, 1.0)
        rpass.set_pipeline(self.post_pipeline(surface_texture.format))
        rpass.set_bind_group(0, post_bg)
        rpass.draw(3)
        rpass.end()
        self.device.queue.submit([encoder.finish()])
