"""All WGSL of the renderer: the shared vertex stage and lighting helpers, one fragment shader per shading model
(``FRAGMENT_WGSL``, keyed like ``material.SCHEMA``), and the pick, highlight, overlay-line and post passes.

Add a shading model: a WGSL entry in ``FRAGMENT_WGSL`` plus its parameter list in ``material.SCHEMA``.
"""

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


LINES_WGSL = """
// Overlay lines (e.g. bounding boxes), drawn in the scene pass against its depth: each segment is one instance of a
// 6-vertex quad, widened on screen to a fixed width in pixels.
struct LineFrame { size: vec4f };   // xy = viewport, z = line width (framebuffer px), w = depth bias (relative)
@group(1) @binding(0) var<uniform> lines: LineFrame;
struct Segment { a: vec4f, b: vec4f, color: vec4f };   // ends in world xyz, colour linear RGBA
@group(2) @binding(0) var<storage, read> segments: array<Segment>;

struct LineOut { @builtin(position) pos: vec4f, @location(0) color: vec4f };

@vertex
fn vs_line(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> LineOut {
    let s = segments[i];
    var p = frame.view_proj * vec4f(s.a.xyz, 1.0);
    var q = frame.view_proj * vec4f(s.b.xyz, 1.0);
    var out: LineOut;
    out.color = s.color;
    let near = 1e-4;
    if (p.w < near && q.w < near) {   // all of it behind the camera
        out.pos = vec4f(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    if (p.w < near) { p = mix(p, q, (near - p.w) / (q.w - p.w)); }   // cut where it crosses the near side
    if (q.w < near) { q = mix(q, p, (near - q.w) / (p.w - q.w)); }
    let half = lines.size.xy * 0.5;
    var d = (q.xy / q.w - p.xy / p.w) * half;   // the segment on screen, pixels
    if (length(d) < 1e-6) { d = vec2f(1.0, 0.0); }
    let n = normalize(vec2f(-d.y, d.x)) * lines.size.z * 0.5 / half;   // half the width, in NDC
    var ends = array<f32, 6>(0.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    var sides = array<f32, 6>(-1.0, 1.0, -1.0, -1.0, 1.0, 1.0);
    let c = select(p, q, ends[v] > 0.5);
    // a little nearer than the surface it may lie on (reversed-Z: larger is nearer), so edges on a mesh don't flicker
    out.pos = vec4f(c.xy + n * sides[v] * c.w, c.z * (1.0 + lines.size.w), c.w);
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4f {
    return in.color;
}
"""

HIGHLIGHT_WGSL = """
struct Highlight { color: vec4f };   // rgb = colour × strength
@group(1) @binding(0) var<uniform> hl: Highlight;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {   // additive: a soft tint plus a brighter rim
    let n = normalize(in.normal);
    let v = select(-frame.view_dir.xyz, normalize(frame.eye.xyz - in.world), frame.view_dir.w > 0.5);
    let rim = pow(1.0 - max(dot(n, v), 0.0), 2.0);
    return vec4f(hl.color.rgb * (0.35 + rim), 0.0);
}
"""

POST_WGSL = """
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> origin: vec4f;   // xy = viewport origin in framebuffer pixels, z = samples

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
    // Framebuffer coords → a k×k block of scene texels (the scene texture is k times the viewport: supersampling).
    // Colours are averaged premultiplied by alpha (alpha < 1 only on a transparent background), then encoded.
    let k = i32(origin.z);
    let base = vec2i(p.xy - origin.xy) * k;
    var sum = vec4f(0.0);
    for (var y = 0; y < k; y++) {
        for (var x = 0; x < k; x++) {
            let t = textureLoad(scene_tex, base + vec2i(x, y), 0);
            sum += vec4f(t.rgb * t.a, t.a);
        }
    }
    let a = sum.a / f32(k * k);
    let c = select(vec3f(0.0), sum.rgb / max(sum.a, 1e-6), sum.a > 0.0);
    return vec4f(linear_to_srgb(clamp(c, vec3f(0.0), vec3f(1.0))), a);
}
"""

FRAGMENT_WGSL = {
    "unlit": """
@fragment fn fs_main(in: VsOut) -> @location(0) vec4f { return vec4f(material.base_color.rgb, 1.0); }""",
    "lambert": """
@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
    let s = surface(in);
    let d = max(dot(s.n, s.l), 0.0);
    let lit = ambient_light(s.n) * material.pbr.z + d * frame.light_color.rgb;
    return vec4f(material.base_color.rgb * lit + material.emissive.rgb, 1.0);
}""",
    "blinn_phong": """
@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
    let s = surface(in);
    let d = max(dot(s.n, s.l), 0.0);
    let spec = select(0.0, pow(max(dot(s.n, normalize(s.l + s.v)), 0.0), material.specular.w), d > 0.0);
    let lit = ambient_light(s.n) * material.pbr.z + d * frame.light_color.rgb;
    return vec4f(material.base_color.rgb * lit
                 + material.specular.rgb * spec * frame.light_color.rgb + material.emissive.rgb, 1.0);
}""",
    "pbr": """
@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {   // metallic-roughness Cook-Torrance (GGX)
    let s = surface(in);
    let base = material.base_color.rgb;
    let metallic = material.pbr.x;
    let rough = max(material.pbr.y, 0.04);
    let h = normalize(s.l + s.v);
    let ndl = max(dot(s.n, s.l), 0.0);
    let ndv = max(dot(s.n, s.v), 1e-4);
    let ndh = max(dot(s.n, h), 0.0);
    let vdh = max(dot(s.v, h), 0.0);
    let a2 = pow(rough, 4.0);
    let dd = a2 / (PI * pow(ndh * ndh * (a2 - 1.0) + 1.0, 2.0));
    let k = (rough + 1.0) * (rough + 1.0) / 8.0;
    let g = (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k));
    let f0 = mix(vec3f(0.04), base, metallic);
    let f = f0 + (1.0 - f0) * pow(1.0 - vdh, 5.0);
    let spec = dd * g * f / max(4.0 * ndv * ndl, 1e-4);
    let direct = ((1.0 - f) * (1.0 - metallic) * base / PI + spec) * 3.0 * ndl * frame.light_color.rgb;
    let fr = f0 + (max(vec3f(1.0 - rough), f0) - f0) * pow(1.0 - ndv, 5.0);
    let env = environment(reflect(-s.v, s.n)) * fr * (1.0 - 0.75 * rough);
    let ambient = material.pbr.z * ambient_light(s.n) * base * (1.0 - metallic);
    return vec4f(direct + env + ambient + material.emissive.rgb, 1.0);
}""",
    "toon": """
@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
    let s = surface(in);
    let bands = max(material.pbr.w, 1.0);
    let d = floor(max(dot(s.n, s.l), 0.0) * bands + 0.5) / bands;
    let rim = smoothstep(0.55, 0.75, 1.0 - max(dot(s.n, s.v), 0.0)) * 0.35;
    let lit = ambient_light(s.n) * material.pbr.z + d * frame.light_color.rgb;
    return vec4f(material.base_color.rgb * lit + vec3f(rim) + material.emissive.rgb, 1.0);
}""",
}
