/* Minimal WebGL2 scene viewer with Blender-style navigation.
 *
 * Written by hand rather than pulled from a CDN so the dashboard keeps working
 * offline and over a forwarded port. Geometry arrives as one binary float32
 * buffer plus a JSON manifest of per-object ranges, which keeps a 50k-triangle
 * interior under a couple of megabytes.
 *
 * The frame is drawn in four passes, because flat shading makes an interior
 * unreadable: every wall meets every other wall at the same brightness, so the
 * rooms read as one grey mass.
 *
 *   1. shadow    scene depth from the key light, into a depth texture
 *   2. scene     colour with a key light, hemispheric ambient and soft shadows
 *   3. occlusion screen-space ambient occlusion from the depth buffer, blurred
 *   4. composite colour x occlusion, contact edges, background, tone mapping
 *
 * Rays and device markers are drawn last, straight to the screen: they are
 * annotation rather than geometry, and should not pick up occlusion.
 *
 * Navigation matches Blender's defaults closely enough to be muscle-memory:
 *   drag            orbit
 *   shift+drag      pan
 *   wheel           dolly
 *   F               frame all
 *   1 / 3 / 7       front / right / top, as on the numpad
 */

import { M4 } from './mat4.js';

const SCENE_VERT = `#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 uMVP;
uniform mat4 uLightVP;
out vec3 vColor;
out vec3 vWorld;
out vec4 vLight;
void main() {
  vColor = aColor;
  vWorld = aPos;
  vLight = uLightVP * vec4(aPos, 1.0);
  gl_Position = uMVP * vec4(aPos, 1.0);
}`;

const SCENE_FRAG = `#version 300 es
precision highp float;
precision highp sampler2DShadow;
in vec3 vColor;
in vec3 vWorld;
in vec4 vLight;
uniform float uClipZ;
uniform float uClipOn;
uniform vec3 uEye;
uniform vec3 uLightDir;
uniform vec3 uSky;
uniform vec3 uGround;
uniform vec3 uKey;
uniform float uAmbient;
uniform float uShadowTexel;
uniform sampler2DShadow uShadow;
uniform vec3 uTint;
uniform float uPlaneMode;
uniform vec2 uPlaneCentre;
uniform vec2 uPlaneFade;
uniform vec3 uFadeTo;
out vec4 outColor;

float shadowed(vec3 normal, float ndl) {
  vec3 p = vLight.xyz / vLight.w * 0.5 + 0.5;
  if (p.z > 1.0 || p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) return 1.0;
  // Slope-scaled bias: a surface seen edge-on by the light spans many depth
  // values inside one texel, which is what produces acne without it.
  float bias = mix(0.0035, 0.0006, ndl);
  float sum = 0.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 o = vec2(float(x), float(y)) * uShadowTexel;
      sum += texture(uShadow, vec3(p.xy + o, p.z - bias));
    }
  }
  return sum / 9.0;
}

void main() {
  // Section plane: discard anything above the cut so interiors are visible.
  if (uClipOn > 0.5 && vWorld.z > uClipZ) discard;
  // Flat shading from screen-space derivatives: no normals need sending, and
  // an interior is faceted anyway.
  vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
  vec3 v = normalize(uEye - vWorld);
  if (dot(n, v) < 0.0) n = -n;             // walls are seen from both sides

  vec3 l = normalize(uLightDir);
  float ndl = max(dot(n, l), 0.0);
  float lit = ndl * shadowed(n, ndl);

  // Hemispheric ambient: sky above, bounced floor light below. This is what
  // separates a ceiling from a floor when neither faces the key light.
  vec3 ambient = mix(uGround, uSky, n.z * 0.5 + 0.5) * uAmbient;
  vec3 h = normalize(l + v);
  float spec = pow(max(dot(n, h), 0.0), 42.0) * 0.10 * lit;
  vec3 shaded = vColor * uTint * (ambient + uKey * lit) + spec;
  if (uPlaneMode > 0.5) {
    // The plane has to end somewhere, and a straight edge across the frame
    // reads as a mistake. Fading it into the background colour hides the edge
    // wherever the camera is, without pretending the floor is infinite.
    float d = distance(vWorld.xy, uPlaneCentre);
    shaded = mix(shaded, uFadeTo, smoothstep(uPlaneFade.x, uPlaneFade.y, d));
  }
  outColor = vec4(shaded, 1.0);
}`;

const DEPTH_VERT = `#version 300 es
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
out vec3 vWorld;
void main() { vWorld = aPos; gl_Position = uMVP * vec4(aPos, 1.0); }`;

const DEPTH_FRAG = `#version 300 es
precision highp float;
in vec3 vWorld;
uniform float uClipZ;
uniform float uClipOn;
void main() { if (uClipOn > 0.5 && vWorld.z > uClipZ) discard; }`;

const POST_VERT = `#version 300 es
out vec2 vUV;
void main() {
  // One oversized triangle, no buffer: cheaper than a quad and avoids a VAO.
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUV = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const AO_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uDepth;
uniform mat4 uProj;
uniform mat4 uInvProj;
uniform float uRadius;
out float outAO;

vec3 viewPos(vec2 uv) {
  float d = texture(uDepth, uv).r;
  vec4 clip = vec4(uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
  vec4 p = uInvProj * clip;
  return p.xyz / p.w;
}

void main() {
  if (texture(uDepth, vUV).r >= 1.0) { outAO = 1.0; return; }
  vec3 p = viewPos(vUV);
  // Normals are reconstructed from the depth buffer rather than stored: one
  // fewer render target, and the result is exact for flat architecture.
  vec3 n = normalize(cross(dFdx(p), dFdy(p)));
  if (dot(n, -normalize(p)) < 0.0) n = -n;
  vec3 tangent = normalize(abs(n.z) < 0.9 ? cross(n, vec3(0, 0, 1)) : cross(n, vec3(1, 0, 0)));
  vec3 bitangent = cross(n, tangent);
  // Per-pixel rotation turns banding into noise, which the blur then removes.
  float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);

  const int SAMPLES = 16;
  float golden = 2.39996323;
  float occlusion = 0.0;
  for (int i = 0; i < SAMPLES; i++) {
    float t = (float(i) + 0.5) / float(SAMPLES);
    float angle = float(i) * golden + jitter * 6.2831853;
    vec3 dir = normalize(
      tangent * cos(angle) + bitangent * sin(angle) + n * (0.35 + 0.65 * t));
    vec3 samplePos = p + dir * uRadius * sqrt(t);

    vec4 proj = uProj * vec4(samplePos, 1.0);
    vec2 uv = (proj.xy / proj.w) * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;
    vec3 q = viewPos(uv);
    // View space looks down -z, so a larger z is nearer: the sample is
    // occluded when the surface there sits in front of it.
    if (q.z > samplePos.z + 0.02) {
      // Range check keeps a distant surface from casting occlusion.
      occlusion += smoothstep(0.0, 1.0, uRadius / max(abs(p.z - q.z), 1e-4));
    }
  }
  outAO = clamp(1.0 - occlusion / float(SAMPLES), 0.0, 1.0);
}`;

const BLUR_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uAO;
uniform vec2 uTexel;
out float outAO;
void main() {
  float sum = 0.0;
  for (int y = -2; y <= 2; y++) {
    for (int x = -2; x <= 2; x++) sum += texture(uAO, vUV + vec2(float(x), float(y)) * uTexel).r;
  }
  outAO = sum / 25.0;
}`;

const COMPOSITE_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uColor;
uniform sampler2D uAO;
uniform sampler2D uDepth;
uniform vec2 uTexel;
uniform vec3 uBackTop;
uniform vec3 uBackBottom;
uniform vec3 uEdge;
uniform float uAOStrength;
uniform float uEdgeStrength;
uniform float uExposure;
out vec4 outColor;

void main() {
  float depth = texture(uDepth, vUV).r;
  gl_FragDepth = depth;
  if (depth >= 1.0) {
    outColor = vec4(mix(uBackBottom, uBackTop, vUV.y), 1.0);
    return;
  }
  vec3 colour = texture(uColor, vUV).rgb;
  float ao = mix(1.0, texture(uAO, vUV).r, uAOStrength);

  // Contact edges: a depth step much larger than its neighbours' spread is a
  // silhouette, and outlining it is what makes rooms legible from outside.
  float d0 = texture(uDepth, vUV + vec2(uTexel.x, 0)).r;
  float d1 = texture(uDepth, vUV - vec2(uTexel.x, 0)).r;
  float d2 = texture(uDepth, vUV + vec2(0, uTexel.y)).r;
  float d3 = texture(uDepth, vUV - vec2(0, uTexel.y)).r;
  float curve = abs(d0 + d1 - 2.0 * depth) + abs(d2 + d3 - 2.0 * depth);
  float edge = smoothstep(0.0, 1.0, curve * 2200.0) * uEdgeStrength;

  vec3 lit = colour * ao;
  lit = mix(lit, uEdge, edge);
  lit = vec3(1.0) - exp(-lit * uExposure);       // filmic-ish shoulder
  outColor = vec4(pow(lit, vec3(1.0 / 2.2)), 1.0);
}`;

const RAY_VERT = `#version 300 es
layout(location = 0) in vec3 aP0;
layout(location = 1) in vec3 aP1;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec2 aParams;   // x: side (-1/+1), y: which endpoint
uniform mat4 uMVP;
uniform vec2 uResolution;
uniform float uWidth;
out vec3 vColor;
out float vSide;
void main() {
  vec4 clip0 = uMVP * vec4(aP0, 1.0);
  vec4 clip1 = uMVP * vec4(aP1, 1.0);
  // Expand in screen space, not world space: a ray keeps the same apparent
  // thickness whether it is a metre away or across the building.
  vec2 screen0 = clip0.xy / max(abs(clip0.w), 1e-6) * uResolution;
  vec2 screen1 = clip1.xy / max(abs(clip1.w), 1e-6) * uResolution;
  vec2 dir = screen1 - screen0;
  dir = length(dir) < 1e-6 ? vec2(1.0, 0.0) : normalize(dir);
  vec2 normal = vec2(-dir.y, dir.x) * aParams.x * uWidth * 0.5;

  vec4 clip = mix(clip0, clip1, aParams.y);
  clip.xy += normal / uResolution * abs(clip.w);
  vColor = aColor;
  vSide = aParams.x;
  gl_Position = clip;
}`;

const RAY_FRAG = `#version 300 es
precision highp float;
in vec3 vColor;
in float vSide;
out vec4 outColor;
void main() {
  // Solid through the middle, fading only at the very edge. Lightening the
  // core would look like a glow on a dark scene but reads as a hollow tube on
  // a white one, which is where these are mostly seen.
  float edge = 1.0 - abs(vSide);
  outColor = vec4(vColor * (0.88 + 0.12 * smoothstep(0.3, 1.0, edge)),
                  smoothstep(0.0, 0.30, edge));
}`;

const LINE_VERT = `#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 uMVP;
out vec3 vColor;
uniform float uSize;
void main() {
  vColor = aColor;
  gl_Position = uMVP * vec4(aPos, 1.0);
  gl_PointSize = uSize;
}`;

const LINE_FRAG = `#version 300 es
precision highp float;
in vec3 vColor;
out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }`;

/** Lighting and background for each look the viewer offers. */
export const THEMES = {
  studio: {
    sky: [1.00, 0.99, 0.97], ground: [0.55, 0.56, 0.60], key: [1.05, 1.02, 0.96],
    ambient: 0.62, backTop: [0.97, 0.975, 0.985], backBottom: [0.97, 0.975, 0.985],
    edge: [0.16, 0.18, 0.22], aoStrength: 0.95, edgeStrength: 0.75, exposure: 1.25,
    plane: [0.93, 0.93, 0.95],
  },
  dark: {
    sky: [0.55, 0.62, 0.75], ground: [0.10, 0.11, 0.13], key: [0.95, 0.93, 0.88],
    ambient: 0.55, backTop: [0.10, 0.12, 0.15], backBottom: [0.05, 0.06, 0.08],
    edge: [0.02, 0.03, 0.04], aoStrength: 0.85, edgeStrength: 0.45, exposure: 1.35,
    plane: [0.14, 0.15, 0.18],
  },
};

const SHADOW_SIZE = 2048;

/** Undo the composite's tone mapping, so a colour survives it unchanged.
 *
 * The background is written straight to the screen while lit geometry goes
 * through exposure and gamma. For the ground plane to fade into the background
 * exactly, it has to start from the value that comes back out as that colour.
 */
/** What each interaction letter along a path means, and how it is drawn.
 *
 * The four hues pass the colour-vision checks against both the studio and the
 * dark background; scattering has no validated slot because the tracer cannot
 * currently produce it, so it falls back to neutral rather than to a fifth hue
 * that would collide with one of these.
 */
export const INTERACTIONS = {
  '': {label: 'line of sight', color: [0.224, 0.529, 0.898]},   // #3987e5
  R: {label: 'reflection', color: [0.851, 0.349, 0.149]},       // #d95926
  D: {label: 'diffraction', color: [0.706, 0.333, 0.690]},      // #b455b0
  T: {label: 'transmission', color: [0.098, 0.620, 0.439]},     // #199e70
  S: {label: 'scattering', color: [0.62, 0.60, 0.58]},
};

/** Map a normalised received power to a ray colour.
 *
 * Violet through red to gold. A ramp through yellow looks hotter in isolation
 * but disappears against the studio background, which is nearly white.
 */
function rayColour(t) {
  const mix = (a, b, k) => a.map((v, i) => v + (b[i] - v) * k);
  const weak = [0.38, 0.05, 0.42], mid = [0.85, 0.13, 0.12], strong = [1.0, 0.55, 0.04];
  const k = Math.min(1, Math.max(0, t));
  return k < 0.5 ? mix(weak, mid, k * 2) : mix(mid, strong, (k - 0.5) * 2);
}

function preTonemap(colour, exposure) {
  return colour.map(c => {
    const linear = Math.min(0.999, Math.pow(c, 2.2));
    return -Math.log(1 - linear) / exposure;
  });
}

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function link(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

/** Cache every active uniform location for a program. */
function uniforms(gl, prog) {
  const out = {};
  const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(prog, i).name.replace('[0]', '');
    out[name] = gl.getUniformLocation(prog, name);
  }
  return out;
}

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) throw new Error('WebGL2 is required for the 3D view');
    this.gl = gl;
    this.float = gl.getExtension('EXT_color_buffer_float');

    this.scene = link(gl, SCENE_VERT, SCENE_FRAG);
    this.depth = link(gl, DEPTH_VERT, DEPTH_FRAG);
    this.ao = link(gl, POST_VERT, AO_FRAG);
    this.blur = link(gl, POST_VERT, BLUR_FRAG);
    this.comp = link(gl, POST_VERT, COMPOSITE_FRAG);
    this.ray = link(gl, RAY_VERT, RAY_FRAG);
    this.lineProg = link(gl, LINE_VERT, LINE_FRAG);
    for (const [key, prog] of Object.entries({
      scene: this.scene, depth: this.depth, ao: this.ao, blur: this.blur,
      comp: this.comp, ray: this.ray, line: this.lineProg,
    })) this[key + 'U'] = uniforms(gl, prog);

    this.emptyVao = gl.createVertexArray();
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);   // interiors are viewed from both sides

    // camera state
    this.yaw = -0.9; this.pitch = 0.9; this.dist = 30;
    this.target = [0, 0, 1.5];
    this.clipOn = true; this.clipZ = 1.3;
    this.invertPitch = false;
    this.hidden = new Set();
    this.groups = [];
    this.markers = null;
    this.rayCount = 0; this.markerCount = 0;
    this.rayWidth = 4.5;
    this.theme = THEMES.studio;
    this.shading = true;
    this.showGround = true;
    this.lightDir = [0.45, 0.32, 0.83];

    this._buildTargets();
    this._bindInput();
    this._resize();
    new ResizeObserver(() => { this._resize(); this.draw(); }).observe(canvas);
  }

  /** Pick a look: 'studio' (light, for reading a layout) or 'dark'. */
  setTheme(name) { this.theme = THEMES[name] || THEMES.studio; this.draw(); }

  /** Turn the shading passes off, leaving flat colour. */
  setShading(on) { this.shading = !!on; this.draw(); }

  /** Show or hide the plane the model casts its shadow onto. */
  setGround(on) { this.showGround = !!on; this.draw(); }

  /** Set the on-screen thickness of propagation rays, in pixels. */
  setRayWidth(px) { this.rayWidth = Math.max(1, +px || 1); this.draw(); }

  _texture(width, height, internal, format, type, filter) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  _buildTargets() {
    const gl = this.gl;
    // Shadow map. A comparison sampler gives hardware bilinear PCF, so the
    // 3x3 tap in the shader is really 12x12 of filtering.
    this.shadowTex = this._texture(
      SHADOW_SIZE, SHADOW_SIZE, gl.DEPTH_COMPONENT24, gl.DEPTH_COMPONENT,
      gl.UNSIGNED_INT, gl.LINEAR,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);
    this.shadowFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.shadowTex, 0);
    gl.drawBuffers([gl.NONE]);
    gl.readBuffer(gl.NONE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  _sizeTargets(width, height) {
    const gl = this.gl;
    if (this.fboWidth === width && this.fboHeight === height) return;
    this.fboWidth = width; this.fboHeight = height;
    for (const name of ['colorTex', 'depthTex', 'aoTex', 'aoBlurTex']) {
      if (this[name]) gl.deleteTexture(this[name]);
    }
    for (const name of ['sceneFbo', 'aoFbo', 'aoBlurFbo']) {
      if (this[name]) gl.deleteFramebuffer(this[name]);
    }
    this.colorTex = this._texture(
      width, height, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR);
    this.depthTex = this._texture(
      width, height, gl.DEPTH_COMPONENT24, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, gl.NEAREST);
    this.sceneFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneFbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.colorTex, 0);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.depthTex, 0);

    // Occlusion runs at half resolution: it is low frequency, and this is the
    // difference between a smooth orbit and a stuttering one on a laptop GPU.
    const aw = Math.max(1, width >> 1), ah = Math.max(1, height >> 1);
    this.aoWidth = aw; this.aoHeight = ah;
    const fmt = this.float
      ? [gl.R16F, gl.RED, gl.HALF_FLOAT] : [gl.R8, gl.RED, gl.UNSIGNED_BYTE];
    this.aoTex = this._texture(aw, ah, fmt[0], fmt[1], fmt[2], gl.LINEAR);
    this.aoBlurTex = this._texture(aw, ah, fmt[0], fmt[1], fmt[2], gl.LINEAR);
    this.aoFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.aoFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.aoTex, 0);
    this.aoBlurFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.aoBlurFbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.aoBlurTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  _bindInput() {
    const c = this.canvas;
    let dragging = false, panning = false, lx = 0, ly = 0;
    let downAt = null, moved = 0;
    c.addEventListener('mousedown', e => {
      dragging = true; panning = e.shiftKey || e.button === 1;
      lx = e.clientX; ly = e.clientY; downAt = [e.clientX, e.clientY]; moved = 0;
      e.preventDefault();
    });
    window.addEventListener('mouseup', e => {
      // A press that barely moved is a pick, not an orbit.
      if (dragging && downAt && moved < 4 && this.pickHandler) {
        const hit = this._pickOnPlane(e.clientX, e.clientY, this.pickZ ?? 1.2);
        if (hit) this.pickHandler(hit);
      }
      dragging = false; downAt = null;
    });
    window.addEventListener('mousemove', e => {
      if (!dragging) return;
      const dx = e.clientX - lx, dy = e.clientY - ly;
      moved += Math.abs(dx) + Math.abs(dy);
      lx = e.clientX; ly = e.clientY;
      if (panning) {
        // Screen-right in world space. At yaw=0 the camera sits on +x looking
        // back along -x, which puts world +y on the right of the screen.
        const s = this.dist * 0.0016;
        const right = [-Math.sin(this.yaw), Math.cos(this.yaw), 0];
        this.target[0] -= right[0] * dx * s;
        this.target[1] -= right[1] * dx * s;
        this.target[2] += dy * s;
      } else {
        // Blender's turntable convention: dragging right orbits the camera to
        // the right, so the scene appears to swing left under the cursor.
        this.yaw -= dx * 0.008;
        // Vertical is a genuine preference rather than a convention, so it is a
        // setting: by default dragging down tilts the scene down with the cursor.
        const sign = this.invertPitch ? -1 : 1;
        this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + sign * dy * 0.008));
      }
      this.draw();
    });
    c.addEventListener('wheel', e => {
      this.dist *= Math.exp(e.deltaY * 0.0012);
      this.dist = Math.max(0.5, Math.min(500, this.dist));
      e.preventDefault(); this.draw();
    }, { passive: false });
    c.addEventListener('contextmenu', e => e.preventDefault());
    window.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === 'f' || e.key === 'F') this.frameAll();
      else if (e.key === '1') { this.yaw = 0; this.pitch = 0; }
      else if (e.key === '3') { this.yaw = Math.PI / 2; this.pitch = 0; }
      else if (e.key === '7') { this.yaw = 0; this.pitch = 1.55; }
      else return;
      this.draw();
    });
  }

  _resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = this.canvas.clientWidth * dpr;
    this.canvas.height = this.canvas.clientHeight * dpr;
    this._sizeTargets(this.canvas.width, this.canvas.height);
  }

  /** Upload a scene: positions is a Float32Array of xyz triples. */
  setScene(positions, groups, bbox, markers) {
    const gl = this.gl;
    this.groups = groups;
    this.bbox = bbox;
    this.markers = markers || null;

    const colors = new Float32Array(positions.length);
    for (const g of groups) {
      const [r, gg, b] = g.color;
      for (let i = g.start; i < g.start + g.count; i++) {
        colors[i * 3] = r; colors[i * 3 + 1] = gg; colors[i * 3 + 2] = b;
      }
    }
    if (this.vao) gl.deleteVertexArray(this.vao);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    this.posBuf = gl.createBuffer();
    for (const [loc, data, buf] of [[0, positions, this.posBuf], [1, colors, gl.createBuffer()]]) {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    }
    gl.bindVertexArray(null);

    // A shadow-only VAO: position is all the depth pass reads.
    if (this.depthVao) gl.deleteVertexArray(this.depthVao);
    this.depthVao = gl.createVertexArray();
    gl.bindVertexArray(this.depthVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this._buildGround(bbox);
    this.frameAll();
  }

  /** A plane under the model, so the key light's shadow has somewhere to land. */
  _buildGround(bbox) {
    const gl = this.gl;
    const [lo, hi] = bbox;
    const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2;
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1]);
    this.planeCentre = [cx, cy];
    this.planeFade = [span * 0.9, span * 2.4];
    const r = span * 6 + 20;
    const z = lo[2] - Math.max(0.01, (hi[2] - lo[2]) * 0.002);
    const quad = [
      cx - r, cy - r, z, cx + r, cy - r, z, cx + r, cy + r, z,
      cx - r, cy - r, z, cx + r, cy + r, z, cx - r, cy + r, z,
    ];
    if (this.groundVao) gl.deleteVertexArray(this.groundVao);
    this.groundVao = gl.createVertexArray();
    gl.bindVertexArray(this.groundVao);
    for (const [loc, data] of [[0, quad], [1, new Array(18).fill(1)]]) {
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    }
    gl.bindVertexArray(null);
  }

  frameAll() {
    if (!this.bbox) return;
    const [lo, hi] = this.bbox;
    this.target = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    this.dist = span * 1.6 + 2;
    this.draw();
  }

  setHidden(names) { this.hidden = new Set(names); this.draw(); }
  setInvertPitch(on) { this.invertPitch = on; }
  onPick(fn) { this.pickHandler = fn; }

  /** Draw propagation paths as ribbons of constant on-screen width.
   *
   * Two colourings answer different questions. By power, the ramp says which
   * paths carry the signal. By interaction, each segment is coloured by what
   * happened at the end of it, so a ray visibly changes colour where it
   * diffracts or passes through a wall.
   */
  setRays(paths, mode = 'power') {
    const gl = this.gl;
    this.rayMode = mode;
    // Colour by received power. The ramp runs violet through red to gold
    // rather than through yellow: a yellow ray vanishes on a white background.
    const powers = paths.map(p => p.power_db).filter(Number.isFinite);
    const hi = powers.length ? Math.max(...powers) : 0;
    const lo = powers.length ? Math.min(...powers) : -1;
    const seen = new Set();

    const p0 = [], p1 = [], cols = [], params = [];
    // Two triangles per segment: (side, endpoint) picks each of the six corners.
    const CORNERS = [[-1, 0], [1, 0], [1, 1], [-1, 0], [1, 1], [-1, 1]];
    for (const path of paths) {
      const t = (path.power_db - lo) / Math.max(hi - lo, 1e-6);
      const byPower = rayColour(Number.isFinite(t) ? t : 1);
      const kinds = path.interactions || '';
      for (let i = 0; i + 1 < path.points.length; i++) {
        let c = byPower;
        if (mode === 'interaction') {
          // Segment i ends at bounce i, so it carries that bounce's kind; the
          // final hop into the receiver keeps the last one it came from.
          const kind = kinds.length === 0 ? '' : (kinds[i] ?? kinds[kinds.length - 1]);
          const entry = INTERACTIONS[kind] || INTERACTIONS.S;
          c = entry.color;
          seen.add(kinds.length === 0 ? '' : kind);
        }
        const a = path.points[i], b = path.points[i + 1];
        for (const [side, end] of CORNERS) {
          p0.push(...a); p1.push(...b); cols.push(...c); params.push(side, end);
        }
      }
    }
    this.rayKinds = [...seen];
    this.rayCount = params.length / 2;
    if (this.rayVao) gl.deleteVertexArray(this.rayVao);
    if (!this.rayCount) { this.draw(); return; }

    this.rayVao = gl.createVertexArray();
    gl.bindVertexArray(this.rayVao);
    for (const [loc, data, size] of [[0, p0, 3], [1, p1, 3], [2, cols, 3], [3, params, 2]]) {
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    }
    gl.bindVertexArray(null);
    this.draw();
  }

  /** Draw device markers: the receiver grid and the transmitters. */
  setMarkers(points, colors, size = 3.0) {
    const gl = this.gl;
    this.markerCount = points.length / 3;
    this.markerSize = size;
    if (this.markerVao) gl.deleteVertexArray(this.markerVao);
    if (!this.markerCount) { this.draw(); return; }
    this.markerVao = this._annotationVao(points, colors);
    this.draw();
  }

  _annotationVao(points, colors) {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    for (const [loc, data] of [[0, points], [1, colors]]) {
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    }
    gl.bindVertexArray(null);
    return vao;
  }

  /** Unproject a click onto a horizontal plane, for picking a receiver. */
  _pickOnPlane(clientX, clientY, planeZ) {
    const r = this.canvas.getBoundingClientRect();
    const ndcX = ((clientX - r.left) / r.width) * 2 - 1;
    const ndcY = 1 - ((clientY - r.top) / r.height) * 2;
    const eye = this._eye();
    // Build the ray through the pixel from the camera basis.
    const fwd = [this.target[0]-eye[0], this.target[1]-eye[1], this.target[2]-eye[2]];
    const fl = Math.hypot(...fwd); const f = fwd.map(v => v / fl);
    const up = [0, 0, 1];
    const right = [f[1]*up[2]-f[2]*up[1], f[2]*up[0]-f[0]*up[2], f[0]*up[1]-f[1]*up[0]];
    const rl = Math.hypot(...right); const rN = right.map(v => v / rl);
    const upN = [rN[1]*f[2]-rN[2]*f[1], rN[2]*f[0]-rN[0]*f[2], rN[0]*f[1]-rN[1]*f[0]];
    const tanHalf = Math.tan(0.9 / 2);
    const aspect = this.canvas.width / this.canvas.height;
    const dir = [0,1,2].map(i => f[i] + rN[i]*ndcX*tanHalf*aspect + upN[i]*ndcY*tanHalf);
    if (Math.abs(dir[2]) < 1e-9) return null;
    const t = (planeZ - eye[2]) / dir[2];
    if (t < 0) return null;
    return [eye[0]+dir[0]*t, eye[1]+dir[1]*t, planeZ];
  }

  setClip(on, z) { this.clipOn = on; this.clipZ = z; this.draw(); }

  _eye() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    return [
      this.target[0] + this.dist * cp * Math.cos(this.yaw),
      this.target[1] + this.dist * cp * Math.sin(this.yaw),
      this.target[2] + this.dist * sp,
    ];
  }

  /** Fit an orthographic light frustum around the scene. */
  _lightMatrix() {
    const [lo, hi] = this.bbox;
    const centre = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2];
    const radius = 0.5 * Math.hypot(hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]) + 1e-3;
    const d = this.lightDir;
    const len = Math.hypot(...d);
    const eye = [0,1,2].map(i => centre[i] + d[i] / len * radius * 2.2);
    const up = Math.abs(d[2] / len) > 0.95 ? [0, 1, 0] : [0, 0, 1];
    const view = M4.lookAt(eye, centre, up);
    const proj = M4.ortho(-radius, radius, -radius, radius, 0.05, radius * 4.6);
    return M4.mul(proj, view);
  }

  _drawGeometry(prog, uni) {
    const gl = this.gl;
    gl.uniform1f(uni.uClipZ, this.clipZ);
    gl.uniform1f(uni.uClipOn, this.clipOn ? 1 : 0);
    for (const g of this.groups) {
      if (this.hidden.has(g.name)) continue;
      gl.drawArrays(gl.TRIANGLES, g.start, g.count);
    }
  }

  draw() {
    const gl = this.gl;
    const { width, height } = this.canvas;
    const theme = this.theme;
    if (!this.vao) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, width, height);
      gl.clearColor(theme.backBottom[0], theme.backBottom[1], theme.backBottom[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      return;
    }

    const eye = this._eye();
    const view = M4.lookAt(eye, this.target, [0, 0, 1]);
    const proj = M4.perspective(0.9, width / height, 0.05, 4000);
    const mvp = M4.mul(proj, view);
    const lightVP = this._lightMatrix();

    // 1. Shadow map.
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFbo);
    gl.viewport(0, 0, SHADOW_SIZE, SHADOW_SIZE);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.depth);
    gl.uniformMatrix4fv(this.depthU.uMVP, false, lightVP);
    gl.bindVertexArray(this.depthVao);
    this._drawGeometry(this.depth, this.depthU);

    // 2. Scene colour.
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneFbo);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.scene);
    const u = this.sceneU;
    gl.uniformMatrix4fv(u.uMVP, false, mvp);
    gl.uniformMatrix4fv(u.uLightVP, false, lightVP);
    gl.uniform3fv(u.uEye, new Float32Array(eye));
    gl.uniform3fv(u.uLightDir, new Float32Array(this.lightDir));
    gl.uniform3fv(u.uSky, new Float32Array(theme.sky));
    gl.uniform3fv(u.uGround, new Float32Array(theme.ground));
    gl.uniform3fv(u.uKey, new Float32Array(theme.key));
    gl.uniform1f(u.uAmbient, theme.ambient);
    gl.uniform1f(u.uShadowTexel, 1 / SHADOW_SIZE);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    gl.uniform1i(u.uShadow, 0);
    if (this.groundVao && this.showGround) {
      gl.uniform3f(u.uTint, theme.plane[0], theme.plane[1], theme.plane[2]);
      gl.uniform1f(u.uClipOn, 0);
      gl.uniform1f(u.uPlaneMode, 1);
      gl.uniform2fv(u.uPlaneCentre, new Float32Array(this.planeCentre));
      gl.uniform2fv(u.uPlaneFade, new Float32Array(this.planeFade));
      gl.uniform3fv(u.uFadeTo, new Float32Array(preTonemap(theme.backBottom, theme.exposure)));
      gl.bindVertexArray(this.groundVao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.uniform1f(u.uPlaneMode, 0);
    gl.uniform3f(u.uTint, 1, 1, 1);
    gl.bindVertexArray(this.vao);
    this._drawGeometry(this.scene, u);
    gl.bindVertexArray(null);

    const invProj = M4.invert(proj);
    const span = this.bbox
      ? Math.max(...[0, 1, 2].map(i => this.bbox[1][i] - this.bbox[0][i])) : 10;

    // 3. Occlusion, then blur.
    gl.bindVertexArray(this.emptyVao);
    gl.disable(gl.DEPTH_TEST);
    if (this.shading) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.aoFbo);
      gl.viewport(0, 0, this.aoWidth, this.aoHeight);
      gl.useProgram(this.ao);
      gl.uniformMatrix4fv(this.aoU.uProj, false, proj);
      gl.uniformMatrix4fv(this.aoU.uInvProj, false, invProj);
      gl.uniform1f(this.aoU.uRadius, Math.max(0.25, span * 0.035));
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.depthTex);
      gl.uniform1i(this.aoU.uDepth, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.aoBlurFbo);
      gl.useProgram(this.blur);
      gl.uniform2f(this.blurU.uTexel, 1 / this.aoWidth, 1 / this.aoHeight);
      gl.bindTexture(gl.TEXTURE_2D, this.aoTex);
      gl.uniform1i(this.blurU.uAO, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    // 4. Composite to the screen. Writing gl_FragDepth here hands the default
    // framebuffer a real depth buffer, so rays drawn next occlude correctly.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.ALWAYS);
    gl.useProgram(this.comp);
    const c = this.compU;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
    gl.uniform1i(c.uColor, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.shading ? this.aoBlurTex : this.colorTex);
    gl.uniform1i(c.uAO, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.depthTex);
    gl.uniform1i(c.uDepth, 2);
    gl.uniform2f(c.uTexel, 1 / width, 1 / height);
    gl.uniform3fv(c.uBackTop, new Float32Array(theme.backTop));
    gl.uniform3fv(c.uBackBottom, new Float32Array(theme.backBottom));
    gl.uniform3fv(c.uEdge, new Float32Array(theme.edge));
    gl.uniform1f(c.uAOStrength, this.shading ? theme.aoStrength : 0);
    gl.uniform1f(c.uEdgeStrength, this.shading ? theme.edgeStrength : 0);
    gl.uniform1f(c.uExposure, theme.exposure);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.depthFunc(gl.LESS);
    gl.bindVertexArray(null);

    // 5. Annotation, straight to the screen against the composited depth.
    if (this.rayCount) {
      gl.useProgram(this.ray);
      gl.uniformMatrix4fv(this.rayU.uMVP, false, mvp);
      gl.uniform2f(this.rayU.uResolution, width, height);
      gl.uniform1f(this.rayU.uWidth, this.rayWidth * (this.canvas.width / this.canvas.clientWidth));
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      // Rays cross each other constantly; writing depth would make the winner
      // of each crossing depend on draw order rather than on distance.
      gl.depthMask(false);
      gl.bindVertexArray(this.rayVao);
      gl.drawArrays(gl.TRIANGLES, 0, this.rayCount);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
    }
    if (this.markerCount) {
      gl.useProgram(this.lineProg);
      gl.uniformMatrix4fv(this.lineU.uMVP, false, mvp);
      gl.uniform1f(this.lineU.uSize, this.markerSize || 3.0);
      gl.bindVertexArray(this.markerVao);
      gl.drawArrays(gl.POINTS, 0, this.markerCount);
    }
    gl.bindVertexArray(null);
  }
}
