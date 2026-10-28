// `matte(v)` on the sidecar's device, run by longpipe's own code: the
// frame's foreground as a gray matte at `v`'s size, white where the subject
// is, black where the background is.
//
// longpipe's `Renderer` (src/pipeline/worker/renderer.ts) is driven as its
// worker drives it, with the browser left out: the `small` tier from
// longpipe's own weight file, its optical-flow stabilizer, and a background
// colour so that every frame runs the model. A frame reaches it as the
// texture `frames.planes` hands over (an rgba plane as it is; yuv420p planes
// made rgba by a pass of this module's own), and the matte it composites
// with, its upscaled alpha at the frame's size, is written into an rgba
// texture with the alpha in every channel and handed to the host. Nothing
// leaves the device.
//
// The renderer composites the frame over the colour into a canvas texture
// on every frame, as it would for display. That pass is the price of
// driving the renderer unchanged: it writes a texture this module owns and
// never emits.

// First: `navigator.gpu` and the WebGPU globals must exist before longpipe's
// modules are evaluated, since one of them reads `GPUBufferUsage` at load.
import { calls, host, profile, GPUTexture, TextureCanvas, TextureImage } from './webgpu.js';

import { emit, planes } from 'ffrwd:gpu/frames@0.1.0';
import { WebGPUBackend } from '../longpipe/src/model/backends/webgpu/index.ts';
import { Renderer } from '../longpipe/src/pipeline/worker/renderer.ts';
import { TIER_CONFIG } from '../longpipe/src/model/tier_config.ts';
import weightBytes from 'longpipe-weights';

import { gpuNode } from './node.js';

const VERSION = '0.3.0-webgpu.0';

// The tier and how it runs: every frame (no skipping, as longpipe's large
// and xl presets run), in f32, the weight file's own precision.
const TIER = 'small';
const PRESET = {
  model: TIER,
  dtype: 'f32',
  resolution: TIER_CONFIG[TIER].canvasRes,
  skipFrames: 0,
};

// How often the timing line is printed, in frames; 0 is never. Set at build
// time (`MATTE_TIMING`), since the node takes no params.
const TIMING_EVERY = Number(globalThis.MATTE_TIMING ?? 0);
if (TIMING_EVERY > 0) profile();

// The formats `v` is taken in, preferred first. rgba first, as every
// module that feeds a model lists it: the sidecar makes a decoder's nv12
// rgba on the device. A build with `MATTE_YUV=1` lists yuv420p first, and
// this module's own pass makes it rgba.
const PIXEL_FORMATS = globalThis.MATTE_YUV ? ['yuv420p', 'rgba'] : ['rgba', 'yuv420p'];

const MATTE_WGSL = /* wgsl */ `
// The renderer's alpha, one vec4 a pixel at the frame's size with the alpha
// in .r, written as rgba8unorm with the alpha in every channel.
@group(0) @binding(0) var<storage, read> alpha: array<vec4<f32>>;
@group(0) @binding(1) var matte: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(matte);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let a = clamp(alpha[id.y * size.x + id.x].r, 0.0, 1.0);
  textureStore(matte, vec2<i32>(id.xy), vec4<f32>(a, a, a, a));
}
`;

const YUV_WGSL = /* wgsl */ `
// yuv420p planes to rgba8unorm, in the range and matrix the stream declares
// (an unknown range is tv and an unknown matrix BT.601, as ffrwd-frame and
// swscale read them). Each chroma sample covers the 2x2 luma block it was
// subsampled from, read nearest, as swscale's default does.
struct Colour {
  // y' = (y - offset) * scale; then r = y' + rv*v', g = y' - gu*u' - gv*v',
  // b = y' + bu*u', with u' and v' centred on zero and scaled by cscale.
  offset: f32, scale: f32, cscale: f32, rv: f32,
  gu: f32, gv: f32, bu: f32, _pad: f32,
};
@group(0) @binding(0) var y_plane: texture_2d<f32>;
@group(0) @binding(1) var u_plane: texture_2d<f32>;
@group(0) @binding(2) var v_plane: texture_2d<f32>;
@group(0) @binding(3) var rgba: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> c: Colour;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(rgba);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = vec2<i32>(id.xy);
  let q = p / 2;
  let y = (textureLoad(y_plane, p, 0).r * 255.0 - c.offset) * c.scale;
  let u = (textureLoad(u_plane, q, 0).r * 255.0 - 128.0) * c.cscale;
  let v = (textureLoad(v_plane, q, 0).r * 255.0 - 128.0) * c.cscale;
  let rgb = vec3<f32>(y + c.rv * v, y - c.gu * u - c.gv * v, y + c.bu * u) / 255.0;
  textureStore(rgba, p, vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
`;

/** The YUV-to-RGB constants for a stream's colour, ffrwd-frame's rules. */
function colourOf(color) {
  const range = color?.range ?? 'unknown';
  const space = color?.space ?? 'unknown';
  const tv = ['tv', 'mpeg', 'limited', 'unknown'].includes(range);
  const pc = ['pc', 'jpeg', 'full'].includes(range);
  if (!tv && !pc) throw `matte cannot read the yuv range '${range}'; ask for rgba`;
  let kr, kb;
  if (['bt470bg', 'smpte170m', 'bt601', 'bt470', 'unknown', 'reserved'].includes(space)) [kr, kb] = [0.299, 0.114];
  else if (space === 'bt709') [kr, kb] = [0.2126, 0.0722];
  else throw `matte cannot read the yuv matrix '${space}'; ask for rgba`;
  const kg = 1 - kr - kb;
  return new Float32Array([
    tv ? 16 : 0,
    tv ? 255 / 219 : 1,
    tv ? 255 / 224 : 1,
    2 * (1 - kr),
    (2 * kb * (1 - kb)) / kg,
    (2 * kr * (1 - kr)) / kg,
    2 * (1 - kb),
    0,
  ]);
}

const now = () => performance.now();

/** The weight file as its own ArrayBuffer, which longpipe's loader views. */
function weightBuffer() {
  return weightBytes.buffer.slice(weightBytes.byteOffset, weightBytes.byteOffset + weightBytes.byteLength);
}

async function open(v) {
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) throw 'matte: the sidecar answered no GPU adapter';
  const device = await adapter.requestDevice();
  const { width, height } = v;

  const canvas = new TextureCanvas(width, height);
  const backend = await WebGPUBackend.create({ canvas, device, dtype: PRESET.dtype });
  const renderer = new Renderer({
    backend,
    backendKind: 'webgpu',
    canvas,
    background: { kind: 'color', rgb: [0, 0, 0] },
    enabled: true,
    topology: { input: 'ffrwd', output: 'ffrwd' },
  });
  renderer.setPreset(PRESET, weightBuffer());

  // The composite's target, made once and never emitted.
  canvas.context.texture = device.createTexture({
    size: [width, height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // The renderer's upscaled alpha, at the frame's size: what its compositor
  // reads, and what the matte is written from.
  const alpha = renderer.renderOp.upscaler.output;
  if (alpha.h !== height || alpha.w !== width) {
    throw `matte: longpipe's alpha came out ${alpha.w}x${alpha.h} for a ${width}x${height} frame`;
  }

  const matte = device.createTexture({
    size: [width, height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const mattePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: MATTE_WGSL }), entryPoint: 'main' },
  });
  const matteGroup = device.createBindGroup({
    layout: mattePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: alpha.buffer } },
      { binding: 1, resource: matte.createView() },
    ],
  });

  let yuv = null;
  if (v.pixFmt === 'yuv420p') {
    const chroma = [Math.ceil(width / 2), Math.ceil(height / 2)];
    const plane = (size) =>
      device.createTexture({
        size,
        format: 'r8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
    const planesHeld = [plane([width, height]), plane(chroma), plane(chroma)];
    const rgba = device.createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const colour = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(colour, 0, colourOf(v.color));
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: YUV_WGSL }), entryPoint: 'main' },
    });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        ...planesHeld.map((t, i) => ({ binding: i, resource: t.createView() })),
        { binding: 3, resource: rgba.createView() },
        { binding: 4, resource: { buffer: colour } },
      ],
    });
    yuv = { planes: planesHeld, rgba, image: new TextureImage(rgba), pipeline, group };
  }

  return {
    v,
    device,
    renderer,
    matte,
    mattePipeline,
    matteGroup,
    yuv,
    dispatch: [Math.ceil(width / 8), Math.ceil(height / 8)],
    timing: { frames: 0, js: 0, sampled: 0, drainJs: 0, drainWait: 0 },
  };
}

/** The frame's picture as an image longpipe's input op copies from. */
function picture(state, index) {
  const handed = planes(state.v.id, index).map((t) => GPUTexture.fromHost(t));
  if (state.yuv === null) return new TextureImage(handed[0]);
  const { device, yuv } = state;
  const encoder = device.createCommandEncoder();
  handed.forEach((source, i) => {
    encoder.copyTextureToTexture({ texture: source }, { texture: yuv.planes[i] }, [source.width, source.height]);
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(yuv.pipeline);
  pass.setBindGroup(0, yuv.group);
  pass.dispatchWorkgroups(...state.dispatch);
  pass.end();
  device.queue.submit([encoder.finish()]);
  return yuv.image;
}

function frame(state, f) {
  const { device, timing } = state;
  const sample = TIMING_EVERY > 0 && timing.frames % TIMING_EVERY === TIMING_EVERY - 1;
  if (sample) device.queue.drain();
  const start = now();

  state.renderer.process(picture(state, f.index));

  const out = device.createTexture({
    size: [state.v.width, state.v.height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.mattePipeline);
  pass.setBindGroup(0, state.matteGroup);
  pass.dispatchWorkgroups(...state.dispatch);
  pass.end();
  encoder.copyTextureToTexture({ texture: state.matte }, { texture: out }, [state.v.width, state.v.height]);
  device.queue.submit([encoder.finish()]);
  // The frame's commands go to the host as one batch, before the texture
  // they write is handed over.
  device.queue.flush();
  emit('out', f.pts, f.duration, host(out));

  const js = now() - start;
  timing.frames += 1;
  timing.js += js;
  if (sample) {
    const before = now();
    device.queue.drain();
    timing.sampled += 1;
    timing.drainJs += js;
    timing.drainWait += now() - before;
    if (timing.frames % (TIMING_EVERY * 8) === 0) {
      const rows = [...calls].filter(([, e]) => e.calls > 0).sort((a, b) => b[1].ms - a[1].ms);
      for (const [name, e] of rows.slice(0, 16)) {
        console.error(
          `[matte-webgpu]   ${name}: ${(e.calls / timing.frames).toFixed(1)} calls/frame, ` +
            `${(e.ms / timing.frames).toFixed(3)} ms/frame, ${((1000 * e.ms) / e.calls).toFixed(1)} us/call`,
        );
      }
    }
    console.error(
      `[matte-webgpu] ${timing.frames} frames: js ${(timing.js / timing.frames).toFixed(3)} ms/frame; ` +
        `on a drained queue js ${(timing.drainJs / timing.sampled).toFixed(3)} ms then the gpu's tail ` +
        `${(timing.drainWait / timing.sampled).toFixed(3)} ms (${timing.sampled} samples)`,
    );
  }
}

export const { node, ports } = gpuNode({
  name: 'matte',
  version: VERSION,
  pixelFormats: PIXEL_FORMATS,
  // The stabilizer carries the previous frame's matte into this one.
  pure: false,
  open,
  frame,
});
