// A `navigator.gpu` over `wasi:webgpu@0.0.1`, for running a WebGPU library
// written for the browser inside a ComponentizeJS component.
//
// The surface is the WebGPU calls longpipe makes (grep its `src/` for the
// list; NOTES.md in the bench directory has it), each translated to the
// wit's spelling: records become the wit's records, enums its enums, sizes
// and offsets u64 (BigInt), and resources the classes ComponentizeJS
// generates. Nothing here is a general WebGPU implementation; a call longpipe
// does not make throws and says which.
//
// Where the wit differs from the browser:
//
// - `request-adapter`, `request-device` and `map-async` are blocking
//   functions in 0.0.1. The promises here are made after the call has
//   returned, so they are already settled when the caller awaits them.
// - `on-submitted-work-done` blocks until the queue drains, then resolves.
// - Mapped ranges cross by copy: `getMappedRange` on a buffer mapped for
//   reading answers a copy of the bytes; on one mapped for writing (or at
//   creation) it answers a fresh ArrayBuffer, which `unmap` writes back
//   through `get-mapped-range-set-with-copy` before unmapping.
// - The flag constants (`GPUBufferUsage` and the rest) are WebGPU's numbers,
//   defined at load because longpipe reads them at module scope, where a
//   component cannot call its imports yet. The adapter checks them against
//   the host's own the first time one is requested.
// - There is no canvas. `canvasContext(texture)` makes a stand-in whose
//   `getCurrentTexture` answers the texture given, and `getPreferredCanvasFormat`
//   is `rgba8unorm`, the format `ffrwd:gpu/frames.emit` takes.
// - There is no ImageBitmap or VideoFrame. `TextureImage` wraps a texture
//   of the device as an image source, and `copyExternalImageToTexture` from
//   one is a texture-to-texture copy on the device.
// - The device is the sidecar's, shared with the host: `destroy` on it is
//   refused rather than passed on.

import * as wg from 'wasi:webgpu/webgpu@0.0.1';

const HOST = Symbol('wasi:webgpu handle');

/** The host handle behind a wrapper. */
export function host(wrapper) {
  const handle = wrapper?.[HOST];
  if (handle === undefined) throw new TypeError(`webgpu shim: ${describe(wrapper)} is not a WebGPU object of this shim`);
  return handle;
}

function describe(value) {
  return value?.constructor?.name ?? typeof value;
}

function unsupported(what) {
  throw new Error(`webgpu shim: ${what} is not implemented (longpipe does not call it)`);
}

const u64 = (n) => (n === undefined || n === null ? undefined : BigInt(n));
const num = (n) => (typeof n === 'bigint' ? Number(n) : n);

// ---- flag constants, WebGPU's numbers -------------------------------------

const BUFFER_USAGE = Object.freeze({
  MAP_READ: 0x1, MAP_WRITE: 0x2, COPY_SRC: 0x4, COPY_DST: 0x8, INDEX: 0x10,
  VERTEX: 0x20, UNIFORM: 0x40, STORAGE: 0x80, INDIRECT: 0x100, QUERY_RESOLVE: 0x200,
});
const TEXTURE_USAGE = Object.freeze({
  COPY_SRC: 0x1, COPY_DST: 0x2, TEXTURE_BINDING: 0x4, STORAGE_BINDING: 0x8, RENDER_ATTACHMENT: 0x10,
});
const MAP_MODE = Object.freeze({ READ: 0x1, WRITE: 0x2 });
const SHADER_STAGE = Object.freeze({ VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 });
const COLOR_WRITE = Object.freeze({ RED: 0x1, GREEN: 0x2, BLUE: 0x4, ALPHA: 0x8, ALL: 0xf });

let flagsChecked = false;

/** The host's flag values, compared once with the constants above. */
function checkFlags() {
  if (flagsChecked) return;
  const pairs = [
    [BUFFER_USAGE.MAP_READ, wg.GpuBufferUsage.mapRead()],
    [BUFFER_USAGE.MAP_WRITE, wg.GpuBufferUsage.mapWrite()],
    [BUFFER_USAGE.COPY_SRC, wg.GpuBufferUsage.copySrc()],
    [BUFFER_USAGE.COPY_DST, wg.GpuBufferUsage.copyDst()],
    [BUFFER_USAGE.INDEX, wg.GpuBufferUsage.index()],
    [BUFFER_USAGE.VERTEX, wg.GpuBufferUsage.vertex()],
    [BUFFER_USAGE.UNIFORM, wg.GpuBufferUsage.uniform()],
    [BUFFER_USAGE.STORAGE, wg.GpuBufferUsage.storage()],
    [BUFFER_USAGE.INDIRECT, wg.GpuBufferUsage.indirect()],
    [TEXTURE_USAGE.COPY_SRC, wg.GpuTextureUsage.copySrc()],
    [TEXTURE_USAGE.COPY_DST, wg.GpuTextureUsage.copyDst()],
    [TEXTURE_USAGE.TEXTURE_BINDING, wg.GpuTextureUsage.textureBinding()],
    [TEXTURE_USAGE.STORAGE_BINDING, wg.GpuTextureUsage.storageBinding()],
    [TEXTURE_USAGE.RENDER_ATTACHMENT, wg.GpuTextureUsage.renderAttachment()],
    [MAP_MODE.READ, wg.GpuMapMode.read()],
    [MAP_MODE.WRITE, wg.GpuMapMode.write()],
    [SHADER_STAGE.VERTEX, wg.GpuShaderStage.vertex()],
    [SHADER_STAGE.FRAGMENT, wg.GpuShaderStage.fragment()],
    [SHADER_STAGE.COMPUTE, wg.GpuShaderStage.compute()],
  ];
  for (const [ours, theirs] of pairs) {
    if (ours !== theirs) throw new Error(`webgpu shim: the host's flag ${theirs} is not WebGPU's ${ours}`);
  }
  flagsChecked = true;
}

// ---- enum spellings that differ ------------------------------------------

const TEXTURE_DIMENSION = { '1d': 'd1', '2d': 'd2', '3d': 'd3' };
const VIEW_DIMENSION = {
  '1d': 'd1', '2d': 'd2', '2d-array': 'd2-array', cube: 'cube', 'cube-array': 'cube-array', '3d': 'd3',
};
const DIMENSION_BACK = { d1: '1d', d2: '2d', d3: '3d' };

// WebGPU's `astc-4x4-unorm` is the wit's `astc4x4-unorm`; every other
// format is spelled the same.
const format = (f) => (f === undefined ? undefined : f.replace(/^astc-/, 'astc'));
const formatBack = (f) => f.replace(/^astc(\d)/, 'astc-$1');
const vertexFormat = (f) => (f === 'unorm10-10-10-2' ? 'unorm1010102' : f);

function enumOf(table, value, what) {
  if (value === undefined) return undefined;
  const spelled = table[value];
  if (spelled === undefined) throw new TypeError(`webgpu shim: ${what} '${value}' is not one WebGPU has`);
  return spelled;
}

// ---- dictionary conversions ----------------------------------------------

function extent(size) {
  if (Array.isArray(size) || ArrayBuffer.isView(size)) {
    return { width: size[0], height: size[1], depthOrArrayLayers: size[2] };
  }
  return { width: size.width, height: size.height, depthOrArrayLayers: size.depthOrArrayLayers };
}

function extentSizes(size) {
  const e = extent(size);
  return [e.width, e.height ?? 1, e.depthOrArrayLayers ?? 1];
}

function origin(o) {
  if (o === undefined) return undefined;
  if (Array.isArray(o)) return { x: o[0], y: o[1], z: o[2] };
  return { x: o.x, y: o.y, z: o.z };
}

function color(c) {
  if (c === undefined) return undefined;
  if (Array.isArray(c)) return { r: c[0], g: c[1], b: c[2], a: c[3] };
  return { r: c.r, g: c.g, b: c.b, a: c.a };
}

function texelCopyTexture(info) {
  return {
    texture: host(info.texture),
    mipLevel: info.mipLevel,
    origin: origin(info.origin),
    aspect: info.aspect,
  };
}

function texelCopyBuffer(info) {
  return {
    buffer: host(info.buffer),
    offset: u64(info.offset),
    bytesPerRow: info.bytesPerRow,
    rowsPerImage: info.rowsPerImage,
  };
}

/** The bytes WebGPU's `(data, dataOffset, size)` names: elements of a typed
 * array, bytes of an ArrayBuffer or DataView. */
function bytesOf(data, dataOffset = 0, size) {
  if (ArrayBuffer.isView(data) && !(data instanceof DataView)) {
    const unit = data.BYTES_PER_ELEMENT;
    const count = size ?? data.length - dataOffset;
    return new Uint8Array(data.buffer, data.byteOffset + dataOffset * unit, count * unit);
  }
  if (data instanceof DataView) {
    return new Uint8Array(data.buffer, data.byteOffset + dataOffset, size ?? data.byteLength - dataOffset);
  }
  return new Uint8Array(data, dataOffset, size ?? data.byteLength - dataOffset);
}

function layoutMode(layout) {
  if (layout === 'auto') return { tag: 'auto' };
  return { tag: 'specific', val: host(layout) };
}

function constants(table) {
  if (table === undefined) return undefined;
  const record = new wg.RecordGpuPipelineConstantValue();
  for (const [key, value] of Object.entries(table)) record.add(key, value);
  return record;
}

function programmable(stage) {
  return {
    module: host(stage.module),
    entryPoint: stage.entryPoint,
    constants: constants(stage.constants),
  };
}

function bindingResource(resource) {
  if (resource instanceof GPUSampler) return { tag: 'gpu-sampler', val: host(resource) };
  if (resource instanceof GPUTextureView) return { tag: 'gpu-texture-view', val: host(resource) };
  if (resource instanceof GPUTexture) return { tag: 'gpu-texture-view', val: host(resource.createView()) };
  if (resource instanceof GPUBuffer) return { tag: 'gpu-buffer-binding', val: { buffer: host(resource) } };
  if (resource?.buffer instanceof GPUBuffer) {
    return {
      tag: 'gpu-buffer-binding',
      val: { buffer: host(resource.buffer), offset: u64(resource.offset), size: u64(resource.size) },
    };
  }
  throw new TypeError(`webgpu shim: a bind group resource of ${describe(resource)} is not one the wit carries`);
}

/** What a bind group entry binds, as values compared by identity. */
function entryKey(entry) {
  const r = entry.resource;
  if (r?.buffer instanceof GPUBuffer) return [entry.binding, r.buffer, r.offset ?? 0, r.size];
  return [entry.binding, r];
}

function sameKey(a, b) {
  return a.length === b.length && a.every((entry, i) => entry.length === b[i].length && entry.every((v, j) => v === b[i][j]));
}

function layoutEntry(entry) {
  return {
    binding: entry.binding,
    visibility: entry.visibility,
    buffer: entry.buffer && {
      type: entry.buffer.type,
      hasDynamicOffset: entry.buffer.hasDynamicOffset,
      minBindingSize: u64(entry.buffer.minBindingSize),
    },
    sampler: entry.sampler && { type: entry.sampler.type },
    texture: entry.texture && {
      sampleType: entry.texture.sampleType,
      viewDimension: enumOf(VIEW_DIMENSION, entry.texture.viewDimension, 'view dimension'),
      multisampled: entry.texture.multisampled,
    },
    storageTexture: entry.storageTexture && {
      access: entry.storageTexture.access,
      format: format(entry.storageTexture.format),
      viewDimension: enumOf(VIEW_DIMENSION, entry.storageTexture.viewDimension, 'view dimension'),
    },
  };
  // `externalTexture` has no wit counterpart.
}

function blendComponent(c) {
  return c && { operation: c.operation, srcFactor: c.srcFactor, dstFactor: c.dstFactor };
}

function renderPipelineDescriptor(d) {
  return {
    vertex: {
      ...programmable(d.vertex),
      buffers: d.vertex.buffers?.map((b) =>
        b === null || b === undefined
          ? undefined
          : {
              arrayStride: u64(b.arrayStride),
              stepMode: b.stepMode,
              attributes: b.attributes.map((a) => ({
                format: vertexFormat(a.format),
                offset: u64(a.offset),
                shaderLocation: a.shaderLocation,
              })),
            },
      ),
    },
    primitive: d.primitive && {
      topology: d.primitive.topology,
      stripIndexFormat: d.primitive.stripIndexFormat,
      frontFace: d.primitive.frontFace,
      cullMode: d.primitive.cullMode,
      unclippedDepth: d.primitive.unclippedDepth,
    },
    depthStencil: d.depthStencil && {
      ...d.depthStencil,
      format: format(d.depthStencil.format),
    },
    multisample: d.multisample && {
      count: d.multisample.count,
      mask: d.multisample.mask,
      alphaToCoverageEnabled: d.multisample.alphaToCoverageEnabled,
    },
    fragment: d.fragment && {
      ...programmable(d.fragment),
      targets: d.fragment.targets.map((t) =>
        t === null || t === undefined
          ? undefined
          : {
              format: format(t.format),
              blend: t.blend && { color: blendComponent(t.blend.color), alpha: blendComponent(t.blend.alpha) },
              writeMask: t.writeMask,
            },
      ),
    },
    layout: layoutMode(d.layout),
    label: d.label,
  };
}

// ---- objects -------------------------------------------------------------

class Wrapper {
  constructor(handle, label) {
    Object.defineProperty(this, HOST, { value: handle });
    this.label = label ?? '';
  }
}

const FEATURE_NAMES = [
  'depth-clip-control', 'depth32float-stencil8', 'texture-compression-bc',
  'texture-compression-bc-sliced-3d', 'texture-compression-etc2', 'texture-compression-astc',
  'texture-compression-astc-sliced-3d', 'timestamp-query', 'indirect-first-instance', 'shader-f16',
  'rg11b10ufloat-renderable', 'bgra8unorm-storage', 'float32-filterable', 'float32-blendable',
  'clip-distances', 'dual-source-blending', 'subgroups',
];

/** A GPUSupportedFeatures: the names the host says it has, as a Set. */
function features(hostFeatures) {
  return new Set(FEATURE_NAMES.filter((name) => hostFeatures.has(name)));
}

const LIMIT_NAMES = [
  'maxTextureDimension1D', 'maxTextureDimension2D', 'maxTextureDimension3D', 'maxTextureArrayLayers',
  'maxBindGroups', 'maxBindGroupsPlusVertexBuffers', 'maxBindingsPerBindGroup',
  'maxDynamicUniformBuffersPerPipelineLayout', 'maxDynamicStorageBuffersPerPipelineLayout',
  'maxSampledTexturesPerShaderStage', 'maxSamplersPerShaderStage', 'maxStorageBuffersPerShaderStage',
  'maxStorageTexturesPerShaderStage', 'maxUniformBuffersPerShaderStage', 'maxUniformBufferBindingSize',
  'maxStorageBufferBindingSize', 'minUniformBufferOffsetAlignment', 'minStorageBufferOffsetAlignment',
  'maxVertexBuffers', 'maxBufferSize', 'maxVertexAttributes', 'maxVertexBufferArrayStride',
  'maxInterStageShaderVariables', 'maxColorAttachments', 'maxColorAttachmentBytesPerSample',
  'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY', 'maxComputeWorkgroupSizeZ', 'maxComputeWorkgroupsPerDimension',
];

/** A GPUSupportedLimits: every limit read once, as numbers. The wit's
 * method names are WebGPU's with the dimension suffix lower-cased. */
function limits(hostLimits) {
  const out = {};
  for (const name of LIMIT_NAMES) {
    const method = name.replace(/(\d)D$/, '$1D');
    out[name] = num(hostLimits[method]());
  }
  return Object.freeze(out);
}

function adapterInfo(info) {
  return Object.freeze({
    vendor: info.vendor(),
    architecture: info.architecture(),
    device: info.device(),
    description: info.description(),
    subgroupMinSize: info.subgroupMinSize(),
    subgroupMaxSize: info.subgroupMaxSize(),
  });
}

export class GPU {
  #gpu = null;

  #host() {
    this.#gpu ??= wg.getGpu();
    return this.#gpu;
  }

  async requestAdapter(options = {}) {
    checkFlags();
    const adapter = this.#host().requestAdapter({
      featureLevel: options.featureLevel,
      powerPreference: options.powerPreference,
      forceFallbackAdapter: options.forceFallbackAdapter,
      xrCompatible: undefined,
    });
    return adapter === undefined || adapter === null ? null : new GPUAdapter(adapter);
  }

  getPreferredCanvasFormat() {
    return 'rgba8unorm';
  }

  get wgslLanguageFeatures() {
    const hostFeatures = this.#host().wgslLanguageFeatures();
    const known = [
      'readonly_and_readwrite_storage_textures', 'packed_4x8_integer_dot_product',
      'unrestricted_pointer_parameters', 'pointer_composite_access',
    ];
    return new Set(known.filter((name) => hostFeatures.has(name)));
  }
}

export class GPUAdapter extends Wrapper {
  constructor(handle) {
    super(handle);
    this.features = features(handle.features());
    this.limits = limits(handle.limits());
    this.info = adapterInfo(handle.info());
    this.isFallbackAdapter = handle.isFallbackAdapter();
  }

  async requestDevice(descriptor = {}) {
    if (descriptor.requiredLimits && Object.keys(descriptor.requiredLimits).length > 0) {
      unsupported('requestDevice with requiredLimits');
    }
    const device = host(this).requestDevice({
      requiredFeatures: descriptor.requiredFeatures ? [...descriptor.requiredFeatures] : undefined,
      requiredLimits: undefined,
      defaultQueue: undefined,
      label: descriptor.label,
    });
    return new GPUDevice(device, descriptor.label);
  }
}

export class GPUDevice extends Wrapper {
  #queue;
  #bindGroups = new Map();

  constructor(handle, label) {
    super(handle, label);
    this.features = features(handle.features());
    this.limits = limits(handle.limits());
    this.adapterInfo = adapterInfo(handle.adapterInfo());
    this.#queue = new GPUQueue(handle.queue(), this);
    // `lost` never settles: the device is the sidecar's and outlives the
    // module.
    this.lost = new Promise(() => {});
    this.onuncapturederror = null;
  }

  get queue() {
    return this.#queue;
  }

  destroy() {
    // The sidecar's device, shared with the host and every other module.
  }

  createBuffer(d) {
    const buffer = host(this).createBuffer({
      size: u64(d.size),
      usage: d.usage,
      mappedAtCreation: d.mappedAtCreation,
      label: d.label,
    });
    return new GPUBuffer(buffer, d);
  }

  createTexture(d) {
    const size = extentSizes(d.size);
    const texture = host(this).createTexture({
      size: { width: size[0], height: size[1], depthOrArrayLayers: size[2] },
      mipLevelCount: d.mipLevelCount,
      sampleCount: d.sampleCount,
      dimension: enumOf(TEXTURE_DIMENSION, d.dimension, 'texture dimension'),
      format: format(d.format),
      usage: d.usage,
      viewFormats: d.viewFormats?.map(format),
      label: d.label,
    });
    return new GPUTexture(texture, {
      width: size[0],
      height: size[1],
      depthOrArrayLayers: size[2],
      mipLevelCount: d.mipLevelCount ?? 1,
      sampleCount: d.sampleCount ?? 1,
      dimension: d.dimension ?? '2d',
      format: d.format,
      usage: d.usage,
      label: d.label,
    });
  }

  createSampler(d = {}) {
    return new GPUSampler(
      host(this).createSampler({
        addressModeU: d.addressModeU,
        addressModeV: d.addressModeV,
        addressModeW: d.addressModeW,
        magFilter: d.magFilter,
        minFilter: d.minFilter,
        mipmapFilter: d.mipmapFilter,
        lodMinClamp: d.lodMinClamp,
        lodMaxClamp: d.lodMaxClamp,
        compare: d.compare,
        maxAnisotropy: d.maxAnisotropy,
        label: d.label,
      }),
      d.label,
    );
  }

  createBindGroupLayout(d) {
    return new GPUBindGroupLayout(
      host(this).createBindGroupLayout({ entries: d.entries.map(layoutEntry), label: d.label }),
      d.label,
    );
  }

  createPipelineLayout(d) {
    return new GPUPipelineLayout(
      host(this).createPipelineLayout({
        bindGroupLayouts: d.bindGroupLayouts.map((l) => (l ? host(l) : undefined)),
        label: d.label,
      }),
      d.label,
    );
  }

  // A bind group is immutable, so one asked for again with the same layout
  // and the same resources is the one made the first time. longpipe's input
  // op makes its bind group anew on every frame, and a handle ComponentizeJS
  // has passed as a borrow is never dropped (see `createView`).
  createBindGroup(d) {
    let made = this.#bindGroups.get(d.layout);
    if (made === undefined) this.#bindGroups.set(d.layout, (made = []));
    const key = d.entries.map(entryKey);
    const found = made.find((m) => sameKey(m.key, key));
    if (found) return found.group;
    const group = new GPUBindGroup(
      host(this).createBindGroup({
        layout: host(d.layout),
        entries: d.entries.map((e) => ({ binding: e.binding, resource: bindingResource(e.resource) })),
        label: d.label,
      }),
      d.label,
    );
    made.push({ key, group });
    return group;
  }

  createShaderModule(d) {
    return new GPUShaderModule(
      host(this).createShaderModule({ code: d.code, compilationHints: undefined, label: d.label }),
      d.label,
    );
  }

  createComputePipeline(d) {
    return new GPUComputePipeline(
      host(this).createComputePipeline({
        compute: programmable(d.compute),
        layout: layoutMode(d.layout),
        label: d.label,
      }),
      d.label,
    );
  }

  async createComputePipelineAsync(d) {
    return this.createComputePipeline(d);
  }

  createRenderPipeline(d) {
    return new GPURenderPipeline(host(this).createRenderPipeline(renderPipelineDescriptor(d)), d.label);
  }

  async createRenderPipelineAsync(d) {
    return this.createRenderPipeline(d);
  }

  createCommandEncoder(d = {}) {
    return new GPUCommandEncoder(host(this).createCommandEncoder({ label: d.label }), d.label);
  }

  pushErrorScope(filter) {
    host(this).pushErrorScope(filter);
  }

  async popErrorScope() {
    const error = host(this).popErrorScope();
    if (error === undefined || error === null) return null;
    return { message: error.message(), kind: error.kind() };
  }

  importExternalTexture() {
    unsupported('importExternalTexture (there are no VideoFrames here)');
  }

  createQuerySet() {
    unsupported('createQuerySet');
  }

  createRenderBundleEncoder() {
    unsupported('createRenderBundleEncoder');
  }
}

export class GPUBuffer extends Wrapper {
  #ranges = [];
  #mode = 0;

  constructor(handle, d) {
    super(handle, d.label);
    this.size = d.size;
    this.usage = d.usage;
    this.mapState = d.mappedAtCreation ? 'mapped' : 'unmapped';
    if (d.mappedAtCreation) this.#mode = MAP_MODE.WRITE;
  }

  async mapAsync(mode, offset, size) {
    host(this).mapAsync(mode, u64(offset), u64(size));
    this.#mode = mode;
    this.mapState = 'mapped';
  }

  getMappedRange(offset = 0, size) {
    if (this.mapState !== 'mapped') throw new Error('webgpu shim: getMappedRange on a buffer that is not mapped');
    const length = size ?? this.size - offset;
    if (this.#mode & MAP_MODE.WRITE) {
      const bytes = new ArrayBuffer(length);
      this.#ranges.push({ offset, length, bytes });
      return bytes;
    }
    const copy = host(this).getMappedRangeGetWithCopy(u64(offset), u64(length));
    return copy.buffer.byteLength === copy.byteLength && copy.byteOffset === 0
      ? copy.buffer
      : copy.slice().buffer;
  }

  unmap() {
    for (const { offset, length, bytes } of this.#ranges) {
      host(this).getMappedRangeSetWithCopy(new Uint8Array(bytes), u64(offset), u64(length));
    }
    this.#ranges = [];
    this.#mode = 0;
    host(this).unmap();
    this.mapState = 'unmapped';
  }

  destroy() {
    host(this).destroy();
  }
}

export class GPUTexture extends Wrapper {
  #view = null;

  constructor(handle, d) {
    super(handle, d.label);
    Object.assign(this, {
      width: d.width,
      height: d.height,
      depthOrArrayLayers: d.depthOrArrayLayers,
      mipLevelCount: d.mipLevelCount,
      sampleCount: d.sampleCount,
      dimension: d.dimension,
      format: d.format,
      usage: d.usage,
    });
  }

  /** A texture the host made (`frames.planes`), described by asking it. */
  static fromHost(handle) {
    return new GPUTexture(handle, {
      width: handle.width(),
      height: handle.height(),
      depthOrArrayLayers: handle.depthOrArrayLayers(),
      mipLevelCount: handle.mipLevelCount(),
      sampleCount: handle.sampleCount(),
      dimension: DIMENSION_BACK[handle.dimension()],
      format: formatBack(handle.format()),
      usage: handle.usage(),
    });
  }

  createView(d) {
    // The default view is made once: longpipe asks for one per frame, and a
    // handle ComponentizeJS has passed as a borrow is never dropped.
    if (d === undefined) {
      this.#view ??= new GPUTextureView(host(this).createView(undefined));
      return this.#view;
    }
    return new GPUTextureView(
      host(this).createView({
        format: format(d.format),
        dimension: enumOf(VIEW_DIMENSION, d.dimension, 'view dimension'),
        usage: d.usage,
        aspect: d.aspect,
        baseMipLevel: d.baseMipLevel,
        mipLevelCount: d.mipLevelCount,
        baseArrayLayer: d.baseArrayLayer,
        arrayLayerCount: d.arrayLayerCount,
        label: d.label,
      }),
      d.label,
    );
  }

  destroy() {
    host(this).destroy();
  }
}

export class GPUTextureView extends Wrapper {}
export class GPUSampler extends Wrapper {}
export class GPUBindGroupLayout extends Wrapper {}
export class GPUBindGroup extends Wrapper {}
export class GPUPipelineLayout extends Wrapper {}
export class GPUCommandBuffer extends Wrapper {}

export class GPUShaderModule extends Wrapper {
  async getCompilationInfo() {
    const messages = host(this)
      .getCompilationInfo()
      .messages()
      .map((m) => ({
        message: m.message(),
        type: m.type(),
        lineNum: num(m.lineNum()),
        linePos: num(m.linePos()),
        offset: num(m.offset()),
        length: num(m.length()),
      }));
    return { messages };
  }
}

class Pipeline extends Wrapper {
  #layouts = [];

  // Made once per index: longpipe asks again for every bind group it makes,
  // and the layout is the key `createBindGroup` finds its groups under.
  getBindGroupLayout(index) {
    this.#layouts[index] ??= new GPUBindGroupLayout(host(this).getBindGroupLayout(index));
    return this.#layouts[index];
  }
}

export class GPUComputePipeline extends Pipeline {}
export class GPURenderPipeline extends Pipeline {}

export class GPUQueue extends Wrapper {
  #device;

  constructor(handle, device) {
    super(handle);
    this.#device = device;
  }

  submit(buffers) {
    host(this).submit(buffers.map(host));
  }

  async onSubmittedWorkDone() {
    host(this).onSubmittedWorkDone();
  }

  writeBuffer(buffer, bufferOffset, data, dataOffset, size) {
    host(this).writeBufferWithCopy(host(buffer), u64(bufferOffset), bytesOf(data, dataOffset, size), undefined, undefined);
  }

  writeTexture(destination, data, layout, size) {
    const e = extentSizes(size);
    host(this).writeTextureWithCopy(
      texelCopyTexture(destination),
      bytesOf(data),
      { offset: u64(layout.offset), bytesPerRow: layout.bytesPerRow, rowsPerImage: layout.rowsPerImage },
      { width: e[0], height: e[1], depthOrArrayLayers: e[2] },
    );
  }

  /** From a `TextureImage` only: a copy between two textures of the device.
   * `flipY`, colour space and premultiplication are refused, not ignored. */
  copyExternalImageToTexture(source, destination, copySize) {
    const image = source.source;
    if (!(image instanceof TextureImage)) {
      unsupported(`copyExternalImageToTexture from ${describe(image)} (only a TextureImage is carried)`);
    }
    if (source.flipY) unsupported('copyExternalImageToTexture with flipY');
    if (destination.premultipliedAlpha) unsupported('copyExternalImageToTexture with premultipliedAlpha');
    const encoder = this.#device.createCommandEncoder();
    encoder.copyTextureToTexture(
      { texture: image.texture, origin: source.origin },
      destination,
      copySize,
    );
    this.submit([encoder.finish()]);
  }
}

export class GPUCommandEncoder extends Wrapper {
  beginComputePass(d = {}) {
    if (d.timestampWrites) unsupported('compute pass timestampWrites');
    return new GPUComputePassEncoder(host(this).beginComputePass({ timestampWrites: undefined, label: d.label }));
  }

  beginRenderPass(d) {
    if (d.timestampWrites) unsupported('render pass timestampWrites');
    if (d.occlusionQuerySet) unsupported('render pass occlusionQuerySet');
    const ds = d.depthStencilAttachment;
    return new GPURenderPassEncoder(
      host(this).beginRenderPass({
        colorAttachments: d.colorAttachments.map((a) =>
          a === null || a === undefined
            ? undefined
            : {
                view: host(a.view instanceof GPUTexture ? a.view.createView() : a.view),
                depthSlice: a.depthSlice,
                resolveTarget: a.resolveTarget ? host(a.resolveTarget) : undefined,
                clearValue: color(a.clearValue),
                loadOp: a.loadOp,
                storeOp: a.storeOp,
              },
        ),
        depthStencilAttachment: ds && { ...ds, view: host(ds.view) },
        occlusionQuerySet: undefined,
        timestampWrites: undefined,
        maxDrawCount: u64(d.maxDrawCount),
        label: d.label,
      }),
    );
  }

  copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
    // The four-argument form (source, destination, size) is newer WebGPU.
    if (destination === undefined || typeof sourceOffset !== 'number') {
      unsupported('copyBufferToBuffer without offsets');
    }
    host(this).copyBufferToBuffer(host(source), u64(sourceOffset), host(destination), u64(destinationOffset), u64(size));
  }

  copyBufferToTexture(source, destination, size) {
    const e = extentSizes(size);
    host(this).copyBufferToTexture(texelCopyBuffer(source), texelCopyTexture(destination), {
      width: e[0], height: e[1], depthOrArrayLayers: e[2],
    });
  }

  copyTextureToBuffer(source, destination, size) {
    const e = extentSizes(size);
    host(this).copyTextureToBuffer(texelCopyTexture(source), texelCopyBuffer(destination), {
      width: e[0], height: e[1], depthOrArrayLayers: e[2],
    });
  }

  copyTextureToTexture(source, destination, size) {
    const e = extentSizes(size);
    host(this).copyTextureToTexture(texelCopyTexture(source), texelCopyTexture(destination), {
      width: e[0], height: e[1], depthOrArrayLayers: e[2],
    });
  }

  clearBuffer(buffer, offset, size) {
    host(this).clearBuffer(host(buffer), u64(offset), u64(size));
  }

  finish(d = {}) {
    return new GPUCommandBuffer(host(this).finish({ label: d.label }), d.label);
  }
}

function setBindGroup(pass, index, group, offsets) {
  const dynamic = offsets && offsets.length > 0 ? Uint32Array.from(offsets) : undefined;
  host(pass).setBindGroup(index, group ? host(group) : undefined, dynamic, undefined, undefined);
}

export class GPUComputePassEncoder extends Wrapper {
  setPipeline(pipeline) {
    host(this).setPipeline(host(pipeline));
  }

  setBindGroup(index, group, offsets) {
    setBindGroup(this, index, group, offsets);
  }

  dispatchWorkgroups(x, y, z) {
    host(this).dispatchWorkgroups(x, y, z);
  }

  dispatchWorkgroupsIndirect(buffer, offset) {
    host(this).dispatchWorkgroupsIndirect(host(buffer), u64(offset));
  }

  end() {
    host(this).end();
  }
}

export class GPURenderPassEncoder extends Wrapper {
  setPipeline(pipeline) {
    host(this).setPipeline(host(pipeline));
  }

  setBindGroup(index, group, offsets) {
    setBindGroup(this, index, group, offsets);
  }

  setVertexBuffer(slot, buffer, offset, size) {
    host(this).setVertexBuffer(slot, buffer ? host(buffer) : undefined, u64(offset), u64(size));
  }

  setIndexBuffer(buffer, indexFormat, offset, size) {
    host(this).setIndexBuffer(host(buffer), indexFormat, u64(offset), u64(size));
  }

  setViewport(x, y, width, height, minDepth, maxDepth) {
    host(this).setViewport(x, y, width, height, minDepth, maxDepth);
  }

  setScissorRect(x, y, width, height) {
    host(this).setScissorRect(x, y, width, height);
  }

  setBlendConstant(c) {
    host(this).setBlendConstant(color(c));
  }

  draw(vertexCount, instanceCount, firstVertex, firstInstance) {
    host(this).draw(vertexCount, instanceCount, firstVertex, firstInstance);
  }

  drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
    host(this).drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
  }

  end() {
    host(this).end();
  }
}

// ---- stand-ins for the browser's image and canvas ------------------------

/** A texture of the device in the place of an ImageBitmap: `width` and
 * `height` as an image has them, and the texture to copy from. */
export class TextureImage {
  constructor(texture) {
    this.texture = texture;
    this.width = texture.width;
    this.height = texture.height;
  }

  close() {}
}

/** A stand-in for a canvas `width` by `height` whose webgpu context draws
 * into textures the caller supplies: `context.texture` is what the next
 * `getCurrentTexture` answers. */
export class TextureCanvas {
  constructor(width, height) {
    this.width = width;
    this.height = height;
    this.context = {
      canvas: this,
      texture: null,
      configuration: null,
      configure(configuration) {
        this.configuration = configuration;
      },
      unconfigure() {
        this.configuration = null;
      },
      getConfiguration() {
        return this.configuration;
      },
      getCurrentTexture() {
        if (this.texture === null) throw new Error('webgpu shim: the canvas has no texture to draw into');
        return this.texture;
      },
    };
  }

  getContext(kind) {
    return kind === 'webgpu' ? this.context : null;
  }
}

// ---- globals -------------------------------------------------------------

const gpu = new GPU();

/** `navigator.gpu` and the WebGPU globals longpipe reads, set before its
 * modules load. */
export function install() {
  if (globalThis.navigator === undefined) {
    globalThis.navigator = {};
  }
  Object.defineProperty(globalThis.navigator, 'gpu', { value: gpu, configurable: true });
  Object.assign(globalThis, {
    GPUBufferUsage: BUFFER_USAGE,
    GPUTextureUsage: TEXTURE_USAGE,
    GPUMapMode: MAP_MODE,
    GPUShaderStage: SHADER_STAGE,
    GPUColorWrite: COLOR_WRITE,
  });
}

install();
