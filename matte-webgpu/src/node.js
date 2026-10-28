// The `node` export of `ffrwd:av/node@0.19.1` and the `ports` export of
// `ffrwd:gpu@0.1.0`, made the way `matte` needs them: no params, one picture
// on a clock input `v` held on the GPU, and one picture out on `out`, held
// there too. What `ffrwd-node` does for a Rust module, cut to that.
//
// The output is not named `v`: `ports.memory` names ports, and the host
// matches an input before an output, so a shared name would leave the
// output in host memory.

const NO_PARAMS = '{"type":"object","properties":{},"additionalProperties":false}';

function readParams(name, params) {
  const text = params.trim();
  if (text === '') return;
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw `${name} cannot read its params: ${error.message}`;
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw `${name} takes its params as a JSON object`;
  }
  const named = Object.keys(parsed).filter((key) => parsed[key] !== null);
  if (named.length > 0) throw `${name} takes no parameters, and was given ${named.join(', ')}`;
}

function message(error) {
  if (typeof error === 'string') return error;
  if (error?.payload !== undefined) return `${error.message}: ${JSON.stringify(error.payload)}`;
  return error?.stack ? `${error.message}\n${error.stack}` : String(error);
}

/** The two exports for a node called `name` that reads pictures in
 * `pixelFormats` on `v` and writes rgba pictures `like` it on `out`.
 *
 * `open(v)` makes the instance from `v`, `{ id, width, height, pixFmt,
 * color }`, and may be async; `frame(state, frame)` runs one frame, `{ pts,
 * index, duration }`, and emits it itself.
 */
export function gpuNode({ name, version, pixelFormats, pure, open, frame }) {
  let state = null;

  const node = {
    describe() {
      return {
        name,
        version,
        paramsSchema: NO_PARAMS,
        rowsSchema: '',
        pixelFormats: [],
        sampleFormats: [],
        sampleRates: new Uint32Array(0),
        channelCounts: new Uint32Array(0),
        rowsLanguage: [],
      };
    },

    shape(params, bound) {
      readParams(name, params);
      for (const { input } of bound) {
        if (input !== 'v') throw `${name} has no input '${input}'; it reads 'v'`;
      }
      return {
        inputs: [
          {
            name: 'v',
            kind: 'video',
            required: true,
            many: false,
            pairing: { tag: 'lockstep' },
            rows: 'ignore',
            window: 1,
            stride: 1,
            accepts: {
              pixelFormats,
              sampleFormats: [],
              sampleRates: new Uint32Array(0),
              channelCounts: new Uint32Array(0),
              codecs: [],
              wants: 'all',
              like: undefined,
            },
            schema: undefined,
          },
        ],
        outputs: [
          {
            name: 'out',
            kind: 'video',
            format: { tag: 'like', val: { port: 'v', pixelFormat: 'rgba', sampleFormat: undefined } },
            timeBase: undefined,
            latency: 0,
            schema: undefined,
            row: undefined,
          },
        ],
        clock: { tag: 'input', val: 'v' },
        pure,
        oneToOne: true,
        bounded: true,
        relation: [],
      };
    },

    async init(bound, latched, params) {
      try {
        readParams(name, params);
        const v = bound.find((stream) => stream.port === 'v');
        if (v === undefined) throw `${name} reads a picture on 'v', and none is bound`;
        if (v.format?.tag !== 'video') throw `${name} reads video on 'v'`;
        const { width, height, pixFmt, color } = v.format.val;
        if (!pixelFormats.includes(pixFmt)) throw `${name} reads ${pixelFormats.join(' or ')}, opened for ${pixFmt}`;
        state = await open({ id: v.id, width, height, pixFmt, color });
      } catch (error) {
        throw message(error);
      }
    },

    setParams(params) {
      readParams(name, params);
    },

    process(tick) {
      if (state === null) throw `${name} was called before init`;
      try {
        for (const f of tick.frames(state.v.id)) frame(state, f);
      } catch (error) {
        throw message(error);
      }
      return { items: [], rows: [], finished: false };
    },
  };

  const ports = {
    memory(params) {
      return [
        { port: 'v', memory: 'gpu' },
        { port: 'out', memory: 'gpu' },
      ];
    },
  };

  return { node, ports };
}
