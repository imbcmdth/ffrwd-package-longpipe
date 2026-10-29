// The `node` export of `ffrwd:av/node@0.19.1` and the `ports` export of
// `ffrwd:gpu@0.1.0`, made the way `matte` needs them: a few string params,
// one picture on a clock input `v` held on the GPU, and one picture out on
// `out`, held there too. What `ffrwd-node` does for a Rust module, cut to
// that.
//
// The output is not named `v`: `ports.memory` names ports, and the host
// matches an input before an output, so a shared name would leave the
// output in host memory.

/** The JSON Schema of `params`, a map of name to `{ choices, fallback,
 * description }`: each an optional string, one of `choices`, `fallback`
 * when it is not given. */
function paramsSchema(params) {
  const properties = {};
  for (const [key, { choices, fallback, description }] of Object.entries(params)) {
    properties[key] = { type: 'string', enum: choices, default: fallback, description };
  }
  return JSON.stringify({ type: 'object', properties, additionalProperties: false });
}

/** `choices` as a sentence's list: `a, b or c`. */
function listed(choices) {
  return choices.length === 1 ? choices[0] : `${choices.slice(0, -1).join(', ')} or ${choices.at(-1)}`;
}

/** The params text read against `params`: every key filled, a key given as
 * null or not given at all taking its fallback. */
function readParams(name, params, text) {
  const trimmed = text.trim();
  let parsed = {};
  if (trimmed !== '') {
    try {
      parsed = JSON.parse(trimmed);
    } catch (error) {
      throw `${name} cannot read its params: ${error.message}`;
    }
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw `${name} takes its params as a JSON object`;
  }
  const known = Object.keys(params);
  const unknown = Object.keys(parsed).filter((key) => parsed[key] !== null && !known.includes(key));
  if (unknown.length > 0) {
    const takes = known.length === 0 ? 'no parameters' : `only ${listed(known)}`;
    throw `${name} takes ${takes}, and was given ${unknown.join(', ')}`;
  }
  const read = {};
  for (const [key, { choices, fallback }] of Object.entries(params)) {
    const value = parsed[key] ?? fallback;
    if (typeof value !== 'string') throw `${name} takes ${key} as a string, and was given ${JSON.stringify(value)}`;
    if (!choices.includes(value)) {
      throw `${name} takes ${key} as one of ${listed(choices)}, and was given '${value}'`;
    }
    read[key] = value;
  }
  return read;
}

function message(error) {
  if (typeof error === 'string') return error;
  if (error?.payload !== undefined) return `${error.message}: ${JSON.stringify(error.payload)}`;
  return error?.stack ? `${error.message}\n${error.stack}` : String(error);
}

/** The two exports for a node called `name` that reads pictures in
 * `pixelFormats` on `v` and writes rgba pictures `like` it on `out`.
 *
 * `params` is the node's parameters, read as `readParams` reads them. They
 * are fixed when the node opens: `setParams` after `init` takes the same
 * values and refuses a change.
 *
 * `open(v, params)` makes the instance from `v`, `{ id, width, height,
 * pixFmt, color }`, and the params read, and may be async; `frame(state,
 * frame)` runs one frame, `{ pts, index, duration }`, and emits it itself.
 */
export function gpuNode({ name, version, pixelFormats, pure, params, open, frame }) {
  let state = null;
  let opened = null;
  const schema = paramsSchema(params);

  const node = {
    describe() {
      return {
        name,
        version,
        paramsSchema: schema,
        rowsSchema: '',
        pixelFormats: [],
        sampleFormats: [],
        sampleRates: new Uint32Array(0),
        channelCounts: new Uint32Array(0),
        rowsLanguage: [],
      };
    },

    shape(text, bound) {
      readParams(name, params, text);
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

    async init(bound, latched, text) {
      try {
        const read = readParams(name, params, text);
        const v = bound.find((stream) => stream.port === 'v');
        if (v === undefined) throw `${name} reads a picture on 'v', and none is bound`;
        if (v.format?.tag !== 'video') throw `${name} reads video on 'v'`;
        const { width, height, pixFmt, color } = v.format.val;
        if (!pixelFormats.includes(pixFmt)) throw `${name} reads ${pixelFormats.join(' or ')}, opened for ${pixFmt}`;
        state = await open({ id: v.id, width, height, pixFmt, color }, read);
        opened = read;
      } catch (error) {
        throw message(error);
      }
    },

    setParams(text) {
      const read = readParams(name, params, text);
      if (opened === null) return;
      for (const key of Object.keys(params)) {
        if (read[key] !== opened[key]) {
          throw `${name} fixes its ${key} when it opens; it opened with '${opened[key]}' and was given '${read[key]}'`;
        }
      }
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
    memory(text) {
      return [
        { port: 'v', memory: 'gpu' },
        { port: 'out', memory: 'gpu' },
      ];
    },
  };

  return { node, ports };
}
