// Builds build/matte.wasm: longpipe's TypeScript (the `longpipe/` submodule,
// pinned, unchanged) bundled by esbuild with this module's shim and node,
// then componentized against the world in wit/.
//
//   node build.mjs              the component
//   MATTE_TIMING=60 node ...    the component printing a timing line every
//                               60 frames
//   LONGPIPE_WEIGHTS=<path>     the small tier's weight file (default: the
//                               spike's copy), checked against its sha256

import { componentize } from '@bytecodealliance/componentize-js';
import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { stageWit } from './tools/stage-wit.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const OUT = join(here, 'build');
const UPSTREAM = join(here, 'longpipe', 'src');

// longpipe's v0.0.5 small tier, as its manifest pins it.
const WEIGHTS = process.env.LONGPIPE_WEIGHTS ?? 'E:/projects/longpipe-spike/model_small.bin';
const WEIGHTS_SHA256 = '56430de7da6cfd6b834753f912441fbca00ad397790e308b8dc041e5c0b03394';

const TIMING = Number(process.env.MATTE_TIMING ?? 0);
const NAME = process.env.MATTE_NAME ?? (TIMING > 0 ? 'matte-timing' : 'matte');

// As jsqr's build: StarlingMonkey imports wasi:http for `fetch` unless it
// is off. Clocks stay: longpipe's renderer reads `performance.now()`.
const DISABLED = ['http', 'fetch-event', 'random'];

// longpipe's own build resolves `~/` to its src/ and loads shaders as text
// (its tsup.config.ts); the same here.
const upstream = {
  name: 'longpipe',
  setup(b) {
    b.onResolve({ filter: /^~\// }, (args) => ({ path: resolve(UPSTREAM, args.path.slice(2)) }));
    b.onResolve({ filter: /^longpipe-weights$/ }, () => ({ path: resolve(WEIGHTS), namespace: 'weights' }));
    b.onLoad({ filter: /.*/, namespace: 'weights' }, (args) => {
      const bytes = readFileSync(args.path);
      const sha = createHash('sha256').update(bytes).digest('hex');
      if (sha !== WEIGHTS_SHA256) throw new Error(`${args.path} is sha256 ${sha}, not the small tier's ${WEIGHTS_SHA256}`);
      return { contents: bytes, loader: 'binary' };
    });
  },
};

async function bundle() {
  const out = join(OUT, `${NAME}.bundle.js`);
  const result = await build({
    entryPoints: [join(here, 'src', 'matte.js')],
    outfile: out,
    bundle: true,
    format: 'esm',
    platform: 'neutral',
    target: 'es2022',
    mainFields: ['module', 'main'],
    loader: { '.wgsl': 'text' },
    external: ['wasi:*', 'ffrwd:*'],
    define: { 'globalThis.MATTE_TIMING': String(TIMING) },
    plugins: [upstream],
    legalComments: 'inline',
    metafile: true,
  });
  const inputs = Object.keys(result.metafile.inputs);
  const fromUpstream = inputs.filter((p) => p.includes('longpipe/src/')).length;
  console.log(`bundled ${inputs.length} files, ${fromUpstream} of them longpipe's`);
  return out;
}

stageWit(join(here, 'wit'));
mkdirSync(OUT, { recursive: true });
const started = Date.now();
const sourcePath = await bundle();
const { component } = await componentize({
  sourcePath,
  witPath: join(here, 'wit'),
  worldName: 'ffrwd:longpipe-webgpu/matte',
  disableFeatures: DISABLED,
});
const wasm = join(OUT, `${NAME}.wasm`);
writeFileSync(wasm, component);
console.log(`${wasm}  ${(component.length / 1024 / 1024).toFixed(2)} MiB  ${((Date.now() - started) / 1000).toFixed(1)} s`);
