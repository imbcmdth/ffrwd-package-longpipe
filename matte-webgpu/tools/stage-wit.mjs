// Stages the wit this component is built against from an ffrwd-cli
// checkout (FFRWD_REPO, default E:/ffrwd-cli): the node world (`ffrwd:av`
// 0.19.1), the GPU companion (`ffrwd:gpu` 0.1.0) and its dependencies,
// `wasi:webgpu` 0.0.1 among them.
//
// One edit is made on the way. ComponentizeJS 0.22's bindings generator
// (js-component-bindgen 1.19.0) panics on a list whose element is a type
// alias of a number ("unsupported type ... for canonical list lower"), and
// webgpu.wit has two: `list<gpu-buffer-dynamic-offset>` (u32) and
// `list<gpu-pipeline-constant-value>` (f64). An alias is not part of a
// component type, so spelling them `list<u32>` and `list<f64>` changes
// nothing the host links against.

import { cpSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const repo = process.env.FFRWD_REPO ?? 'E:/ffrwd-cli';
const sidecar = join(repo, 'sidecar');

export function stageWit(dir = 'wit') {
  mkdirSync(join(dir, 'deps', 'ffrwd-av'), { recursive: true });
  mkdirSync(join(dir, 'deps', 'ffrwd-gpu'), { recursive: true });
  cpSync(join(sidecar, 'wit', 'av.wit'), join(dir, 'deps', 'ffrwd-av', 'av.wit'));
  cpSync(join(sidecar, 'wit-gpu', 'gpu.wit'), join(dir, 'deps', 'ffrwd-gpu', 'gpu.wit'));
  cpSync(join(sidecar, 'wit-gpu', 'deps'), join(dir, 'deps'), { recursive: true });
  const webgpu = join(dir, 'deps', 'webgpu', 'webgpu.wit');
  const text = readFileSync(webgpu, 'utf8')
    .replaceAll('list<gpu-buffer-dynamic-offset>', 'list<u32>')
    .replaceAll('list<gpu-pipeline-constant-value>', 'list<f64>');
  writeFileSync(webgpu, text);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) stageWit();
