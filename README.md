# ffrwd/longpipe

Whole-frame foreground matting on the GPU, run by
[sb2702/longpipe](https://github.com/sb2702/longpipe)'s own code.
`matte` turns each frame into one alpha matte, white where the subject
is, black where the background is, blended at the edges, and everything
downstream is native ffmpeg or `ffrwd/mask_tools`.

Requires ffrwd 0.33 or later: the node reads its weights through
`ffrwd:models`, which that release's sidecar is the first to serve.

## How it runs

The node is longpipe's TypeScript and WGSL, unchanged, bundled with a
`navigator.gpu` shim over the sidecar's `wasi:webgpu`. Frames arrive as
textures on the sidecar's device, the network and its optical-flow
stabilizer run there, and the matte leaves as a texture: a decode on
the GPU, this node, and a GPU encoder move no pixels through host
memory. On a 4090 the small tier mattes 1080p at about 290 frames a
second with the decoder and encoder beside it.

The weights are longpipe's own files, mirrored at
[imbcmdth/longpipe-weights](https://huggingface.co/imbcmdth/longpipe-weights)
and pinned in the manifest by repo, revision, file and sha256.
`ffrwd install` fetches and verifies them; they are not in the archive.
The code and weights are MIT, and this package follows that license.

## Model export

- `matte(v, tier)` returns the frame's foreground as one matte the size
  of `v`, the alpha in every channel. `tier` picks the network: `xs`,
  `small`, `medium`, `large` or `xl`, smallest and fastest first; left
  out or NULL it is `small`. A yuv420p `v` is read in the range and
  matrix it declares. Every tier runs on every frame; the stabilizer
  carries the previous frame's matte into the next, so the node is not
  pure.

The tier is fixed when the node opens. A query that wants two tiers
calls `matte` twice.

## Composition

The composition layer lives in `ffrwd/mask_tools` (`blur_where`,
`spotlight`, `cutout` and the `masked` spelling they share) because it
is model-agnostic: any matte beside any stream, all native ffmpeg. This
package's recipes feed it the matte.

## Recipes

`background-blur`, `background-replace`, `greenscreen` and `matte`; run
`ffrwd list` for each one's variables, or read the header of the recipe
file. Each takes an optional `tier`. `greenscreen` writes ProRes 4444 in
mov, the picture carrying the matte as its own alpha channel, for
compositing in an editor. `matte` writes the matte as 4:2:0, which every
player opens.

```
ffrwd run ffrwd/longpipe:background-blur -v source=call.mp4 -v dest=blurred.mp4
ffrwd run ffrwd/longpipe:matte -v source=call.mp4 -v tier=xl -v dest=matte.mp4
```

## Building

The node is built in `matte-webgpu/` with Node:

```
git submodule update --init
cd matte-webgpu && npm ci && node build.mjs
```

`build.mjs` bundles longpipe's sources from the `longpipe/` submodule
with esbuild and componentizes the bundle with ComponentizeJS 0.19.3
against the wit it stages from an ffrwd-cli checkout (`FFRWD_REPO`,
default `E:/ffrwd-cli`). The output is `matte-webgpu/build/matte.wasm`,
which the lib SQL names and the archive carries. ComponentizeJS 0.22 is
selectable (`COMPONENTIZE=0.22`) but not usable: its bindings have no
dispose on imported resources and cost about thirty times as much per
host call.

A development checkout runs the recipes once the pinned weights sit
beside the module as `ffrwd install` lands them: the first pin as
`matte.onnx` (the small tier's bytes, under the name every model's
first file takes) and each other tier under its own file name.
