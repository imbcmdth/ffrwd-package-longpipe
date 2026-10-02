# ffrwd/longpipe

Lightninght fast, whole-frame foreground matting, hosted in wasm.
`matte` turns each frame into one grayscale alpha - white where the subject is, black
where the background is, blended at the edges - and everything
downstream is native ffmpeg.

Requires ffrwd 0.29.

The model is [sb2702/longpipe](https://github.com/sb2702/longpipe)'s
work: the small matting net, MIT-licensed code and weights, converted
to ONNX at [imbcmdth/longpipe-onnx](https://huggingface.co/imbcmdth/longpipe-onnx).
This package follows the weights' license.

The weights are not in the archive: the manifest pins them - exact
repo, revision, file and sha256 - and `ffrwd install` fetches and
verifies them.

## Model export

- `matte(v)` returns the frame's foreground as one grayscale matte:
  the size of `v`, in `gray`, one byte a pixel. The net is whole-frame
  matting: no classes, no threshold, no parameters - the output IS the
  foreground. A yuv420p `v` is read in the range and matrix it
  declares.

## Composition

The composition layer lives in `ffrwd/mask_tools` - `blur_where`,
`spotlight`, `cutout` and the `masked` spelling they share - because
it is model-agnostic: any grayscale matte beside any stream, all
native ffmpeg. This package's recipes feed it the matte.

## Recipes

`background-blur`, `background-replace`, `greenscreen`, `matte` - run
`ffrwd list` for each one's variables, or read the header of the
recipe file. `greenscreen` writes ProRes 4444 in mov, the picture
carrying the matte as its own alpha channel, for compositing in an
editor. `matte` writes the gray matte as 4:2:0, which every player
opens, where libx264 would otherwise code it as 4:0:0.

```
ffrwd run ffrwd/longpipe:background-blur -v source=call.mp4 -v dest=blurred.mp4
```

## Building

```
cargo build --target wasm32-wasip2 --release
```

`matte` is a node written with
[`ffrwd-node`](https://github.com/imbcmdth/ffrwd-node), which carries
the world it is built against, so there is nothing to install before a
build.

The weights are fetched at `ffrwd install` time by whoever installs
the published package; a development checkout runs the recipes only
after placing the pinned model beside the built module.
