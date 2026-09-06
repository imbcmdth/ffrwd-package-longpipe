# ffrwd/longpipe

Lightninght fast, whole-frame foreground matting, hosted in wasm.
`matte` turns each frame into one grayscale alpha - white where the subject is, black
where the background is, blended at the edges - and everything
downstream is native ffmpeg.

The model is [sb2702/longpipe](https://github.com/sb2702/longpipe)'s
work: the small matting net, MIT-licensed code and weights, converted
to ONNX at [imbcmdth/longpipe-onnx](https://huggingface.co/imbcmdth/longpipe-onnx).
This package follows the weights' license.

The weights are not in the archive: the manifest pins them - exact
repo, revision, file and sha256 - and `ffrwd install` fetches and
verifies them.

## Model export

- `matte(v)` returns the frame's foreground as one grayscale matte.
  The net is whole-frame matting: no classes, no threshold, no
  parameters - the output IS the foreground.

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
editor.

```
ffrwd run ffrwd/longpipe:background-blur -v source=call.mp4 -v dest=blurred.mp4
```

## Building

The module builds against the wit from the installed `ffrwd/wasm`
package:

```
ffrwd install -g ffrwd/wasm
cargo build --target wasm32-wasip2 --release
```

The weights are fetched at `ffrwd install` time by whoever installs
the published package; a development checkout runs the recipes only
after placing the pinned model beside the built module.
