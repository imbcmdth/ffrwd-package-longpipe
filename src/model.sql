-- The model export, hosted as the wasm module the package ships. The
-- weights are pinned in the manifest, one file per tier, and land beside
-- the module at install; the node reads the tier's file when it opens.
--
-- `matte` returns the frame's foreground as one alpha matte: white where
-- the subject is, black where the background is, blended at the edges. The
-- net is whole-frame matting: there are no classes and nothing to narrow.
-- `tier` picks the network, xs, small, medium, large or xl; NULL or left
-- out is small. The matte is the picture's own size with the alpha in every
-- channel, ready for `ffrwd/mask_tools` and everything else that reads a
-- mask beside the picture.
CREATE FUNCTION matte(v video_stream, tier text DEFAULT 'small')
RETURNS video_stream
  AS 'matte-webgpu/build/matte.wasm', 'matte' LANGUAGE wasm;
