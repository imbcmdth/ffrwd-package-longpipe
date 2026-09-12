-- The model export, hosted as the wasm module the package ships. The
-- weights themselves are pinned in the manifest and land beside the module
-- at install.
--
-- `matte` returns the frame's foreground as one grayscale alpha - white
-- where the subject is, black where the background is, blended at the
-- edges. The net is whole-frame matting: there are no classes and nothing
-- to narrow, so it takes only the stream. The matte keeps the picture's own
-- geometry and pixel format, ready for `ffrwd/mask_tools` and everything else
-- that reads a mask beside the picture.
CREATE FUNCTION matte(v video_stream)
RETURNS video_stream
  AS 'target/wasm32-wasip2/release/matte.wasm', 'matte' LANGUAGE wasm;
