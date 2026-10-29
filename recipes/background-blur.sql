-- Blur the background and keep the subject sharp: the matte inverted picks
-- everything that is not the foreground, and the blur lands there.
-- variables: source (input media path), sigma (blur strength, defaults to 12), tier (model size: xs, small, medium, large or xl, defaults to small), track (video track index, defaults to the first), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/longpipe/recipes/background-blur.sql -v source=call.mp4 -v dest=blurred.mp4 -v tier=xl
COPY (
  SELECT ffrwd.mask_tools.blur_where(v, ffmpeg.negate(ffrwd.longpipe.matte(v, :'tier')), :sigma), f.audio
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'libx264', crf 20)
