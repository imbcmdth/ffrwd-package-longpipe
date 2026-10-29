-- The raw matte as its own grayscale video, for anyone composing by hand.
-- variables: source (input media path), tier (model size: xs, small, medium, large or xl, defaults to small), track (video track index, defaults to the first), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/longpipe/recipes/matte.sql -v source=call.mp4 -v dest=matte.mp4 -v tier=xl
COPY (
  SELECT ffrwd.longpipe.matte(v, :'tier')
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'libx264', crf 20, pix_fmt 'yuv420p')
