-- The raw matte as its own grayscale video, for anyone composing by hand.
-- variables: source (input media path), track (video track index, defaults to the first), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/longpipe/recipes/matte.sql -v source=call.mp4 -v dest=matte.mp4
COPY (
  SELECT ffrwd.longpipe.matte(v)
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'libx264', crf 20)
