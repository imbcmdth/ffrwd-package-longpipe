-- The matte as the picture's own alpha channel, in a codec that keeps it:
-- the background is transparent, ready for compositing elsewhere.
-- variables: source (input media path), tier (model size: xs, small, medium, large or xl, defaults to small), track (video track index, defaults to the first), dest (output path, a .mov)
-- example: ffrwd compile -f packages/ffrwd/longpipe/recipes/greenscreen.sql -v source=call.mp4 -v dest=subject.mov -v tier=xl
COPY (
  SELECT ffmpeg.alphamerge(ffmpeg.format(v, 'rgba'), ffrwd.longpipe.matte(v, :'tier')), f.audio
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'prores_ks', pix_fmt 'yuva444p10le')
