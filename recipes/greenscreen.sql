-- The matte as the picture's own alpha channel, in a codec that keeps it:
-- the background is transparent, ready for compositing elsewhere.
-- variables: source (input media path), track (video track index, defaults to the first), dest (output path, a .mov)
-- example: ffrwd compile -f packages/ffrwd/longpipe/recipes/greenscreen.sql -v source=call.mp4 -v dest=subject.mov
COPY (
  SELECT ffmpeg.alphamerge(ffmpeg.format(v, 'rgba'),
                           ffmpeg.format(ffrwd.longpipe.matte(v), 'gray')), f.audio
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'prores_ks', pix_fmt 'yuva444p10le')
