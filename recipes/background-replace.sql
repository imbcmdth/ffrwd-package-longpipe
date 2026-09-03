-- Cut the foreground out and lay it over a new background, scaled to fit
-- the picture.
-- variables: source (input media path), background (media whose first video track becomes the backdrop), track (video track index, defaults to the first), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/longpipe/recipes/background-replace.sql -v source=call.mp4 -v background=beach.mp4 -v dest=composited.mp4
COPY (
  SELECT ffrwd.mask_tools.cutout(v, ffrwd.longpipe.matte(v),
                                 ffmpeg.scale(g.video[1], v.width, v.height)), f.audio
  FROM input(:'source') f, unnest(f.video) v, input(:'background') g
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'libx264', crf 20)
