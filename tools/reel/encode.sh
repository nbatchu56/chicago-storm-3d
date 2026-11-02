#!/bin/bash
# one encoded file per scene (same size, rate and settings), then a concat with -c copy
set -e
cd "$(dirname "$0")"
ENC="-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 30"
scene() { # name fade-filter   (frame 0 of a scene is skipped: the first draw after a scene change comes back empty)
  local n=$1 extra=$2
  ffmpeg -loglevel error -y -framerate 15 -start_number 1 -i frames/$n/f%04d.jpg -i frames/$n/overlay.png \
    -filter_complex "[0]scale=1280:720:flags=lanczos[v];[1]scale=1280:720[o];[v][o]overlay=format=auto,fps=30${extra},format=yuv420p" $ENC tmp_$n.mp4 && mv tmp_$n.mp4 s_$n.mp4
}
scene open ",fade=t=in:st=0:d=0.6"
scene flow ""
scene wind ""
scene river ""
scene wildmile ""
scene launch ""
scene night ",fade=t=out:st=2.5:d=0.4"
# the credits: a held still, 5 s, fading in and out
ffmpeg -loglevel error -y -framerate 30 -i credits.png -vf "format=yuv420p,loop=-1:1,trim=duration=5,fade=t=in:st=0:d=0.4,fade=t=out:st=4.4:d=0.6" $ENC tmp_credits.mp4 && mv tmp_credits.mp4 s_credits.mp4
printf "file 's_%s.mp4'\n" open flow wind river wildmile launch night credits > list.txt
ffmpeg -loglevel error -y -f concat -safe 0 -i list.txt -c copy tmp_reel.mp4 && mv tmp_reel.mp4 lakeview-reel.mp4
ffprobe -v error -show_entries format=duration:stream=width,height,r_frame_rate -of default=nw=1 lakeview-reel.mp4
