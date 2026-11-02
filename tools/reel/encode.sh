#!/bin/bash
# one encoded file per scene (same size, rate and settings), a concat with -c copy, then the music with -c:v copy
set -e
cd "$(dirname "$0")"
ENC="-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 30"
scene() { # name seconds fade-filter   (frame 0 of a scene is skipped: the first draw after a scene change comes back empty)
  local n=$1 d=$2 extra=$3
  ffmpeg -loglevel error -y -framerate 15 -start_number 1 -i frames/$n/f%04d.jpg -i frames/$n/overlay.png \
    -filter_complex "[0]scale=1280:720:flags=lanczos[v];[1]scale=1280:720[o];[v][o]overlay=format=auto,fps=30,trim=duration=$d${extra},format=yuv420p" $ENC tmp_$n.mp4 && mv tmp_$n.mp4 s_$n.mp4
}
scene open 3.0 ",fade=t=in:st=0:d=0.6"
scene flow 2.5 ""
scene wind 2.5 ""
scene storm 3.5 ""
scene surf 3.0 ""
scene upsets 3.0 ""
scene river 2.5 ""
scene wildmile 2.5 ""
scene launch 3.0 ",fade=t=out:st=2.6:d=0.4"
# the credits: a held still, 4 s, fading in and out
ffmpeg -loglevel error -y -framerate 30 -i credits.png -vf "format=yuv420p,loop=-1:1,trim=duration=4,fade=t=in:st=0:d=0.4,fade=t=out:st=3.4:d=0.6" $ENC tmp_credits.mp4 && mv tmp_credits.mp4 s_credits.mp4
printf "file 's_%s.mp4'\n" open flow wind storm surf upsets river wildmile launch credits > list.txt
ffmpeg -loglevel error -y -f concat -safe 0 -i list.txt -c copy tmp_silent.mp4 && mv tmp_silent.mp4 reel_silent.mp4
# the music: the synthesized ocean bed, levelled to -16 LUFS, muxed without touching the picture
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 reel_silent.mp4)
[ -f music.wav ] || python3 -I music.py "$D" music.wav
ffmpeg -loglevel error -y -i reel_silent.mp4 -i music.wav -map 0:v -map 1:a -c:v copy -af "loudnorm=I=-16:TP=-1.5:LRA=9" -c:a aac -b:a 160k -shortest tmp_reel.mp4 && mv tmp_reel.mp4 lakeview-reel.mp4
ffprobe -v error -show_entries format=duration:stream=codec_type,width,height,r_frame_rate -of default=nw=1 lakeview-reel.mp4
