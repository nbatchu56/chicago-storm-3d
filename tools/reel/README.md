# The LakeView reel

A 26 s feature reel filmed from the page itself, plus a credits card.

1. Serve a built page: `python3 -m http.server 8123` in a folder holding `world.html` and `data/` (the snapshot branch).
2. `node cap.js` films the scenes listed in `cap.js` (frames at 15 fps, 960x540, read straight off the WebGL canvas on a
   virtual clock; the UI and the caption as one transparent overlay per scene). `node cap.js night launch` refilms some.
3. `node still.js credits.html credits.png` renders the credits card.
4. `bash encode.sh` encodes a file per scene at 1280x720, 30 fps, and joins them into `lakeview-reel.mp4`.

Needs Playwright with its Chromium (software GL is fine: ~2 s a frame) and ffmpeg.
