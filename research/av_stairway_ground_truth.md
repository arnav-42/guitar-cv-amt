# Stairway short clip: played TAB ground truth

Source: `videos/stairway-intro-short.mp4` (1920×1080, 30 fps, 15.018 s). The TAB score is overlaid below the performance. The played portion contains a whole-measure rest followed by 26 attacks and 36 notes through the third chord of measure 5. Machine-readable events are in [`av_stairway_ground_truth.json`](av_stairway_ground_truth.json).

String numbers use standard TAB convention: 1 = high E, 2 = B, 3 = G, 4 = D, 5 = A, 6 = low E. Each parenthesized item below is `(string, fret)`; notes in one bracket sound together.

| Measure | Played attacks in order |
| --- | --- |
| 1 | Whole-measure rest |
| 2 | `[(4,7)] [(3,5)] [(2,5)] [(1,5)] [(4,6),(1,7)] [(2,5)] [(3,5)] [(1,7)]` |
| 3 | `[(4,5),(1,8)] [(2,5)] [(3,5)] [(1,8)] [(4,4),(1,2)] [(2,3)] [(3,2)] [(1,2)]` |
| 4 | `[(4,3),(1,0)] [(2,1)] [(3,2)] [(2,1)] [(1,0)] [(2,1)] [(3,2)]` |
| 5 (within clip) | `[(5,2),(3,0),(2,0)] [(5,0),(3,2),(2,1)] [(5,0),(3,2),(2,1)]` |

## Evidence and timing

- At 3–8 seconds, the overlay shows measures 1–3 and the first measure-2 attack as `(4,7)`. At 8 seconds, all eight measure-2 attacks and all eight measure-3 attacks are readable. The cyan playhead crosses measure-2 slots at about 3.30–6.20 seconds.
- At 9–12 seconds, the score scrolls and shows the end of measure 3 and all of measure 4. The measure-4 first attack `(4,3),(1,0)` is at about 9.97 seconds.
- At 13–14.97 seconds, the score shows the end of measure 4 and measure 5. The playhead crosses the first three measure-5 chords at about 13.30, 13.73, and 14.20 seconds. By the last decoded video frame at 14.967 seconds, the cyan line is at x≈984. The next measure-5 note, an open A string, is at x≈1036 and has not been reached.

Times in the JSON are score-playhead crossings, estimated to roughly ±0.1 second. They are suitable for event matching after allowing a small audio/visual offset. The string/fret sequence is read directly from the overlay, with six TAB rows at y≈856, 885, 914, 943, 971, 1000 in the 1080-pixel frame. The score's unplayed lookahead in measures 5–7 must not be counted as ground truth for this short clip.
