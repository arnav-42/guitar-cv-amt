# Stairway 28-second clip: played TAB ground truth

Source: `videos/stairway-intro.webm` (1920×1080, 30 fps, 28.021 s). The TAB score is overlaid below the performance. The played portion contains a whole-measure rest followed by 55 attacks and 75 notes through the third chord of measure 9. Machine-readable events, with approximate playhead-crossing times, are in [`av_stairway_full_ground_truth.json`](av_stairway_full_ground_truth.json).

String numbers use standard TAB convention: 1 = high E, 2 = B, 3 = G, 4 = D, 5 = A, 6 = low E. Each parenthesized item below is `(string, fret)`; notes in one bracket sound together.

| Measure | Played attacks in order |
| --- | --- |
| 1 | Whole-measure rest |
| 2 | `[(4,7)] [(3,5)] [(2,5)] [(1,5)] [(4,6),(1,7)] [(2,5)] [(3,5)] [(1,7)]` |
| 3 | `[(4,5),(1,8)] [(2,5)] [(3,5)] [(1,8)] [(4,4),(1,2)] [(2,3)] [(3,2)] [(1,2)]` |
| 4 | `[(4,3),(1,0)] [(2,1)] [(3,2)] [(2,1)] [(1,0)] [(2,1)] [(3,2)]` |
| 5 | `[(5,2),(3,0),(2,0)] [(5,0),(3,2),(2,1)] [(5,0),(3,2),(2,1)] [(5,0)] [(5,8)] [(5,7)]` |
| 6 | `[(5,0)] [(4,7)] [(3,5)] [(2,5)] [(4,6),(1,7)] [(2,5)] [(3,5)] [(1,7)]` |
| 7 | `[(4,5),(1,8)] [(2,5)] [(3,5)] [(1,8)] [(4,4),(1,2)] [(2,3)] [(3,2)] [(1,2)]` |
| 8 | `[(4,3),(1,0)] [(2,1)] [(3,2)] [(2,1)] [(1,0)] [(2,1)] [(3,2)]` |
| 9 (within clip) | `[(5,2),(3,0),(2,0)] [(5,0),(3,2),(2,1)] [(5,0),(3,2),(2,1)]` |

## Evidence and timing

- Measures 1–5 through 14.2 seconds match the short clip reference in [`av_stairway_ground_truth.md`](av_stairway_ground_truth.md). The full clip reveals measure 5's remaining open A, A-string fret 8, and A-string fret 7 at approximately 15.50, 15.83, and 16.20 seconds. A slide mark joins open A to fret 8 in the score.
- At 17–18 seconds, measure 6 is visible. Its arpeggio repeats measure 2 except that the first note is open A string rather than D-string fret 7. The playhead crosses its first and last notes at approximately 16.63 and 19.53 seconds.
- At 20–22 seconds, measure 7 is visible and repeats measure 3. Its first and last attacks are approximately 19.97 and 22.87 seconds.
- At 23–26 seconds, measure 8 is visible and repeats measure 4. Its first and last attacks are approximately 23.27 and 26.20 seconds.
- At 27–27.97 seconds, measure 9 is visible. The playhead crosses only its first three chords, at approximately 26.63, 27.03, and 27.50 seconds. At the last decoded frame (27.967 seconds), the playhead is at x≈1452; the following open A note is at x≈1538 and is not played in this clip. Measures 9–10 beyond that point are score lookahead, not ground truth.

Times in the JSON are score-playhead crossings, usually within ±0.1 second. The sixth attack in measures 3 and 6 coincides with a fast score-scroll animation, so its timing is interpolated to about ±0.2 second. These times can be aligned to audio with a small global offset. String/fret assignments are read directly from the overlaid TAB, whose six rows are at y≈856, 885, 914, 943, 971, 1000 in the 1080-pixel frame. This is an overlay transcription, not a pipeline prediction.
