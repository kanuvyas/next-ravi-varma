# Names vs description test
Winner: D_lr2e-4 checkpoint-800, scale 0.8. Grid: names_test_grid.png (columns: name s1, name s2, description s1, description s2).

CLIP similarity of each image to the DESCRIPTION prompt:

|   pair | prompt   |   seed |   clip_vs_description |
|-------:|:---------|-------:|----------------------:|
|      1 | name     |      1 |                0.1991 |
|      1 | name     |      2 |                0.2216 |
|      1 | desc     |      1 |                0.2188 |
|      1 | desc     |      2 |                0.2465 |
|      2 | name     |      1 |                0.212  |
|      2 | name     |      2 |                0.2589 |
|      2 | desc     |      1 |                0.2664 |
|      2 | desc     |      2 |                0.2731 |
|      3 | name     |      1 |                0.2436 |
|      3 | name     |      2 |                0.3263 |
|      3 | desc     |      1 |                0.3052 |
|      3 | desc     |      2 |                0.3275 |
|      4 | name     |      1 |                0.2661 |
|      4 | name     |      2 |                0.2551 |
|      4 | desc     |      1 |                0.3143 |
|      4 | desc     |      2 |                0.3518 |

Conclusion: The name images are partly similar to the description images, but with weaker or more generic content.
Implication: training captions had no character names, so M3's prompt expander must turn names into visual descriptions.
