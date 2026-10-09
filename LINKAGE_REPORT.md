# Linkage criteria benchmark

Best of 3 runs per cell, release build, single thread.
`rel` is wall time relative to `abs_max` (mutex watershed) on the same input;
`segments` is the number of non-background labels. `_nocl` disables cannot-link constraints.

| Scenario | Shape | Linkage | Time (ms) | rel | Segments |
|----------|-------|---------|----------:|----:|---------:|
| 2D random normal | 128x128 | abs_max | 12.0 | 1.00x | 745 |
| 2D random normal | 128x128 | sum | 41.4 | 3.45x | 724 |
| 2D random normal | 128x128 | sum_nocl | 45.3 | 3.78x | 728 |
| 2D random normal | 128x128 | mean | 43.6 | 3.64x | 509 |
| 2D random normal | 128x128 | mean_nocl | 48.3 | 4.03x | 514 |
| 2D random normal | 128x128 | max | 42.6 | 3.56x | 3 |
| 2D random normal | 128x128 | max_nocl | 47.1 | 3.93x | 3 |
| 2D random normal | 128x128 | min | 41.9 | 3.50x | 790 |
| 2D random normal | 128x128 | min_nocl | 45.7 | 3.82x | 790 |
| 2D blobs (16px) | 128x128 | abs_max | 7.7 | 1.00x | 223 |
| 2D blobs (16px) | 128x128 | sum | 35.8 | 4.65x | 69 |
| 2D blobs (16px) | 128x128 | sum_nocl | 36.9 | 4.79x | 69 |
| 2D blobs (16px) | 128x128 | mean | 45.7 | 5.94x | 38 |
| 2D blobs (16px) | 128x128 | mean_nocl | 47.3 | 6.15x | 45 |
| 2D blobs (16px) | 128x128 | max | 43.8 | 5.70x | 1 |
| 2D blobs (16px) | 128x128 | max_nocl | 45.3 | 5.89x | 1 |
| 2D blobs (16px) | 128x128 | min | 43.8 | 5.70x | 230 |
| 2D blobs (16px) | 128x128 | min_nocl | 45.3 | 5.90x | 230 |
| 2D random normal | 256x256 | abs_max | 58.9 | 1.00x | 2652 |
| 2D random normal | 256x256 | sum | 279.0 | 4.74x | 2660 |
| 2D random normal | 256x256 | sum_nocl | 299.6 | 5.09x | 2716 |
| 2D random normal | 256x256 | mean | 314.5 | 5.34x | 1860 |
| 2D random normal | 256x256 | mean_nocl | 336.3 | 5.71x | 1863 |
| 2D random normal | 256x256 | max | 286.2 | 4.86x | 14 |
| 2D random normal | 256x256 | max_nocl | 309.4 | 5.25x | 14 |
| 2D random normal | 256x256 | min | 284.3 | 4.83x | 2950 |
| 2D random normal | 256x256 | min_nocl | 302.1 | 5.13x | 2950 |
| 2D blobs (16px) | 256x256 | abs_max | 34.0 | 1.00x | 840 |
| 2D blobs (16px) | 256x256 | sum | 218.5 | 6.44x | 271 |
| 2D blobs (16px) | 256x256 | sum_nocl | 224.2 | 6.60x | 271 |
| 2D blobs (16px) | 256x256 | mean | 323.1 | 9.52x | 148 |
| 2D blobs (16px) | 256x256 | mean_nocl | 331.9 | 9.78x | 168 |
| 2D blobs (16px) | 256x256 | max | 292.1 | 8.60x | 2 |
| 2D blobs (16px) | 256x256 | max_nocl | 301.4 | 8.88x | 2 |
| 2D blobs (16px) | 256x256 | min | 299.8 | 8.83x | 908 |
| 2D blobs (16px) | 256x256 | min_nocl | 308.3 | 9.08x | 908 |
| 3D random normal | 32x32x32 | abs_max | 32.9 | 1.00x | 359 |
| 3D random normal | 32x32x32 | sum | 161.1 | 4.90x | 498 |
| 3D random normal | 32x32x32 | sum_nocl | 177.8 | 5.41x | 492 |
| 3D random normal | 32x32x32 | mean | 183.1 | 5.57x | 170 |
| 3D random normal | 32x32x32 | mean_nocl | 205.0 | 6.23x | 170 |
| 3D random normal | 32x32x32 | max | 167.2 | 5.08x | 1 |
| 3D random normal | 32x32x32 | max_nocl | 184.0 | 5.59x | 1 |
| 3D random normal | 32x32x32 | min | 169.3 | 5.15x | 319 |
| 3D random normal | 32x32x32 | min_nocl | 185.9 | 5.65x | 319 |
| 3D blobs (8vx) | 32x32x32 | abs_max | 22.4 | 1.00x | 229 |
| 3D blobs (8vx) | 32x32x32 | sum | 125.4 | 5.60x | 66 |
| 3D blobs (8vx) | 32x32x32 | sum_nocl | 131.1 | 5.86x | 66 |
| 3D blobs (8vx) | 32x32x32 | mean | 188.3 | 8.42x | 14 |
| 3D blobs (8vx) | 32x32x32 | mean_nocl | 196.5 | 8.79x | 14 |
| 3D blobs (8vx) | 32x32x32 | max | 169.4 | 7.57x | 1 |
| 3D blobs (8vx) | 32x32x32 | max_nocl | 177.0 | 7.91x | 1 |
| 3D blobs (8vx) | 32x32x32 | min | 176.0 | 7.87x | 306 |
| 3D blobs (8vx) | 32x32x32 | min_nocl | 181.7 | 8.13x | 306 |
| 3D random normal | 48x48x48 | abs_max | 146.6 | 1.00x | 1083 |
| 3D random normal | 48x48x48 | sum | 924.7 | 6.31x | 1387 |
| 3D random normal | 48x48x48 | sum_nocl | 1010.8 | 6.89x | 1490 |
| 3D random normal | 48x48x48 | mean | 1077.0 | 7.34x | 386 |
| 3D random normal | 48x48x48 | mean_nocl | 1202.7 | 8.20x | 386 |
| 3D random normal | 48x48x48 | max | 978.3 | 6.67x | 3 |
| 3D random normal | 48x48x48 | max_nocl | 1080.6 | 7.37x | 3 |
| 3D random normal | 48x48x48 | min | 1001.9 | 6.83x | 788 |
| 3D random normal | 48x48x48 | min_nocl | 1072.2 | 7.31x | 788 |
| 3D blobs (8vx) | 48x48x48 | abs_max | 93.8 | 1.00x | 777 |
| 3D blobs (8vx) | 48x48x48 | sum | 642.2 | 6.84x | 219 |
| 3D blobs (8vx) | 48x48x48 | sum_nocl | 658.7 | 7.02x | 219 |
| 3D blobs (8vx) | 48x48x48 | mean | 1095.3 | 11.67x | 28 |
| 3D blobs (8vx) | 48x48x48 | mean_nocl | 1136.2 | 12.11x | 38 |
| 3D blobs (8vx) | 48x48x48 | max | 969.3 | 10.33x | 1 |
| 3D blobs (8vx) | 48x48x48 | max_nocl | 1022.2 | 10.89x | 1 |
| 3D blobs (8vx) | 48x48x48 | min | 1020.5 | 10.88x | 962 |
| 3D blobs (8vx) | 48x48x48 | min_nocl | 1055.2 | 11.25x | 962 |
