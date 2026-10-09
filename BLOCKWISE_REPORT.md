



# Blockwise Mutex Watershed Benchmark Report

Available parallelism: 96








## Results

| Scenario | Dim | Size | Block | Seq(ms) | BW(ms) | Speedup | SeqLabels | BWLabels | Match |
|----------|-----|------|-------|---------|--------|---------|-----------|----------|-------|
| 2D random (worst: many splits) | 2D | 512 | 2 | 155.8 | 280.7 | 0.55x | 44201 | 44201 | ✓ |
| 2D random (worst: many splits) | 2D | 512 | 4 | 155.8 | 176.8 | 0.88x | 44201 | 44201 | ✓ |
| 2D random (worst: many splits) | 2D | 512 | 8 | 155.8 | 158.1 | 0.99x | 44201 | 44201 | ✓ |
| 2D random (worst: many splits) | 2D | 512 | 16 | 155.8 | 123.7 | 1.26x | 44201 | 44201 | ✓ |
| 2D random (worst: many splits) | 2D | 512 | 32 | 155.8 | 97.8 | 1.59x | 44201 | 44201 | ✓ |
| 2D random (worst: many splits) | 2D | 512 | 64 | 155.8 | 74.2 | 2.10x | 44201 | 44201 | ✓ |
| 2D random (worst: many splits) | 2D | 512 | 128 | 155.8 | 79.8 | 1.95x | 44201 | 44201 | ✓ |
| 2D random (worst: many splits) | 2D | 512 | 256 | 155.8 | 108.8 | 1.43x | 44201 | 44201 | ✓ |
| 2D random (worst: many splits) | 2D | 512 | 512 | 155.8 | 190.8 | 0.82x | 44201 | 44201 | ✓ |
| 2D all-merge (best: one cluster) | 2D | 512 | 2 | 90.6 | 157.4 | 0.58x | 1 | 1 | ✓ |
| 2D all-merge (best: one cluster) | 2D | 512 | 4 | 90.6 | 101.8 | 0.89x | 1 | 1 | ✓ |
| 2D all-merge (best: one cluster) | 2D | 512 | 8 | 90.6 | 81.0 | 1.12x | 1 | 1 | ✓ |
| 2D all-merge (best: one cluster) | 2D | 512 | 16 | 90.6 | 66.9 | 1.35x | 1 | 1 | ✓ |
| 2D all-merge (best: one cluster) | 2D | 512 | 32 | 90.6 | 56.2 | 1.61x | 1 | 1 | ✓ |
| 2D all-merge (best: one cluster) | 2D | 512 | 64 | 90.6 | 53.7 | 1.69x | 1 | 1 | ✓ |
| 2D all-merge (best: one cluster) | 2D | 512 | 128 | 90.6 | 60.6 | 1.49x | 1 | 1 | ✓ |
| 2D all-merge (best: one cluster) | 2D | 512 | 256 | 90.6 | 86.5 | 1.05x | 1 | 1 | ✓ |
| 2D all-merge (best: one cluster) | 2D | 512 | 512 | 90.6 | 105.8 | 0.86x | 1 | 1 | ✓ |
| 2D small-objects (best for blockwise) | 2D | 512 | 2 | 105.9 | 168.8 | 0.63x | 4096 | 4096 | ✓ |
| 2D small-objects (best for blockwise) | 2D | 512 | 4 | 105.9 | 128.2 | 0.83x | 4096 | 4096 | ✓ |
| 2D small-objects (best for blockwise) | 2D | 512 | 8 | 105.9 | 92.2 | 1.15x | 4096 | 4096 | ✓ |
| 2D small-objects (best for blockwise) | 2D | 512 | 16 | 105.9 | 78.7 | 1.35x | 4096 | 4096 | ✓ |
| 2D small-objects (best for blockwise) | 2D | 512 | 32 | 105.9 | 65.6 | 1.62x | 4096 | 4096 | ✓ |
| 2D small-objects (best for blockwise) | 2D | 512 | 64 | 105.9 | 56.2 | 1.88x | 4096 | 4096 | ✓ |
| 2D small-objects (best for blockwise) | 2D | 512 | 128 | 105.9 | 56.4 | 1.88x | 4096 | 4096 | ✓ |
| 2D small-objects (best for blockwise) | 2D | 512 | 256 | 105.9 | 79.2 | 1.34x | 4096 | 4096 | ✓ |
| 2D small-objects (best for blockwise) | 2D | 512 | 512 | 105.9 | 122.2 | 0.87x | 4096 | 4096 | ✓ |
| 2D large-objects (worst for blockwise) | 2D | 512 | 2 | 120.0 | 187.7 | 0.64x | 11471 | 11471 | ✓ |
| 2D large-objects (worst for blockwise) | 2D | 512 | 4 | 120.0 | 131.7 | 0.91x | 11471 | 11471 | ✓ |
| 2D large-objects (worst for blockwise) | 2D | 512 | 8 | 120.0 | 109.7 | 1.09x | 11471 | 11471 | ✓ |
| 2D large-objects (worst for blockwise) | 2D | 512 | 16 | 120.0 | 92.3 | 1.30x | 11471 | 11471 | ✓ |
| 2D large-objects (worst for blockwise) | 2D | 512 | 32 | 120.0 | 83.2 | 1.44x | 11471 | 11471 | ✓ |
| 2D large-objects (worst for blockwise) | 2D | 512 | 64 | 120.0 | 72.5 | 1.66x | 11471 | 11471 | ✓ |
| 2D large-objects (worst for blockwise) | 2D | 512 | 128 | 120.0 | 77.5 | 1.55x | 11471 | 11471 | ✓ |
| 2D large-objects (worst for blockwise) | 2D | 512 | 256 | 120.0 | 108.8 | 1.10x | 11471 | 11471 | ✓ |
| 2D large-objects (worst for blockwise) | 2D | 512 | 512 | 120.0 | 128.1 | 0.94x | 11471 | 11471 | ✓ |
| 3D random (worst: many splits) | 3D | 40 | 2 | 47.8 | 60.7 | 0.79x | 4986 | 4986 | ✓ |
| 3D random (worst: many splits) | 3D | 40 | 4 | 47.8 | 46.7 | 1.02x | 4986 | 4986 | ✓ |
| 3D random (worst: many splits) | 3D | 40 | 8 | 47.8 | 41.1 | 1.16x | 4986 | 4986 | ✓ |
| 3D random (worst: many splits) | 3D | 40 | 16 | 47.8 | 40.3 | 1.18x | 4986 | 4986 | ✓ |
| 3D random (worst: many splits) | 3D | 40 | 32 | 47.8 | 69.9 | 0.68x | 4986 | 4986 | ✓ |
| 3D small-objects (best for blockwise) | 3D | 40 | 2 | 28.9 | 36.9 | 0.78x | 125 | 125 | ✓ |
| 3D small-objects (best for blockwise) | 3D | 40 | 4 | 28.9 | 26.3 | 1.10x | 125 | 125 | ✓ |
| 3D small-objects (best for blockwise) | 3D | 40 | 8 | 28.9 | 21.6 | 1.34x | 125 | 125 | ✓ |
| 3D small-objects (best for blockwise) | 3D | 40 | 16 | 28.9 | 22.1 | 1.31x | 125 | 125 | ✓ |
| 3D small-objects (best for blockwise) | 3D | 40 | 32 | 28.9 | 47.3 | 0.61x | 125 | 125 | ✓ |
| 3D large-objects (worst for blockwise) | 3D | 40 | 2 | 34.5 | 45.6 | 0.76x | 1164 | 1164 | ✓ |
| 3D large-objects (worst for blockwise) | 3D | 40 | 4 | 34.5 | 33.9 | 1.02x | 1164 | 1164 | ✓ |
| 3D large-objects (worst for blockwise) | 3D | 40 | 8 | 34.5 | 28.8 | 1.20x | 1164 | 1164 | ✓ |
| 3D large-objects (worst for blockwise) | 3D | 40 | 16 | 34.5 | 28.1 | 1.23x | 1164 | 1164 | ✓ |
| 3D large-objects (worst for blockwise) | 3D | 40 | 32 | 34.5 | 52.1 | 0.66x | 1164 | 1164 | ✓ |

## Speedup by Block Size

### 2D all-merge (best: one cluster)

  Size=512²  Sequential=90.6ms  Labels: seq=1 

  bs=   2 | █████████████                            0.58x  ✓
  bs=   4 | █████████████████████                    0.89x  ✓
  bs=   8 | ██████████████████████████               1.12x  ✓
  bs=  16 | ████████████████████████████████         1.35x  ✓
  bs=  32 | ██████████████████████████████████████   1.61x  ✓
  bs=  64 | ████████████████████████████████████████ 1.69x  ✓
  bs= 128 | ███████████████████████████████████      1.49x  ✓
  bs= 256 | ████████████████████████                 1.05x  ✓
  bs= 512 | ████████████████████                     0.86x  ✓

### 2D large-objects (worst for blockwise)

  Size=512²  Sequential=120.0ms  Labels: seq=11471 

  bs=   2 | ███████████████                          0.64x  ✓
  bs=   4 | ██████████████████████                   0.91x  ✓
  bs=   8 | ██████████████████████████               1.09x  ✓
  bs=  16 | ███████████████████████████████          1.30x  ✓
  bs=  32 | ██████████████████████████████████       1.44x  ✓
  bs=  64 | ████████████████████████████████████████ 1.66x  ✓
  bs= 128 | █████████████████████████████████████    1.55x  ✓
  bs= 256 | ██████████████████████████               1.10x  ✓
  bs= 512 | ██████████████████████                   0.94x  ✓

### 2D random (worst: many splits)

  Size=512²  Sequential=155.8ms  Labels: seq=44201 

  bs=   2 | ██████████                               0.55x  ✓
  bs=   4 | ████████████████                         0.88x  ✓
  bs=   8 | ██████████████████                       0.99x  ✓
  bs=  16 | ███████████████████████                  1.26x  ✓
  bs=  32 | ██████████████████████████████           1.59x  ✓
  bs=  64 | ████████████████████████████████████████ 2.10x  ✓
  bs= 128 | █████████████████████████████████████    1.95x  ✓
  bs= 256 | ███████████████████████████              1.43x  ✓
  bs= 512 | ███████████████                          0.82x  ✓

### 2D small-objects (best for blockwise)

  Size=512²  Sequential=105.9ms  Labels: seq=4096 

  bs=   2 | █████████████                            0.63x  ✓
  bs=   4 | █████████████████                        0.83x  ✓
  bs=   8 | ████████████████████████                 1.15x  ✓
  bs=  16 | ████████████████████████████             1.35x  ✓
  bs=  32 | ██████████████████████████████████       1.62x  ✓
  bs=  64 | ████████████████████████████████████████ 1.88x  ✓
  bs= 128 | ███████████████████████████████████████  1.88x  ✓
  bs= 256 | ████████████████████████████             1.34x  ✓
  bs= 512 | ██████████████████                       0.87x  ✓

### 3D large-objects (worst for blockwise)

  Size=40³  Sequential=34.5ms  Labels: seq=1164 

  bs=   2 | ████████████████████                     0.76x  ✓
  bs=   4 | ███████████████████████████              1.02x  ✓
  bs=   8 | ███████████████████████████████          1.20x  ✓
  bs=  16 | ████████████████████████████████         1.23x  ✓
  bs=  32 | █████████████████                        0.66x  ✓

### 3D random (worst: many splits)

  Size=40³  Sequential=47.8ms  Labels: seq=4986 

  bs=   2 | ████████████████████                     0.79x  ✓
  bs=   4 | ███████████████████████████              1.02x  ✓
  bs=   8 | ███████████████████████████████          1.16x  ✓
  bs=  16 | ███████████████████████████████          1.18x  ✓
  bs=  32 | ██████████████████                       0.68x  ✓

### 3D small-objects (best for blockwise)

  Size=40³  Sequential=28.9ms  Labels: seq=125 

  bs=   2 | ████████████████████                     0.78x  ✓
  bs=   4 | █████████████████████████████            1.10x  ✓
  bs=   8 | ███████████████████████████████████      1.34x  ✓
  bs=  16 | ██████████████████████████████████       1.31x  ✓
  bs=  32 | ████████████████                         0.61x  ✓

