import numpy as np
import matplotlib.pyplot as plt
import mwatershed

# Generate an empty ndarray of labels
labels = np.zeros((100, 100))

# Add circles with different labels
labels[25:40, 25:40] = 1
labels[60:75, 60:75] = 2
labels[50:70, 10:30] = 3

affs = np.zeros((2, 100, 100))
# affs[0] = labels > 0
# affs[1] = labels > 0
affs[0, 0, 0] = 1

result = mwatershed.agglom(affs - 0.5, [[0, 1], [1, 0]]).astype(np.uint32)
print(result.min(), result.max())

fig, ax = plt.subplots(1, 3, figsize=(10, 5))
# Plot the labels using imshow
ax[0].imshow(labels)
ax[1].imshow(affs[0])
ax[2].imshow(result)
plt.show()
