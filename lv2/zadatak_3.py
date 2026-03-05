import numpy as np
import matplotlib.pyplot as plt

img = plt.imread(r'C:\Users\student\Desktop\lv1_osu_pb\osnove_strojnog_ucenja_lv\lv2\road.jpg')

plt.imshow(img)
plt.axis("off")
plt.show()

bright = img * 1.2
bright = np.clip(bright, 0, 1)

plt.imshow(bright)
plt.axis("off")
plt.show()

h, w, _ = img.shape

second_quarter = img[:, w//4:w//2]

plt.imshow(second_quarter)
plt.axis("off")
plt.show()

rotated = np.rot90(img, -1)

plt.imshow(rotated)
plt.axis("off")
plt.show()

mirror = img[:, ::-1]

plt.imshow(mirror)
plt.axis("off")
plt.show()