import skimage.io
import matplotlib.pyplot as plt

img = skimage.io.imread("/home/james/data/a4c3d/mva_train/01-008a78aa99bb79f6fce38aa90a458b7173a08a0bc74048e027452e26acd8d873/label_a4c_5c_7p_multires_0065_0102_001_012.png")
plt.imshow(img)
plt.show()