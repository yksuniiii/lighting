import random

import numpy as np
from skimage import transform, io


def imread(path):
    # return scipy.misc.imread(path).astype(np.float)
    return io.imread(path)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    # scipy.misc.imsave(path, img)
    io.imsave(path, img)


def resize_image(image, dshape):
    factor = float(min(dshape[:2])) / min(image.shape[:2])
    new_size = [int(image.shape[0] * factor), int(image.shape[1] * factor)]
    if new_size[0] < dshape[0]:
        new_size[0] = dshape[0]
    if new_size[1] < dshape[0]:
        new_size[1] = dshape[0]
    resized_image = transform.resize(image, new_size)
    sample = np.asarray(resized_image) * 256
    if dshape[0] < sample.shape[0] or dshape[1] < sample.shape[1]:
        # random crop
        xx = int((sample.shape[0] - dshape[0]))
        yy = int((sample.shape[1] - dshape[1]))
        xstart = random.randint(0, xx)
        ystart = random.randint(0, yy)
        xend = xstart + dshape[0]
        yend = ystart + dshape[1]
        sample = sample[xstart:xend, ystart:yend, :]
    return sample