import cv2
import os
import shutil
import numpy as np


# size filter
def size_filter(root, size=(512, 512), mode='move'):
    paths = os.listdir(root)
    length = len(paths)
    for i, path in enumerate(paths, start=1):
        img_path = os.path.join(root, path)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, c = img.shape
        if h < size[0] or w < size[1]:
            if mode == 'move':
                shutil.move(img_path, 'path/to/small_img_dir')
            elif mode == 'remove':
                os.remove(img_path)
        if i % 1000 == 0:
            print(f'{i}/{length} images ended')


# judge image color or black-white
def image_filter(root, mode='move', with_size_filter=False, size=(512, 512), threshold=0.1):
    paths = os.listdir(root)
    length = len(paths)
    for i, path in enumerate(paths, start=1):
        img_path = os.path.join(root, path)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, c = img.shape
        if with_size_filter:
            if h < size[0] or w < size[1]:
                if mode == 'move':
                    shutil.move(img_path, 'path/to/small_img_dir')
                elif mode == 'remove':
                    os.remove(img_path)
                continue
        b, g, r = cv2.split(img)

        r_g = np.count_nonzero(abs(r - g))
        r_b = np.count_nonzero(abs(r - b))
        g_b = np.count_nonzero(abs(g - b))
        diff_sum = float(r_g + r_b + g_b)

        ratio = diff_sum / img.size
        # ----------------------------------------------
        # FIXME
        # please change threshold to preferred value.
        # i use 0.1 because i want brilliant illustration.
        # i fill accuracy is about 80%.
        # ----------------------------------------------
        # if color
        if ratio > threshold:
            continue
        # if grey
        else:
            if mode == 'move':
                shutil.move(img_path, 'path/to/grey_img_dir')
            elif mode == 'remove':
                os.remove(img_path)
        if i % 10000 == 0:
            print(f'{i} / {length} ended')


if __name__ == '__main__':
    image_filter(root='path/to/ColorArtFolder',
                 mode='move',
                 with_size_filter=True,
                 threshold=0.1)
