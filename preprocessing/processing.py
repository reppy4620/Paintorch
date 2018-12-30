import cv2
import os
import shutil
import numpy as np

from collections import Counter


def size_filter(in_root, out_root, size=(256, 256)):
    if not os.path.isdir(out_root):
        os.mkdir(out_root)
    paths = os.listdir(in_root)
    cnt = Counter()
    for i, img_path in enumerate(paths):
        cnt.clear()
        im = cv2.imread(os.path.join(in_root, img_path))
        if im is None:
            continue
        h, w, c = im.shape
        if h < size[0] or w < size[1]:
            shutil.copyfile(os.path.join(in_root, img_path),
                            os.path.join(out_root, img_path))
        if i % 100 == 0:
            print(f'{i} / {len(paths)} Ended')


if __name__ == '__main__':
    size_filter('D:/ImageDatas/Color', 'D:/ImageDatas/Color', size=(512, 512))
