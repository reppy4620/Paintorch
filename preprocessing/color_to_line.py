import numpy as np
import cv2
import os
import shutil


# in_root: color image directory
# out_root: line image directory
def change2line(in_root, out_root):

    # change color art to line art
    def color_to_line(i, path, out_root):
        neighbor_hood_8 = np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]],
                                   np.uint8)
        try:
            img = cv2.imread(path, 0)
            img_dilate = cv2.dilate(img, neighbor_hood_8, iterations=1)
            img_diff = cv2.absdiff(img, img_dilate)
            img_diff_not = cv2.bitwise_not(img_diff)
            cv2.imwrite(os.path.join(out_root, f'l_{i}.jpg'), img_diff_not)
        except:
            # if error
            shutil.move(path, 'path/to/err_img_dir')
    if not os.path.isdir(out_root):
        os.mkdir(out_root)
    length = len(os.listdir(in_root))
    for i, path in enumerate(os.listdir(in_root)):
        root, ext = os.path.splitext(path)
        if ext == '.jpeg' or ext == '.jpg' or ext == 'png':
            color_path = os.path.join(in_root, path)
            color_to_line(i, color_path, out_root)
            os.rename(color_path, os.path.join(in_root, f'color_{i}.jpg'))
            if i % 1000 == 0:
                print(f'{i}/{length} Ended')


if __name__ == '__main__':
    change2line('path/to/color_img_dir', 'path/to/line_img_dir')
