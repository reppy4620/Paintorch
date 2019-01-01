import os
import torchvision.transforms as T
import torchvision.transforms.functional as F

from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TwoImgRandomCrop(T.RandomCrop):

    def __call__(self, img1, img2):
        i, j, h, w = self.get_params(img1, self.size)
        return F.crop(img1, i, j, h, w), F.crop(img2, i, j, h, w)


class LineColorDataset(Dataset):

    def __init__(self, line_path, color_path, resize=True, size=(512, 512)):
        self.line_path = line_path
        self.color_path = color_path
        self.lines = os.listdir(line_path)
        self.colors = os.listdir(color_path)
        self.resize = TwoImgRandomCrop(size) if resize else None
        assert len(self.lines) == len(self.colors)
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        line_img = Image.open(os.path.join(self.line_path,
                                           self.lines[index])).convert('L')
        color_img = Image.open(os.path.join(self.color_path,
                                            self.colors[index])).convert('RGB')
        if self.resize is not None:
            line_img, color_img = self.resize(line_img, color_img)
        return self.transforms(line_img), self.transforms(color_img)

    def __len__(self):
        return len(self.colors)


def test(line_path, color_path, size=(512, 512)):
    import matplotlib.pyplot as plt
    import numpy as np
    resize = TwoImgRandomCrop(size)
    line = Image.open(line_path).convert('L')
    color = Image.open(color_path).convert('RGB')
    line, color = resize(line, color)
    plt.figure()
    plt.imshow(np.asarray(line))
    plt.show()
    plt.figure()
    plt.imshow(np.asarray(color))
    plt.show()
