from dataset import LineColorDataset
from torch.utils.data import DataLoader


def get_train_loader(color_root, batch_size=64, resize=False, size=(256, 256)):
    data_set = LineColorDataset(color_root, resize=resize, size=size)
    data_loader = DataLoader(
        dataset=data_set,
        shuffle=True,
        batch_size=batch_size
    )
    return data_loader
