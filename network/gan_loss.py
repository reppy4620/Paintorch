import torch
import torch.nn as nn


class GanLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, x, is_real):
        if is_real:
            target = self.real_label
        else:
            target = self.fake_label
        return target.expand_as(x)

    def __call__(self, x, is_real):
        target = self.get_target_tensor(x, is_real)
        return self.loss(x, target.cuda())
