import numpy as np
import torch.nn as nn

from .norm_type import NormType
from .utils import weights_init


def define_D(input_nc, ndf=64, n_layers=3, norm=NormType.Group):
    d = Discriminator(input_nc, ndf, n_layers, norm=norm)
    d.apply(weights_init)
    return d


class Discriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm=NormType.Group):
        super().__init__()

        kw = 4
        padw = int(np.ceil(kw-1)/2)
        s = [
            nn.Conv2d(input_nc, ndf, kw, 2, padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            s += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kw, 2, padw),
                nn.GroupNorm(32, ndf*nf_mult) if norm == NormType.Group else
                nn.InstanceNorm2d(ndf*nf_mult) if norm == NormType.Instance else
                nn.BatchNorm2d(ndf*nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        s += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kw, 1, padw),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        s += [nn.Conv2d(ndf * nf_mult, 1, kw, 1, padw)]

        s += [nn.Sigmoid()]

        self.model = nn.Sequential(*s)

    def forward(self, x):
        return self.model(x)
