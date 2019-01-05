import torch.nn as nn

from .utils import weights_init


def define_D(device, input_nc, ndf=64, n_layers=3):
    d = Discriminator(input_nc, ndf, n_layers).to(device)
    d.apply(weights_init)
    return d


class Down(nn.Module):

    def __init__(self, i, o, k, s, p):
        super(Down, self).__init__()
        self.conv = nn.Conv2d(i, o, k, s, p)
        self.bn = nn.BatchNorm2d(o)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Discriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(Discriminator, self).__init__()
        self.in_layer = Down(input_nc, ndf, 4, 2, 1)
        layers = list()
        for _ in range(n_layers):
            layers += [
                Down(ndf, ndf, 3, 1, 1),
                Down(ndf, ndf * 2, 4, 2, 1),
            ]
            ndf *= 2
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Sequential(
            nn.Conv2d(ndf, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.layers(x)
        return self.out_layer(x)


if __name__ == '__main__':
    import torch

    net = define_D(torch.device('cpu'), 3)
    print(net)
