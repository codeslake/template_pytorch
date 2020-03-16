import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        model = [
                #### Half Size
                 nn.Conv2d(self.input_nc, dim, kernel_size=3, padding=0, stride=2),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                 nn.LeakyReLU(0.1, True),
                #### -1 Size
                 nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=1),
                 nn.LeakyReLU(0.1, True),
                #### Out Change
                 nn.Conv2d(dim, self.output_nc, kernel_size=1, padding=0),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        out = self.model(input)

        return out

class Network2(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        model = [
                #### Half Size
                 nn.Conv2d(self.input_nc, dim, kernel_size=3, padding=0, stride=2),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                 nn.LeakyReLU(0.1, True),
                #### -1 Size
                 nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=1),
                 nn.LeakyReLU(0.1, True),
                #### Out Change
                 nn.Conv2d(dim, self.output_nc, kernel_size=1, padding=0),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        out = self.model(input)

        return out
