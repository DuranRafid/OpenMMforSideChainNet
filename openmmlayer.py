import torch
import torch.nn as nn

import sidechainnet as scn
from openmmfunction import OpenMMFunction


class OpenMMLayer(nn.Module):
    def __init__(self, sequence, coords):
        super(OpenMMLayer, self).__init__()
        self.seq = sequence
        self.coords = nn.Parameter(torch.Tensor(coords))
        #self.register_parameter('seq', None)

    def forward(self):
        return OpenMMFunction.apply(self.coords, self.seq)

if __name__ == '__main__':
    data = scn.load(casp_version=12, thinning=30)
    model = OpenMMLayer(data['train']['seq'][0],data['train']['crd'][0])

