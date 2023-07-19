import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math

from torch.nn.modules import loss



class CJS(nn.Module):
    def __init__(self):
        super(CJS, self).__init__()
    def forward(self, input, target):
        pdtldl = input
        gtldl = target
        losses = torch.zeros(1).cuda()
        lo1_sum= torch.zeros(1).cuda()
        
        for i in range(0, 10):
            PD = pdtldl[:, :i+1].sum(1)
            GT = gtldl[:, :i+1].sum(1)
            M = (PD+GT)/2
            temp = 0.5 * PD * (torch.log(PD) - torch.log(M)) + 0.5 * (GT) * (torch.log(GT + 1e-8) - torch.log(M))
            lo = temp
            lo1_sum += torch.mean(lo)            
        losses=lo1_sum    
        return losses    
            
        