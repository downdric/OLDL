import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
# def cross_entropy(logits, target, size_average=True):
#     if size_average:
#         return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
#     else:
#         return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


class CAD(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self):
        super(CAD, self).__init__()
    def forward(self, input, target):
        pdtldl = input
        gtldl = target
        losses = torch.zeros(1).cuda()
        lo1_sum = torch.zeros(1).cuda()
        
        for i in range(0, 10):
            m = i+1
            lo1 = abs(pdtldl[:, 0:m].sum(1) - gtldl[:, 0:m].sum(1))
            lo1_sum=lo1_sum+torch.mean(lo1)
            
        losses=lo1_sum
        return losses
    
    

