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


class EMDloss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self):
        super(EMDloss, self).__init__()
    def forward(self, input, target):
        pdtldl = input
        gtldl = target
        #temp = (gtldl-pdtldl)*(torch.log(gtldl+1e-8)-torch.log(pdtldl+1e-8))
        losses = torch.zeros(1).cuda()
        #m = (pdtldl+gtldl)/2
        #temp = 0.5*(pdtldl)*(torch.log(pdtldl)-torch.log(m))+0.5*(gtldl)*(torch.log(gtldl+1e-8)-torch.log(m))
        for i in range(0, 85):
            m = i+1
            lo = abs(pdtldl[:, 0:m].sum(0) - gtldl[:, 0:m].sum(0))
            #torch.exp((abs(pdtldl[:, i] - gtldl[:, i])) / 100) *
            losses += torch.mean(lo)

        # losses = losses / 5
        #losses = temp.sum(1).mean()


        return losses/10

# lo = torch.exp((abs(pdtldl[:, i] - gtldl[:, i])) / 0.5) * (pdtldl[:, i] - gtldl[:, i]) ** 2 + abs(i - j) * torch.exp(gtldl[:, i]) * (pdtldl[:, j] - gtldl[:, j]) ** 2
