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


class QFDloss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self):
        super(QFDloss, self).__init__()

        self.A1= [[1, 9 / 10, 8 / 10, 7 / 10, 6 / 10, 5 / 10, 4 / 10, 3 / 10, 2 / 10, 1 / 10],
                  [9 / 10, 1, 9 / 10, 8 / 10, 7 / 10, 6 / 10, 5 / 10, 4 / 10, 3 / 10, 2 / 10],
                  [8 / 10, 9 / 10, 1, 9 / 10, 8 / 10, 7 / 10, 6 / 10, 5 / 10, 4 / 10, 3 / 10],
                  [7 / 10, 8 / 10, 9 / 10, 1, 9 / 10, 8 / 10, 7 / 10, 6/ 10, 5 / 10, 4 / 10],
                  [6 / 10, 7 / 10, 8 / 10, 9 / 10, 1, 9 / 10, 8 / 10, 7 / 10, 6 / 10, 5 / 10],
                  [5 / 10, 6 / 10, 7 / 10, 8 / 10, 9 / 10, 1, 9 / 10, 8 / 10, 7 / 10, 6 / 10],
                  [4 / 10, 5 / 10, 6 / 10, 7 / 10, 8 / 10, 9 / 10, 1, 9 / 10, 8 / 10, 7 / 10],
                  [3 / 10, 4 / 10, 5 / 10, 6 / 10, 7 / 10, 8 / 10, 9 / 10, 1, 9 / 10, 8 / 10],
                  [2 / 10, 3 / 10, 4 / 10, 5 / 10, 6 / 10, 7 / 10, 8 / 10, 9 / 10, 1, 9 / 10],
                  [1 / 10, 2 / 10, 3 / 10, 4 / 10, 5 / 10, 6 / 10, 7 / 10, 8 / 10, 9 / 10, 1]]

        self.A2 = [[1, 4 / 5, 3 / 5, 2 / 5, 1 / 5],
                   [4 / 5, 1, 4 / 5, 3 / 5, 2 / 5],
                   [3 / 5, 4 / 5, 1, 4 / 5, 3 / 5],
                   [2 / 5, 3 / 5, 4 / 5, 1, 4 / 5],
                   [1 / 5, 2 / 5, 3 / 5, 4 / 5, 1]]
        
        self.A3=[]
        for i in range(85):
            tmp=[]
            for j in range(85):
                tmp.append(1-abs(i-j)/85)
            self.A3.append(tmp)


    def forward(self, input, target):
        length = len(input[0])
        if length == 5:
            A = torch.Tensor(self.A2).cuda()
        elif length == 10:
            A = torch.Tensor(self.A1).cuda()
        else:
            A = torch.Tensor(self.A3).cuda()
        pdtldl = input
        gtldl = target
        losses = torch.zeros(1).cuda()
        temp = abs(pdtldl-gtldl) 
        losses = 0.1*torch.diag(torch.mm(torch.mm(temp, A), torch.transpose(temp, 1, 0))).sum()

        return losses
        