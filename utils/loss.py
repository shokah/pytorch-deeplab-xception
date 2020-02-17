import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


class DepthLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False, num_class=250,
                 min_depth=0.0, max_depth=655.0):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.shift = min_depth
        self.bin_size = (max_depth - min_depth) / num_class
        self.num_class = num_class
        self.softmax = nn.Softmax(1)

    def build_loss(self, mode='depth_loss'):
        """Choices: ['depth_loss' or 'depth_lod']"""
        if mode == 'depth_loss':
            return self.DepthLoss
        elif mode == 'depth_lod':
            return self.DepthLODLoss
        else:
            raise NotImplementedError

    def DepthLoss(self, predict, target):
        '''
        scale invariant depth on label space
        :param predict: network output
        :param target: data set label
        :return:
        '''

        lamda = 0.5
        n, c, h, w = predict.size()
        predict = self.pred_to_continous_depth(predict)
        # di = target - predict
        di = torch.log(target) - torch.log(predict)
        k = h * w
        di2 = torch.pow(di, 2)
        first_term = torch.sum(di2, (1, 2)) / k
        second_term = torch.pow(torch.sum(di, (1, 2)), 2) / (k ** 2)
        loss = first_term - lamda * second_term
        if self.batch_average:
            loss /= n
        return loss.mean()

    def pred_to_continous_depth(self, predict):
        # import pdb;
        # pdb.set_trace()
        predict = self.softmax(predict)
        n, c, h, w = predict.size()
        bins = torch.from_numpy(np.arange(0, c)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) \
            .expand(predict.size()).cuda()
        predict = self.shift + self.bin_size * torch.sum(bins * predict, 1)
        return predict

    def pred_to_argmax_depth(self, predict):
        predict = self.softmax(predict)
        predict = torch.argmax(predict, dim=1)
        predict = self.shift + self.bin_size * predict
        return predict

    def DepthLODLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
