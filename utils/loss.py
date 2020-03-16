import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class SegmentationLosses(object):
    def __init__(self, weight=None, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight

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
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        return loss.mean()

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
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
    def __init__(self, weight=None, ignore_index=255, cuda=False, num_class=250,
                 min_depth=0.0, max_depth=655.0):
        self.ignore_index = ignore_index
        self.weight = weight
        self.cuda = cuda
        self.shift = min_depth
        self.bin_size = (max_depth - min_depth) / num_class
        self.num_class = num_class
        self.softmax = nn.Softmax(1)
        self.l2_loss = nn.MSELoss()
        k = np.array([[2015.0, 0, 960.0],
                      [0, 2015.0, 540.0],
                      [0, 0, 1]])
        self.k_inv = np.linalg.inv(k)
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.sigmoid = nn.Sigmoid()
        # self.sigmoid = self.my_sigmoid

    def build_loss(self, mode='depth_loss'):
        """Choices: ['depth_loss' or 'depth_lod']"""
        if mode == 'depth_loss':
            return self.DepthLoss
        elif mode == 'depth_pc_loss':
            return self.DepthPCLoss
        elif mode == 'depth_sigmoid_loss':
            return self.DepthSigmoid
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
        predict = self.pred_to_continous_depth(predict)
        # targets that are out of depth range wont affect loss calculation (they will have nan value after log)
        di = torch.log(predict) - torch.log(target)
        k = torch.sum(torch.eq(di, di).float(), (1, 2))  # number of valid pixels
        k[k == 0] = 1  # in case all pixels are out of range
        di[torch.isnan(di)] = 0  # ignore values out of range

        di2 = torch.pow(di, 2)
        first_term = torch.sum(di2, (1, 2)) / k
        second_term = lamda * torch.pow(torch.sum(di, (1, 2)), 2) / (k ** 2)
        loss = first_term - second_term
        return loss.mean()

    def DepthSigmoid(self, predict, target):
        '''
        scale invariant depth on label space
        :param predict: network output
        :param target: data set label
        :return:
        '''
        lamda = 0.5
        # import pdb;pdb.set_trace()
        target = self.depth_to_01(target)
        predict = self.sigmoid(predict.squeeze(1))
        # targets that are out of depth range wont affect loss calculation (they will have nan value after log)
        di = torch.log(predict) - torch.log(target)
        k = torch.sum(torch.eq(di, di).float(), (1, 2))  # number of valid pixels
        k[k == 0] = 1  # in case all pixels are out of range
        di[torch.isnan(di)] = 0  # ignore values out of range

        di2 = torch.pow(di, 2)
        first_term = torch.sum(di2, (1, 2)) / k
        second_term = lamda * torch.pow(torch.sum(di, (1, 2)), 2) / (k ** 2)
        loss = first_term - second_term
        return loss.mean()

    def DepthPCLoss(self, predict, target):
        '''
        depth in 3d space
        :param predict: network output
        :param target: data set label
        :return:
        '''

        predict = self.pred_to_continous_depth(predict)
        x_pred, y_pred = self.depth_to_pointcloud(predict)
        x_target, y_target = self.depth_to_pointcloud(target)

        # k = target.shape[1] * target.shape[2]
        # di = torch.log(predict) - torch.log(target)
        # di2 = torch.pow(di, 2)
        # first_term = torch.sum(di2, (1, 2)) / k
        # second_term = 0.5 * torch.pow(torch.sum(di, (1, 2)), 2) / (k ** 2)
        # loss_z = first_term - second_term

        loss_x = self.l2_loss(x_pred, x_target)
        loss_y = self.l2_loss(y_pred, y_target)
        loss_z = self.l2_loss(predict, target)
        loss = (loss_x + loss_y) / loss_z

        # loss = loss_x + loss_y + loss_z
        # import pdb;
        # pdb.set_trace()
        return loss.mean()

    def pred_to_continous_depth(self, predict):
        predict = self.softmax(predict)
        n, c, h, w = predict.size()
        bins = torch.from_numpy(np.arange(0, c)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(predict.size())
        if self.cuda:
            bins = bins.cuda()
        predict = self.shift + self.bin_size * torch.sum(bins * predict, 1)
        return predict

    def pred_to_argmax_depth(self, predict):
        predict = self.softmax(predict)
        predict = torch.argmax(predict, dim=1)
        predict = self.shift + self.bin_size * predict
        return predict

    def depth_to_pointcloud(self, depth):
        x = torch.linspace(0, depth.shape[2], depth.shape[2]).expand_as(depth)
        y = torch.linspace(0, depth.shape[1], depth.shape[1]).expand_as(depth)
        if self.cuda:
            x = x.cuda()
            y = y.cuda()
        x = depth * (self.k_inv[0, 0] * x + self.k_inv[0, 2])
        y = depth * (self.k_inv[1, 1] * y + self.k_inv[1, 2])
        return x, y

    def depth_to_01(self, depth):
        const = (self.max_depth - self.min_depth)
        depth01 = 1 + (self.min_depth - depth) / const
        return depth01

    def depth01_to_depth(self, depth01):
        const = (self.max_depth - self.min_depth)
        depth = -((depth01 - 1) * const - self.min_depth)
        return depth

    def my_sigmoid(self, x, k=1):
        return 1 / (1 + torch.exp(-k * x))


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
