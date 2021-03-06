import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


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
                 min_depth=0.0, max_depth=655.0, cut_point=-1, num_class2=-1):
        '''

        :param weight:
        :param ignore_index:
        :param cuda:
        :param num_class: total number of classes
        :param min_depth: min of all the range
        :param max_depth: max of all the range
        :param cut_point:
        :param num_class2: number of classes of the far part
        '''
        self.ignore_index = ignore_index
        self.weight = weight
        self.cuda = cuda
        self.shift = min_depth
        if cut_point == -1 and num_class2 == -1:
            self.bin_size = (max_depth - min_depth) / num_class
            self.num_class = num_class
            self.max_depth = max_depth
            self.min_depth = min_depth
        else:
            self.max_depth = cut_point
            self.min_depth = min_depth
            self.num_class = num_class
            self.bin_size = (cut_point - min_depth) / self.num_class

            self.shift2 = cut_point
            self.bin_size2 = (max_depth - cut_point) / num_class2
            self.num_class2 = num_class2
            self.max_depth2 = max_depth
            self.min_depth2 = cut_point


        self.softmax = nn.Softmax(1)
        self.l2_loss = nn.MSELoss()
        k = np.array([[2015.0, 0, 960.0],
                      [0, 2015.0, 540.0],
                      [0, 0, 1]])
        self.k_inv = np.linalg.inv(k)

        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=1, eps=0)
        self.get_gradient = Sobel()
        if self.cuda:
            self.get_gradient = self.get_gradient.cuda()
        # self.sigmoid = self.my_sigmoid

    def build_loss(self, mode='depth_loss'):
        """Choices: ['depth_loss' or 'depth_lod']"""
        if mode == 'depth_loss':
            return self.DepthLoss
        if mode == 'depth_loss_two_distributions':
            return self.DepthLoss2Distributions
        elif mode == 'depth_pc_loss':
            return self.DepthPCLoss
        elif mode == 'depth_sigmoid_loss' or mode == 'depth_sigmoid_loss_inverse':
            return self.DepthSigmoid
        elif mode == 'depth_sigmoid_grad_loss':
            return self.DepthGradSigmoid
        elif mode == 'depth_avg_sigmoid_class':
            return self.DepthAvgSigmoidClass
        elif mode == 'depth_loss_combination':
            return self.DepthLossCombination
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
        di[torch.isinf(di)] = 0  # ignore values out of range

        di2 = torch.pow(di, 2)
        first_term = torch.sum(di2, (1, 2)) / k
        second_term = lamda * torch.pow(torch.sum(di, (1, 2)), 2) / (k ** 2)
        loss = first_term - second_term

        return loss.mean()

    def DepthLoss2Distributions(self, predict, target):
        '''
        scale invariant depth on label space
        :param predict: network output
        :param target: data set label
        :return:
        '''
        lamda = 0.5
        predict_joined = self.pred_to_continous_depth_two_distributions(predict)
        # import pdb;
        # pdb.set_trace()
        # targets that are out of depth range wont affect loss calculation (they will have nan value after log)
        di = torch.log(predict_joined) - torch.log(target)
        k = torch.sum(torch.eq(di, di).float(), (1, 2))  # number of valid pixels
        k[k == 0] = 1  # in case all pixels are out of range
        di[torch.isnan(di)] = 0  # ignore values out of range
        di[torch.isinf(di)] = 0  # ignore values out of range

        di2 = torch.pow(di, 2)
        first_term = torch.sum(di2, (1, 2)) / k
        second_term = lamda * torch.pow(torch.sum(di, (1, 2)), 2) / (k ** 2)
        loss = first_term - second_term

        return loss.mean()

    def DepthLossCombination(self, predict, target):
        '''
        scale invariant depth on label space
        :param predict: network output
        :param target: data set label
        :return:
        '''
        lamda = 0.5
        predict = self.pred_to_continous_combination(predict)
        # import pdb;
        # pdb.set_trace()
        # targets that are out of depth range wont affect loss calculation (they will have nan value after log)
        di = torch.log(predict) - torch.log(target)
        k = torch.sum(torch.eq(di, di).float(), (1, 2))  # number of valid pixels
        k[k == 0] = 1  # in case all pixels are out of range
        di[torch.isnan(di)] = 0  # ignore values out of range
        di[torch.isinf(di)] = 0  # ignore values out of range

        di2 = torch.pow(di, 2)
        first_term = torch.sum(di2, (1, 2)) / k
        second_term = lamda * torch.pow(torch.sum(di, (1, 2)), 2) / (k ** 2)
        loss = first_term - second_term

        return loss.mean()

    def DepthAvgSigmoidClass(self, predict, target, inverse=False):
        '''
        scale invariant depth on label space
        :param predict: network output
        :param target: data set label
        :return:
        '''
        lamda = 0.5
        # import pdb;pdb.set_trace()
        predict = self.pred_to_continous_depth_avg(predict, inverse)
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

    def DepthSigmoid(self, predict, target, inverse=False):
        '''
        scale invariant depth on label space
        :param predict: network output
        :param target: data set label
        :return:
        '''
        lamda = 0.5
        # import pdb;pdb.set_trace()
        target = self.depth_to_01(target, inverse)
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

    def DepthGradSigmoid(self, predict, target):
        '''
        revisiting depth estimation loss function
        :param depth: depth label # target
        :param output: CNN output # predict
        :return: loss values
        '''
        loss_depth = self.DepthSigmoid(predict, target)
        target = target.unsqueeze(1)

        depth_grad = self.get_gradient(target).squeeze(1)
        output_grad = self.get_gradient(predict).squeeze(1)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(predict)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(predict)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(predict)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(predict)

        loss_dx = self.sigmoid(output_grad_dx) - self.sigmoid(depth_grad_dx)
        loss_dy = self.sigmoid(output_grad_dy) - self.sigmoid(depth_grad_dy)

        loss_grad = (loss_dx + loss_dy).mean()
        loss = loss_depth + loss_grad
        # import pdb;pdb.set_trace()
        return loss

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
        bins = torch.from_numpy(np.arange(0, c, dtype=np.float)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            predict.size())
        if self.cuda:
            bins = bins.cuda()
        predict = self.shift + self.bin_size * torch.sum(bins * predict, 1)
        return predict

    def pred_to_continous_depth2(self, predict):
        predict = self.softmax(predict)
        n, c, h, w = predict.size()
        bins = torch.from_numpy(np.arange(0, c, dtype=np.float)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            predict.size())
        if self.cuda:
            bins = bins.cuda()
        predict = self.shift2 + self.bin_size2 * torch.sum(bins * predict, 1)
        return predict

    def pred_to_continous_depth_avg(self, predict, inverse=False):
        pred_sig = self.depth01_to_depth(self.sigmoid(predict[:, -1, :, :]), inverse)
        pred_class = self.pred_to_continous_depth(predict[:, :-1, :, :])
        predict = 0.5 * (pred_sig + pred_class)
        return predict

    def pred_to_continous_depth_two_distributions(self, predict):
        predict_near = self.pred_to_continous_depth(predict[:, :self.num_class, :, :])
        predict_far = self.pred_to_continous_depth2(predict[:, self.num_class:self.num_class + self.num_class2, :, :])
        seg = self.sigmoid(predict[:, -1, :, :])
        predict_near[seg >= 0.5] = 0
        predict_far[seg < 0.5] = 0
        predict_joined = predict_near + predict_far
        return predict_joined

    def pred_to_continous_combination(self, predict):
        predict = self.sigmoid(predict)
        # import pdb;
        # pdb.set_trace()
        bins = torch.from_numpy(np.array([1.0, 10.0, 100.0], dtype=np.float)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            predict.size())
        if self.cuda:
            bins = bins.cuda()
        predict = torch.sum(bins * predict, 1)
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

    def depth_to_01(self, depth, inverse=False):
        const = (self.max_depth - self.min_depth)
        depth01 = (depth - self.min_depth) / const
        if inverse:
            depth01 = -depth01 + 1
        return depth01

    def depth01_to_depth(self, depth01, inverse=False):
        const = (self.max_depth - self.min_depth)
        if inverse:
            depth01 = -depth01 + 1
        depth = depth01 * const + self.min_depth
        return depth


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
