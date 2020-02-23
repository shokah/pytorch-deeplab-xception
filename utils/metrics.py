import numpy as np
import math
import torch


class Evaluator(object):
    ''' evaluator for segmentation '''

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class EvaluatorDepth(object):
    ''' evaluator for depth '''

    def __init__(self, batch_size):
        self.averageError = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                             'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
        self.errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                         'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
        self.batch_size = batch_size
        self.total_number = 0

    def reset(self):
        self.averageError = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                             'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
        self.errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                         'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
        self.total_number = 0

    def lg10(self, x):
        return torch.div(torch.log(x), math.log(10))

    def maxOfTwo(self, x, y):
        z = x.clone()
        maskYLarger = torch.lt(x, y)
        z[maskYLarger.detach()] = y[maskYLarger.detach()]
        return z

    def nValid(self, x):
        return torch.sum(torch.eq(x, x).float())

    def nNanElement(self, x):
        return torch.sum(torch.ne(x, x).float())

    def getNanMask(self, x):
        return torch.ne(x, x)

    def setNanToZero(self, input, target):
        # target[target == -1] = torch.tensor(float('nan'))
        nanMask = self.getNanMask(target)
        nValidElement = self.nValid(target)

        _input = input.clone()
        _target = target.clone()

        _input[nanMask] = 0
        _target[nanMask] = 0

        return _input, _target, nanMask, nValidElement

    def evaluateError(self, output, target):
        errors = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                  'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

        _output, _target, nanMask, nValidElement = self.setNanToZero(output, target)

        if (nValidElement.data.cpu().numpy() > 0):
            diffMatrix = torch.abs(_output - _target)

            errors['MSE'] = torch.sum(torch.pow(diffMatrix, 2)) / nValidElement
            errors['RMSE'] = torch.sqrt(errors['MSE'])

            errors['MAE'] = torch.sum(diffMatrix) / nValidElement

            realMatrix = torch.div(diffMatrix, _target)
            realMatrix[nanMask] = 0
            errors['ABS_REL'] = torch.sum(realMatrix) / nValidElement

            LG10Matrix = torch.abs(self.lg10(_output) - self.lg10(_target))
            LG10Matrix[nanMask] = 0
            errors['LG10'] = torch.sum(LG10Matrix) / nValidElement

            yOverZ = torch.div(_output, _target)
            zOverY = torch.div(_target, _output)

            maxRatio = self.maxOfTwo(yOverZ, zOverY)

            errors['DELTA1'] = torch.sum(
                torch.le(maxRatio, 1.25).float()) / nValidElement
            errors['DELTA2'] = torch.sum(
                torch.le(maxRatio, math.pow(1.25, 2)).float()) / nValidElement
            errors['DELTA3'] = torch.sum(
                torch.le(maxRatio, math.pow(1.25, 3)).float()) / nValidElement

            errors['MSE'] = float(errors['MSE'].data.cpu().numpy())
            errors['RMSE'] = float(errors['RMSE'].data.cpu().numpy())
            errors['ABS_REL'] = float(errors['ABS_REL'].data.cpu().numpy())
            errors['LG10'] = float(errors['LG10'].data.cpu().numpy())
            errors['MAE'] = float(errors['MAE'].data.cpu().numpy())
            errors['DELTA1'] = float(errors['DELTA1'].data.cpu().numpy())
            errors['DELTA2'] = float(errors['DELTA2'].data.cpu().numpy())
            errors['DELTA3'] = float(errors['DELTA3'].data.cpu().numpy())

        self.addErrors(errors)
        self.averageErrors()

    def addErrors(self, errors):
        self.total_number += self.batch_size
        self.errorSum['MSE'] += errors['MSE'] * self.batch_size
        self.errorSum['RMSE'] += errors['RMSE'] * self.batch_size
        self.errorSum['ABS_REL'] += errors['ABS_REL'] * self.batch_size
        self.errorSum['LG10'] += errors['LG10'] * self.batch_size
        self.errorSum['MAE'] += errors['MAE'] * self.batch_size

        self.errorSum['DELTA1'] += errors['DELTA1'] * self.batch_size
        self.errorSum['DELTA2'] += errors['DELTA2'] * self.batch_size
        self.errorSum['DELTA3'] += errors['DELTA3'] * self.batch_size

    def averageErrors(self):
        self.averageError['MSE'] = self.errorSum['MSE'] / self.total_number
        self.averageError['ABS_REL'] = self.errorSum['ABS_REL'] / self.total_number
        self.averageError['LG10'] = self.errorSum['LG10'] / self.total_number
        self.averageError['MAE'] = self.errorSum['MAE'] / self.total_number

        self.averageError['DELTA1'] = self.errorSum['DELTA1'] / self.total_number
        self.averageError['DELTA2'] = self.errorSum['DELTA2'] / self.total_number
        self.averageError['DELTA3'] = self.errorSum['DELTA3'] / self.total_number
        self.averageError['RMSE'] = self.errorSum['RMSE'] / self.total_number
