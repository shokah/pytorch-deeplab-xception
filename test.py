import argparse
import os
from tqdm import tqdm
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import DepthLosses
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import EvaluatorDepth


class Eval(object):
    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        _, _, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        # Define Criterion
        self.infer = DepthLosses(
            cuda=args.cuda,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            num_class=args.num_class)
        self.model = model

        # Define Evaluator
        self.evaluator_depth = EvaluatorDepth(args.batch_size)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        if not args.cuda:
            ckpt = torch.load(args.ckpt, map_location='cpu')
        else:
            ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {}) (best rmse {})"
              .format(args.ckpt, ckpt['epoch'], ckpt['best_pred']))

    def evaluate(self):
        self.model.eval()
        self.evaluator_depth.reset()
        tbar = tqdm(self.test_loader, desc='\r')

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)

            pred = self.infer.pred_to_continous_depth(output)
            # Add batch sample into evaluator
            self.evaluator_depth.evaluateError(pred, target)

        # Fast test during the training
        MSE = self.evaluator_depth.averageError['MSE']
        RMSE = self.evaluator_depth.averageError['RMSE']
        ABS_REL = self.evaluator_depth.averageError['ABS_REL']
        LG10 = self.evaluator_depth.averageError['LG10']
        MAE = self.evaluator_depth.averageError['MAE']
        DELTA1 = self.evaluator_depth.averageError['DELTA1']
        DELTA2 = self.evaluator_depth.averageError['DELTA2']
        DELTA3 = self.evaluator_depth.averageError['DELTA3']

        print('Test:')
        print(
            "MSE:{}, RMSE:{}, ABS_REL:{}, LG10: {}\nMAE:{}, DELTA1:{}, DELTA2:{}, DELTA3: {}".format(MSE, RMSE, ABS_REL,
                                                                                                     LG10, MAE, DELTA1,
                                                                                                     DELTA2, DELTA3))


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'apollo'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--num_class', type=int, default=100,
                        help='number of wanted classes')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    testing (default: auto)')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='min depth to predict')
    parser.add_argument('--max_depth', type=float, default=655,
                        help='max depth to predict')
    parser.add_argument('--task', type=str, default='depth',
                        help='depth or segmentation')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    # checking point
    parser.add_argument('--ckpt', type=str, default=None, required=True,
                        help='put the path to resuming file if needed')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    print(args)

    tester = Eval(args)
    tester.evaluate()


if __name__ == "__main__":
    main()
