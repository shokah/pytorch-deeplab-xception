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
        if args.loss_type == 'depth_loss_two_distributions':
            self.nclass = args.num_class + args.num_class2 + 1
        if args.loss_type == 'depth_avg_sigmoid_class':
            self.nclass = args.num_class + args.num_class2

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
        self.model = model
        if self.args.loss_type == 'depth_multi_dnn':
            model2 = DeepLab(num_classes=args.num_class2,
                             backbone=args.backbone,
                             output_stride=args.out_stride,
                             sync_bn=args.sync_bn,
                             freeze_bn=args.freeze_bn)
            self.model2 = model2
            model_seg = DeepLab(num_classes=2,
                                backbone=args.backbone,
                                output_stride=args.out_stride,
                                sync_bn=args.sync_bn,
                                freeze_bn=args.freeze_bn)
            self.model_seg = model_seg

        if self.args.loss_type == 'depth_with_aprox_depth':
            # add input layer to the model
            self.input_conv = nn.Conv2d(4, 3, 3, padding=1)
            model2 = DeepLab(num_classes=1,
                             backbone=args.backbone,
                             output_stride=args.out_stride,
                             sync_bn=args.sync_bn,
                             freeze_bn=args.freeze_bn)
            self.model2 = model2 # aprox model


        # Define Criterion

        self.infer2 = None
        if self.args.loss_type == 'depth_multi_dnn':
            self.infer = DepthLosses(
                cuda=args.cuda,
                min_depth=args.min_depth,
                max_depth=args.cut_point,
                num_class=args.num_class,
                cut_point=-1,
                num_class2=-1)

            self.infer2 = DepthLosses(
                cuda=args.cuda,
                min_depth=args.cut_point,
                max_depth=args.max_depth,
                num_class=args.num_class2,
                cut_point=-1,
                num_class2=-1)
        else:
            self.infer = DepthLosses(
                cuda=args.cuda,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                num_class=args.num_class,
                cut_point=args.cut_point,
                num_class2=args.num_class2)

        self.softmax = nn.Softmax(1)

        # Define Evaluator
        self.evaluator_depth = EvaluatorDepth(args.batch_size)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
            if self.args.loss_type == 'depth_multi_dnn':
                self.model2 = torch.nn.DataParallel(self.model2, device_ids=self.args.gpu_ids)
                patch_replication_callback(self.model2)
                self.model2 = self.model2.cuda()

                self.model_seg = torch.nn.DataParallel(self.model_seg, device_ids=self.args.gpu_ids)
                patch_replication_callback(self.model_seg)
                self.model_seg = self.model_seg.cuda()

            if self.args.loss_type == 'depth_with_aprox_depth':
                self.input_conv = self.input_conv.cuda()

                self.model = nn.Sequential(
                    self.input_conv,
                    self.model
                )
                self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
                patch_replication_callback(self.model)
                self.model = self.model.cuda()

                self.model2 = torch.nn.DataParallel(self.model2, device_ids=self.args.gpu_ids)
                patch_replication_callback(self.model2)
                self.model2 = self.model2.cuda()

        if not args.cuda:
            ckpt = torch.load(args.ckpt, map_location='cpu')
            if self.args.loss_type == 'depth_multi_dnn':
                ckpt2 = torch.load(args.ckpt2, map_location='cpu')
                ckpt_seg = torch.load(args.ckpt_seg, map_location='cpu')
                self.model2.load_state_dict(ckpt2['state_dict'])
                self.model_seg.load_state_dict(ckpt_seg['state_dict'])

            if self.args.loss_type == 'depth_with_aprox_depth':
                ckpt2 = torch.load(args.ckpt2, map_location='cpu')
                self.model2.load_state_dict(ckpt2['state_dict'])

            self.model.load_state_dict(ckpt['state_dict'])
        else:
            ckpt = torch.load(args.ckpt)
            if self.args.loss_type == 'depth_multi_dnn':
                ckpt2 = torch.load(args.ckpt2)
                ckpt_seg = torch.load(args.ckpt_seg)
                self.model2.module.load_state_dict(ckpt2['state_dict'])
                self.model_seg.load_state_dict(ckpt_seg['state_dict'])

            if self.args.loss_type == 'depth_with_aprox_depth':
                ckpt2 = torch.load(args.ckpt2)
                self.model2.module.load_state_dict(ckpt2['state_dict'])

            self.model.module.load_state_dict(ckpt['state_dict'])

        print("\nLoad checkpoints...\n")

    def evaluate(self):
        self.model.eval()
        if self.args.loss_type == 'depth_multi_dnn':
            self.model2.eval()
            self.model_seg.eval()
        if self.args.loss_type == 'depth_with_aprox_depth':
            self.model2.eval()
        self.evaluator_depth.reset()
        tbar = tqdm(self.test_loader, desc='\r')

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                if 'depth_with_aprox_depth' in self.args.loss_type:
                    # import pdb;pdb.set_trace()
                    aprox_depth = self.model2(image)
                    aprox_depth = self.infer.sigmoid(aprox_depth)
                    input = torch.cat([image, aprox_depth], dim=1)
                    output = self.model(input)
                else:
                    output = self.model(image)

            pred = None
            if 'depth' in self.args.loss_type:
                if self.args.loss_type == 'depth_loss':
                    pred = self.infer.pred_to_continous_depth(output)
                elif self.args.loss_type == 'depth_avg_sigmoid_class':
                    pred = self.infer.pred_to_continous_depth_avg(output)
                elif self.args.loss_type == 'depth_loss_combination':
                    pred = self.infer.pred_to_continous_combination(output)
                elif self.args.loss_type == 'depth_loss_two_distributions':
                    pred = self.infer.pred_to_continous_depth_two_distributions(output)
                elif self.args.loss_type == 'depth_multi_dnn':
                    with torch.no_grad():
                        output2 = self.model2(image)
                        output_seg = self.model_seg(image)
                    pred = self.infer.pred_to_continous_depth(output)
                    pred2 = self.infer2.pred_to_continous_depth(output2)
                    pred_seg = self.softmax(output_seg)
                    # join results
                    pred_seg = torch.argmax(pred_seg, dim=1)
                    pred = torch.where(pred_seg == 0, pred, pred2)

                elif 'depth_sigmoid_loss' in self.args.loss_type:
                    output = self.infer.sigmoid(output.squeeze(1))
                    if 'inverse' in self.args.loss_type:
                        pred = self.infer.depth01_to_depth(output, True)
                    else:
                        pred = self.infer.depth01_to_depth(output)

                elif 'depth_with_aprox_depth' in self.args.loss_type:
                    output = self.infer.sigmoid(output.squeeze(1))
                    pred = self.infer.depth01_to_depth(output)


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

    def evaluate2stage(self):
        self.model.eval()
        self.model2.eval()
        self.model_seg.eval()
        self.evaluator_depth.reset()
        tbar = tqdm(self.test_loader, desc='\r')

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                output2 = self.model2(image)
                output_seg = self.model_seg(image)
            if self.infer.num_class > 1:
                pred = self.infer.pred_to_continous_depth(output)
                if self.infer2 is not None:
                    pred2 = self.infer2.pred_to_continous_depth(output2)
                    pred_seg = self.softmax(output_seg)
                    # join results
                    pred_seg = torch.argmax(pred_seg, dim=1)
                    pred = torch.where(pred_seg == 0, pred, pred2)
            else:
                output = self.infer.sigmoid(output)
                pred = self.infer.depth01_to_depth(output).detach().cpu().numpy().squeeze()

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
            "MSE:{}, RMSE:{}, ABS_REL:{}, LG10: {}\nMAE:{}, DELTA1:{}, DELTA2:{}, DELTA3: {}".format(MSE, RMSE,
                                                                                                     ABS_REL,
                                                                                                     LG10, MAE,
                                                                                                     DELTA1,
                                                                                                     DELTA2,
                                                                                                     DELTA3))


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'apollo', 'farsight'],
                        help='dataset name (default: pascal)')

    parser.add_argument('--num_class', type=int, default=100,
                        help='number of wanted classes')
    parser.add_argument('--num_class2', type=int, default=-1,
                        help='number of wanted classes')

    parser.add_argument('--loss-type', type=str, default='depth_loss',
                        choices=['depth_loss', 'depth_pc_loss', 'depth_sigmoid_loss',
                                 'depth_sigmoid_grad_loss', 'depth_loss_two_distributions',
                                 'depth_sigmoid_loss_inverse', 'depth_avg_sigmoid_class', 'depth_loss_combination',
                                 'depth_multi_dnn', 'depth_with_aprox_depth'],
                        help='loss func type - affect they way nn output is converted to depth')

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
    parser.add_argument('--cut_point', type=float, default=-1,
                        help='beyond this value depth is considered far')
    # parser.add_argument('--min_depth2', type=float, default=-1,
    #                     help='min depth to predict')
    # parser.add_argument('--max_depth2', type=float, default=-1,
    #                     help='max depth to predict')

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
    parser.add_argument('--ckpt2', type=str, default=None, required=False,
                        help='put the path to resuming file if needed if using 2 stage prediction')
    parser.add_argument('--ckpt_seg', type=str, default=None, required=False,
                        help='put the path to resuming file if needed if using 2 stage prediction')

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
