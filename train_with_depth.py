import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses, DepthLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator, EvaluatorDepth


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
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


        print("\nDefine models...\n")
        self.model_aprox_depth = DeepLab(num_classes=1,
                             backbone=args.backbone,
                             output_stride=args.out_stride,
                             sync_bn=args.sync_bn,
                             freeze_bn=args.freeze_bn)

        self.input_conv = nn.Conv2d(4, 3, 3, padding=1)
        # Using cuda
        if args.cuda:
            self.model_aprox_depth = torch.nn.DataParallel(self.model_aprox_depth, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model_aprox_depth)
            self.model_aprox_depth = self.model_aprox_depth.cuda()
            self.input_conv = self.input_conv.cuda()


        print("\nLoad checkpoints...\n")
        if not args.cuda:
            ckpt_aprox_depth = torch.load(args.ckpt, map_location='cpu')
            self.model_aprox_depth.load_state_dict(ckpt_aprox_depth['state_dict'])
        else:
            ckpt_aprox_depth = torch.load(args.ckpt)
            self.model_aprox_depth.module.load_state_dict(ckpt_aprox_depth['state_dict'])


        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        # set optimizer
        optimizer = torch.optim.Adam(train_params, args.lr)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        if 'depth' in args.loss_type:
            self.criterion = DepthLosses(weight=weight,
                                         cuda=args.cuda,
                                         min_depth=args.min_depth,
                                         max_depth=args.max_depth,
                                         num_class=args.num_class,
                                         cut_point=args.cut_point,
                                         num_class2=args.num_class2).build_loss(mode=args.loss_type)
            self.infer = DepthLosses(weight=weight,
                                     cuda=args.cuda,
                                     min_depth=args.min_depth,
                                     max_depth=args.max_depth,
                                     num_class=args.num_class,
                                     cut_point=args.cut_point,
                                     num_class2=args.num_class2)
            self.evaluator_depth = EvaluatorDepth(args.batch_size)
        else:
            self.criterion = SegmentationLosses(cuda=args.cuda, weight=weight).build_loss(mode=args.loss_type)
            self.evaluator = Evaluator(self.nclass)

        self.model, self.optimizer = model, optimizer

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        if 'depth' in args.loss_type:
            self.best_pred = 1e6
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            if not args.cuda:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                state_dict = checkpoint['state_dict']
                state_dict.popitem(last=True)
                state_dict.popitem(last=True)
                self.model.module.load_state_dict(state_dict, strict=False)
            else:
                state_dict = checkpoint['state_dict']
                state_dict.popitem(last=True)
                state_dict.popitem(last=True)
                self.model.load_state_dict(state_dict, strict=False)
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            if 'depth' in args.loss_type:
                self.best_pred = 1e6
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        # add input layer to the model
        self.model = nn.Sequential(
            self.input_conv,
            self.model
        )
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        self.model_aprox_depth.eval()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.dataset == 'apollo_seg' or self.args.dataset == 'farsight_seg':
                target[target <= self.args.cut_point] = 0
                target[target > self.args.cut_point] = 1
            if image.shape[0] == 1:
                target = torch.cat([target, target], dim=0)
                image = torch.cat([image, image], dim=0)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            aprox_depth = self.model_aprox_depth(image)
            aprox_depth = self.infer.sigmoid(aprox_depth)
            input = torch.cat([image, aprox_depth], dim=1)
            output = self.model(input)
            if self.args.loss_type == 'depth_sigmoid_loss_inverse':
                loss = self.criterion(output, target, inverse=True)
            else:
                loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                target[
                    torch.isnan(target)] = 0  # change nan values to zero for display (handle warning from tensorboard)
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step,
                                             n_class=self.args.num_class)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.model_aprox_depth.eval()
        if 'depth' in self.args.loss_type:
            self.evaluator_depth.reset()
        else:
            softmax = nn.Softmax(1)
            self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                aprox_depth = self.model_aprox_depth(image)
                aprox_depth = self.infer.sigmoid(aprox_depth)
                input = torch.cat([image, aprox_depth], dim=1)
                output = self.model(input)
                loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Val loss: %.3f' % (test_loss / (i + 1)))
            if 'depth' in self.args.loss_type:
                if self.args.loss_type == 'depth_loss':
                    pred = self.infer.pred_to_continous_depth(output)
                elif self.args.loss_type == 'depth_avg_sigmoid_class':
                    pred = self.infer.pred_to_continous_depth_avg(output)
                elif self.args.loss_type == 'depth_loss_combination':
                    pred = self.infer.pred_to_continous_combination(output)
                elif self.args.loss_type == 'depth_loss_two_distributions':
                    pred = self.infer.pred_to_continous_depth_two_distributions(output)
                elif 'depth_sigmoid_loss' in self.args.loss_type:
                    output = self.infer.sigmoid(output.squeeze(1))
                    pred = self.infer.depth01_to_depth(output)
                # Add batch sample into evaluator
                self.evaluator_depth.evaluateError(pred, target)
            else:
                output = softmax(output)
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                # Add batch sample into evaluator
                self.evaluator.add_batch(target, pred)
        if 'depth' in self.args.loss_type:
            # Fast test during the training
            MSE = self.evaluator_depth.averageError['MSE']
            RMSE = self.evaluator_depth.averageError['RMSE']
            ABS_REL = self.evaluator_depth.averageError['ABS_REL']
            LG10 = self.evaluator_depth.averageError['LG10']
            MAE = self.evaluator_depth.averageError['MAE']
            DELTA1 = self.evaluator_depth.averageError['DELTA1']
            DELTA2 = self.evaluator_depth.averageError['DELTA2']
            DELTA3 = self.evaluator_depth.averageError['DELTA3']

            self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
            self.writer.add_scalar('val/MSE', MSE, epoch)
            self.writer.add_scalar('val/RMSE', RMSE, epoch)
            self.writer.add_scalar('val/ABS_REL', ABS_REL, epoch)
            self.writer.add_scalar('val/LG10', LG10, epoch)

            self.writer.add_scalar('val/MAE', MAE, epoch)
            self.writer.add_scalar('val/DELTA1', DELTA1, epoch)
            self.writer.add_scalar('val/DELTA2', DELTA2, epoch)
            self.writer.add_scalar('val/DELTA3', DELTA3, epoch)

            print('Validation:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            print(
                "MSE:{}, RMSE:{}, ABS_REL:{}, LG10: {}\nMAE:{}, DELTA1:{}, DELTA2:{}, DELTA3: {}".format(MSE, RMSE,
                                                                                                         ABS_REL,
                                                                                                         LG10, MAE,
                                                                                                         DELTA1,
                                                                                                         DELTA2,
                                                                                                         DELTA3))
            new_pred = RMSE
            if new_pred < self.best_pred:
                is_best = True
                self.best_pred = new_pred
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best)
        else:
            # Fast test during the training
            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
            self.writer.add_scalar('val/mIoU', mIoU, epoch)
            self.writer.add_scalar('val/Acc', Acc, epoch)
            self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
            self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
            print('Validation:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
            print('Loss: %.3f' % test_loss)

            new_pred = mIoU
            if new_pred > self.best_pred:
                is_best = True
                self.best_pred = new_pred
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best)

        print('Loss: %.3f' % test_loss)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'apollo', 'apollo_seg', 'farsight', 'farsight_seg'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--num_class', type=int, default=100,
                        help='number of wanted classes')
    parser.add_argument('--num_class2', type=int, default=-1,
                        help='number of wanted classes for far network')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='depth_loss',
                        choices=['ce', 'focal', 'depth_loss', 'depth_pc_loss', 'depth_sigmoid_loss',
                                 'depth_sigmoid_grad_loss', 'depth_loss_two_distributions',
                                 'depth_sigmoid_loss_inverse', 'depth_avg_sigmoid_class', 'depth_loss_combination'],
                        help='loss func type (default: ce)')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='min depth to predict')
    parser.add_argument('--max_depth', type=float, default=656.0,
                        help='max depth to predict')
    parser.add_argument('--cut_point', type=float, default=-1,
                        help='beyond this value depth is considered far')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # aprox depth cnn path
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

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
