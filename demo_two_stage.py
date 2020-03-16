#
# demo.py
#
import argparse
import os
import numpy as np
from utils.loss import DepthLosses

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import *
from torchvision.utils import make_grid, save_image


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-path', type=str, required=True, help='image to test')
    parser.add_argument('--out-path', type=str, required=True, help='mask image to save')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')

    parser.add_argument('--ckpt_near', type=str, default='deeplab-resnet.pth',
                        help='saved model')
    parser.add_argument('--ckpt_far', type=str, default='deeplab-resnet.pth',
                        help='saved model')
    parser.add_argument('--ckpt_seg', type=str, default='deeplab-resnet.pth',
                        help='saved model')

    parser.add_argument('--num_class_near', type=int, default=100,
                        help='number of classes to predict')
    parser.add_argument('--num_class_far', type=int, default=50,
                        help='number of classes to predict')

    parser.add_argument('--min_depth_near', type=float, default=0.0,
                        help='min depth to predict')
    parser.add_argument('--min_depth_far', type=float, default=100.0,
                        help='min depth to predict')

    parser.add_argument('--max_depth_near', type=float, default=100.0,
                        help='max depth to predict')
    parser.add_argument('--max_depth_far', type=float, default=655.0,
                        help='max depth to predict')

    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')

    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

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

    print("\nDefine models...\n")
    model_near = DeepLab(num_classes=args.num_class_near,
                         backbone=args.backbone,
                         output_stride=args.out_stride,
                         sync_bn=args.sync_bn,
                         freeze_bn=args.freeze_bn)
    model_far = DeepLab(num_classes=args.num_class_far,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
    model_seg = DeepLab(num_classes=2,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
    if not args.cuda:
        ckpt_near = torch.load(args.ckpt_near, map_location='cpu')
        ckpt_far = torch.load(args.ckpt_far, map_location='cpu')
        ckpt_seg = torch.load(args.ckpt_seg, map_location='cpu')
    else:
        ckpt_near = torch.load(args.ckpt_near)
        ckpt_far = torch.load(args.ckpt_far)
        ckpt_seg = torch.load(args.ckpt_seg)

    print("\nLoad checkpoints...\n")
    model_near.load_state_dict(ckpt_near['state_dict'])
    model_far.load_state_dict(ckpt_far['state_dict'])
    model_seg.load_state_dict(ckpt_seg['state_dict'])

    infer_near = DepthLosses(
        cuda=args.cuda,
        min_depth=args.min_depth_near,
        max_depth=args.max_depth_near,
        num_class=args.num_class_near)

    infer_far = DepthLosses(
        cuda=args.cuda,
        min_depth=args.min_depth_far,
        max_depth=args.max_depth_far,
        num_class=args.num_class_far)

    softmax = nn.Softmax(1)

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    image = Image.open(args.in_path).convert('RGB')
    target = Image.open(args.in_path).convert('L')
    sample = {'image': image, 'label': target}
    tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

    model_near.eval()
    model_far.eval()
    model_seg.eval()
    with torch.no_grad():
        # pass the image in all models
        print("\nPass near model...\n")
        output_near = model_near(tensor_in)
        output_near = infer_near.pred_to_continous_depth(output_near).detach().cpu().numpy().squeeze()
        print("\nPass far model...\n")
        output_far = model_far(tensor_in)
        output_far = infer_far.pred_to_continous_depth(output_far).detach().cpu().numpy().squeeze()
        print("\nPass seg model...\n")
        output_seg = model_seg(tensor_in)

        output_seg = softmax(output_seg)

        output_seg = torch.argmax(output_seg, dim=1).detach().cpu().numpy().squeeze()

        # compose the image based on segmentation mask
        print("\nFuse results...\n")
        output = np.where(output_seg == 0, output_near, output_far)

    plt.imsave(args.out_path, output)
    seg_path = args.out_path.split('.')[0] + "_seg.png"
    near_path = args.out_path.split('.')[0] + "_near.png"
    far_path = args.out_path.split('.')[0] + "_far.png"
    gt_path = args.out_path.split('.')[0] + "_GT.png"
    plt.imsave(seg_path, output_seg)
    plt.imsave(near_path, output_near)
    plt.imsave(far_path, output_far)
    # plt.imsave(gt_path, tensor_in.detach().cpu().numpy().squeeze())


if __name__ == "__main__":
    main()
