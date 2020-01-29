import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr


class ApolloDepthSegmentation(data.Dataset):

    def __init__(self, args, root=Path.db_root_dir('apollo'), split="train", NUM_CLASSES=250, min_depth=5.0,
                 max_depth=655.0, split_method='sid'):
        self.NUM_CLASSES = NUM_CLASSES
        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.split_method = split_method

        self.images_base = os.path.join(self.root, self.split, 'RGB')
        self.annotations_base = os.path.join(self.root, self.split, 'Depth')
        self.mapping = self.depth_class_spliter(self.NUM_CLASSES, self.min_depth, self.max_depth, self.split_method)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.jpg')

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        path = img_path.split(os.sep)[-8:]
        path[-1] = path[-1].split('.')[0] + '.png'
        lbl_path = os.path.join(self.annotations_base,
                                os.sep.join(path))

        _img = Image.open(img_path).convert('RGB')
        if not os.path.exists(lbl_path):
            os.remove(img_path)
            print(f"{img_path} was deleted")
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.decode_apollo(_tmp)
        _tmp = self.classify_depth(_tmp, self.mapping)  # convert continuous depth to discrete depth
        # _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    @staticmethod
    def depth_class_spliter(n_class, min_depth, max_depth, method='ud'):
        def ud(n_class, min_depth, max_depth):
            for j in range(n_class - 1):
                t = np.linspace(min_depth, max_depth, n_class)
            return t

        def sid(n_class, min_depth, max_depth):
            m = []
            c = n_class - 1
            for j in range(n_class - 2):
                t = np.exp(np.log(min_depth) + ((j * np.log(max_depth / min_depth)) / c))
                m.append(t)
            m = np.array(m)
            return m

        if method == 'ud':  # uniform distribution
            return ud(n_class, min_depth, max_depth)
        elif method == 'sid':
            return sid(n_class, min_depth, max_depth)

    @staticmethod
    def classify_depth(depth, mapping):
        dig_depth = np.digitize(depth, mapping).astype(np.uint8)
        return dig_depth

    def decode_apollo(self, depth_im):
        R = depth_im[:, :, 0] / 255
        G = depth_im[:, :, 1] / 255
        real_depth = (R + G / 255.0) * 655.36
        return real_depth


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    # apollo = ApolloDepthSegmentation(args, split='train')
    # apollo = ApolloDepthSegmentation(args, split='val')
    apollo = ApolloDepthSegmentation(args, split='test')


    dataloader = DataLoader(apollo, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='apollo')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
