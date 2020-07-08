class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'apollo':
            return r'D:\Shai_Schneider\Apollo_dataset'
            # return r'D:\Shai_Schneider\Apollo_tiny'
        elif dataset == 'farsight':
            return r'D:\Shai_Schneider\farsight_513x513'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
