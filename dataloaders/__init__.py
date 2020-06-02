from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, apollo, farsight
from torch.utils.data import DataLoader, random_split


def make_data_loader(args, **kwargs):
    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'apollo' or args.dataset == 'apollo_seg':
        num_class = args.num_class
        min_depth = args.min_depth
        max_depth = args.max_depth
        train_n_val = apollo.ApolloDepthSegmentation(args, split='train', num_class=num_class,
                                                     min_depth=min_depth, max_depth=max_depth)
        n_train = int(train_n_val.__len__() * 0.8)
        n_val = train_n_val.__len__() - n_train
        train_set, val_set = random_split(train_n_val, [n_train, n_val])
        # val_set = apollo.ApolloDepthSegmentation(args, split='val', num_class=num_class,
        #                                          min_depth=min_depth, max_depth=max_depth)
        test_set = apollo.ApolloDepthSegmentation(args, split='test', num_class=num_class,
                                                  min_depth=min_depth, max_depth=max_depth)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'farsight' or args.dataset== 'farsight_seg':

        num_class = args.num_class
        min_depth = args.min_depth
        max_depth = args.max_depth
        train_n_val = farsight.FarsightDepthSegmentation(args, split='train', num_class=num_class,
                                                         min_depth=min_depth, max_depth=max_depth)
        n_train = int(train_n_val.__len__() * 0.8)
        n_val = train_n_val.__len__() - n_train
        train_set, val_set = random_split(train_n_val, [n_train, n_val])
        # val_set = farsight.FarsightDepthSegmentation(args, split='val', num_class=num_class,
        #                                          min_depth=min_depth, max_depth=max_depth)
        test_set = farsight.FarsightDepthSegmentation(args, split='test', num_class=num_class,
                                                      min_depth=min_depth, max_depth=max_depth)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError
