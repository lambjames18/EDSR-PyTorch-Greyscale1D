from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            if type(args.data_train) is str:
                args.data_train = [args.data_train]
            for module_name in args.data_train:
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=module_name))

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                #num_workers=args.n_threads,
                # for testing
                num_workers= 0,
            )

        # for when we load our own test
        self.loader_test = []
        if type(args.data_test) is str:
            args.data_test = [args.data_test]
        for module_name in args.data_test:
            m = import_module('data.' + module_name.lower())
            testset = getattr(m, module_name)(args, train=False, name=module_name)

        self.loader_test.append(
            dataloader.DataLoader(
                testset,
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                #num_workers=args.n_threads,
                # for testing
                num_workers= 0,
            )
        )
