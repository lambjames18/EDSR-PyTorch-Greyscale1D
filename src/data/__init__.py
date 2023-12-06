from importlib import import_module
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
    # new way of loading data will return only one list of the high and low res
    def __init__(self, args):
        self.total_loader = []

        if type(args.data_test) is str:
            args.data_test = [args.data_test]

        for module_name in args.data_test:
            m = import_module('data.' + module_name.lower())
            testset = getattr(m, module_name)(args, train=False, name=module_name)
        
        self.total_loader = dataloader.DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=not args.cpu,
            #num_workers=args.n_threads,
            # for testing
            num_workers= 0,
        )



