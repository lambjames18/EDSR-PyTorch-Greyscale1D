import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.args = args
        self.scale = args.scale
        self.idx_scale = 0
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        if self.cpu:
            self.device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        if args.model.lower() == "restormer":
            args.model = "EDSR"
            pretrain = str(args.pre_train)
            args.pre_train = args.EDSR_path
            EDSR = Model(args, ckp)
            if args.EDSR_path == '':
                raise Exception("EDSR path is empty. The EDSR_Restormer model requires a path to the EDSR model.")
            args.pre_train = pretrain
            module = import_module('model.restormer')
            print("Model import: ", module)
            self.model = module.make_model(EDSR=EDSR, scale=args.scale).to(self.device)
        else:
            module = import_module('model.' + args.model.lower())
            print("Model import: ", module)
            self.model = module.make_model(args).to(self.device)

        if args.precision == 'half':
            self.model.half()

        if args.pre_train != '':
            self.load(
                args.pre_train,
                cpu=args.cpu
            )

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            return forward_function(x)

    # use this to save the model
    def save(self, apath, epoch, is_best=False):
        model_dir = os.path.join(self.args.dir_data, 'model')
        os.makedirs(model_dir, exist_ok=True)

        if is_best:
            save_dirs=os.path.join(model_dir, 'model_best.pt')
        else: 
            save_dirs = os.path.join(model_dir, 'model_{}.pt'.format(epoch))
            
        torch.save(self.model.state_dict(), save_dirs)

    # loads the model when testing on the best model
    # load any model we give it
    # for testing after training, load in the best model
    def load(self, apath="", cpu=False):
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {'map_location': self.device}

        load_from = torch.load(apath, **kwargs)
        self.model.load_state_dict(load_from, strict=False)

    def forward_chop(self, *args, shave=10, min_size=160000):
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        # height, width
        h, w = args[0].size()[-2:]

        top = slice(0, h//2 + shave)
        bottom = slice(h - h//2 - shave, h)
        left = slice(0, w//2 + shave)
        right = slice(w - w//2 - shave, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]

        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                y = P.data_parallel(self.model, *x, range(n_GPUs))
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1: y = y[0]

        return y

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y
