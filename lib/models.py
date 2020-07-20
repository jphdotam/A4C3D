import torch
import torch.nn as nn

from lib.nets.vnet import VNet
from lib.nets.unet import UNet

def load_model(cfg, local_rank=None):
    arch = cfg['training']['arch']
    mixed_precision = cfg['training'].get('mixed_precision', False)
    data_parallel = cfg['training']['data_parallel']
    n_outputs = len(cfg['data']['labels']['names'])

    if arch == 'vnet':
        batchnormfunc = lambda n_channels: nn.BatchNorm3d(n_channels)
        model = VNet(outChans=n_outputs, mixed_precision=mixed_precision, normfunc=batchnormfunc)
    elif arch == 'vnet2':
        batchnormfunc = lambda n_channels: nn.BatchNorm3d(n_channels)
        model = VNet(outChans=n_outputs, mixed_precision=mixed_precision, multiplier=2, normfunc=batchnormfunc)
    elif arch == 'unet':
        model = UNet(n_channels=1, n_classes=n_outputs)
    else:
        raise ValueError()

    if data_parallel == 'local':
        model = nn.DataParallel(model).to(cfg['training']['device'])
        m = model.module
    elif data_parallel == 'distributed':
        assert local_rank is not None, "if distributed, need a device to be passed"
        device = torch.device(f"cuda:{local_rank}")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        m = model.module
    else:
        m = model.to(cfg['training']['device'])

    if modelpath := cfg['resume'].get('path', None):
        state = torch.load(modelpath)
        m.load_state_dict(state['model'])
        starting_epoch = state['epoch']
        if conf_epoch := cfg['resume'].get('epoch', None):
            print(
                f"WARNING: Loaded model trained for {starting_epoch - 1} epochs but config explicitly overrides to {conf_epoch}")
            starting_epoch = conf_epoch
    else:
        starting_epoch = 1
        state = {}

    return model, starting_epoch, state