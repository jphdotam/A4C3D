import argparse

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lib.vis import vis_mse
from lib.models import load_model
from lib.config import load_config
from lib.dataset import E32Dataset
from lib.losses import load_criterion
from lib.optimizers import load_optimizer
from lib.logging import get_summary_writer
from lib.transforms import load_transforms
from lib.training import cycle, save_state

import torch.distributed

CONFIG = "./experiments/002.yaml"
cfg = load_config(CONFIG)

# distributed settings
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--ngpu', type=int, default=4)
args = parser.parse_args()

if cfg['training']['data_parallel'] == 'distributed':
    distributed = True
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    world_size = args.ngpu
    torch.distributed.init_process_group('nccl', init_method='env://', world_size=world_size, rank=local_rank)
else:
    distributed = False
    local_rank = None
    world_size = None

# settings
bs_train, bs_test, n_workers = cfg['training']['batch_size_train'], cfg['training']['batch_size_test'], cfg['training']['n_workers']
n_epochs = cfg['training']['n_epochs']
writer = get_summary_writer(cfg, local_rank)
transforms_train, transforms_test = load_transforms(cfg)

# data
ds_train = E32Dataset(cfg, 'train', transforms_train)
ds_test = E32Dataset(cfg, 'test', transforms_test)
sampler_train = DistributedSampler(ds_train, num_replicas=world_size, rank=local_rank) if distributed else None
sampler_test = DistributedSampler(ds_test, num_replicas=world_size, rank=local_rank) if distributed else None
dl_train = DataLoader(ds_train, bs_train, shuffle=False if distributed else True, num_workers=n_workers, pin_memory=True, sampler=sampler_train)
dl_test = DataLoader(ds_test, bs_test, shuffle=False, num_workers=n_workers, pin_memory=True, sampler=sampler_test)

# model
print(f"local rank is {local_rank}")
model, starting_epoch, state = load_model(cfg, local_rank)
optimizer, scheduler = load_optimizer(model, cfg, state, steps_per_epoch=(len(dl_train)))
train_criterion, test_criterion = load_criterion(cfg)

# training
best_loss, best_path, last_save_path = 1e10, None, None

for epoch in range(starting_epoch, n_epochs + 1):
    if local_rank == 0:
        print(f"\nEpoch {epoch} of {n_epochs}")

    # Cycle
    train_loss = cycle('train', model, dl_train, epoch, train_criterion, optimizer, cfg, scheduler, writer, local_rank)
    test_loss = cycle('test', model, dl_test, epoch, test_criterion, optimizer, cfg, scheduler, writer, local_rank)

    # Save state if required
    if local_rank == 0:
        model_weights = model.module.state_dict() if cfg['training']['data_parallel'] else model.state_dict()
        state = {'epoch': epoch + 1,
                 'model': model_weights,
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler}
        save_name = f"{epoch}_{test_loss:.05f}.pt"
        best_loss, last_save_path = save_state(state, save_name, test_loss, best_loss, cfg, last_save_path, lowest_best=True)

        # Vis seg
        vis_mse(ds_test, model, epoch, cfg, writer)

if local_rank == 0:
    save_name = f"FINAL_{epoch}_{test_loss:.05f}.pt"
    best_loss, last_save_path = save_state(state, save_name, test_loss, best_loss, cfg, last_save_path, force=True)
