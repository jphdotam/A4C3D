from torch.utils.tensorboard import SummaryWriter


def get_summary_writer(cfg, local_rank=None):
    if local_rank:
        return None
    log_dir = cfg['paths']['logs']
    if not cfg['output']['use_tensorboard']:
        return None
    else:
        return SummaryWriter(log_dir=log_dir)
