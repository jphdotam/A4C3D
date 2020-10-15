import torch
import torch.nn as nn

from lib.config import load_config
from lib.transforms import load_transforms
from lib.dataset import E32Dataset
from lib.nets.hrnet160 import get_2dnet_cfg, get_seg_model

CONFIG = "./experiments/006.yaml"
FOLD = 1

cfg = load_config(CONFIG)

# Label generation model
cfg_2d = get_2dnet_cfg(cfg)
model_2d = get_seg_model(cfg_2d).cuda()
hrnet_model = nn.DataParallel(model_2d)
hrnet_model.load_state_dict(torch.load(cfg['paths']['2d_model']))

# Data
ds_train = E32Dataset(cfg, cfg['paths']['data_train'], 'train', transforms=None, label_generation_cnn=hrnet_model)
ds_test = E32Dataset(cfg, cfg['paths']['data_test'], 'test', transforms=None, label_generation_cnn=hrnet_model)
