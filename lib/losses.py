import torch.nn as nn

def load_criterion(cfg):
    def _get_criterion(crit):
        if crit == 'mseloss':
            return nn.MSELoss()
        elif crit == 'bcewithlogits':
            return nn.BCEWithLogitsLoss()
        elif crit == 'kldivloss':
            return nn.KLDivLoss()
        else:
            raise ValueError()

    train_crit = cfg['training']['train_criterion']
    test_crit = cfg['training']['test_criterion']

    train_criterion = _get_criterion(train_crit)
    test_criterion = _get_criterion(test_crit)

    return train_criterion, test_criterion