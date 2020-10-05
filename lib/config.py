import os
import yaml

def load_config(configpath):
    with open(configpath) as f:
        cfg = yaml.safe_load(f)

    experiment_id = os.path.splitext(os.path.basename(configpath))[0]
    cfg['experiment_id'] = experiment_id

    cfg['paths']['models'] = os.path.join(cfg['paths']['models'], experiment_id)
    cfg['paths']['logs'] = os.path.join(cfg['paths']['logs'], experiment_id)
    cfg['paths']['vis'] = os.path.join(cfg['paths']['vis'], experiment_id)
    cfg['paths']['predictions'] = os.path.join(cfg['paths']['predictions'], experiment_id)

    for path in (cfg['paths']['models'],
                 cfg['paths']['logs'],
                 cfg['paths']['vis'],
                 cfg['paths']['predictions']):
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except FileExistsError:  # Race condition on the workers
                pass

    return cfg
