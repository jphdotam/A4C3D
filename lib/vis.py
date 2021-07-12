import os
import wandb
import numpy as np
import skvideo.io

import torch
from torch.utils.data import DataLoader


def put_masks_on_img(img, masks, colours, layers_to_visualise=None, raise_preds_to_power=1):
    img_rgb = np.concatenate((img, img, img), -1)
    overlays = np.zeros_like(img_rgb)

    for i_mask, mask in enumerate(masks):
        if layers_to_visualise and i_mask not in layers_to_visualise:
            continue

        # mask = mask / max(np.max(mask), 0.1)  # Stops division by a very very small prediction, causing everything to be white
        if raise_preds_to_power > 1:
            mask = np.power(mask, raise_preds_to_power)
        colour_rgb = colours[i_mask]
        overlays = np.maximum(overlays,
                              np.dstack((
                                  colour_rgb[0] / 255 * mask,  # red
                                  colour_rgb[1] / 255 * mask,  # green
                                  colour_rgb[2] / 255 * mask)))  # blue
    return np.maximum(np.minimum((img / 2 + overlays), 1), 0)


def create_video(x, y_pred, y_true=None, filename_pred=None, filename_true=None, colours=None, layers_to_visualise=None,
                 raise_preds_to_power_mp4=1):
    if colours is None:
        colours = [[255, 0, 0],
                   [0, 255, 0],
                   [0, 0, 255],
                   [128, 128, 0],
                   [0, 128, 128],
                   [128, 0, 128],
                   [255, 128, 0],
                   [0, 255, 128],
                   [128, 0, 255],
                   [128, 255, 0],
                   [0, 128, 255],
                   [255, 0, 128],
                   [255, 255, 0],
                   [0, 255, 255],
                   [255, 0, 255]]

    # remove batch dims if present
    if len(x.shape) == 5:
        x = x[0]
    if y_pred is not None and len(y_pred.shape) == 5:
        y_pred = y_pred[0]
    if y_true is not None and len(y_true.shape) == 5:
        y_true = y_true[0]

    x = x.float()
    x = x.permute(1, 2, 3, 0).cpu().numpy()  # 1 * 32 * 224 * 224 -> 32 * 224 * 224 * 1
    if y_true is not None:
        y_true = y_true.permute(1, 0, 2, 3).cpu().numpy()  # 2 * 32 * 224 * 224 -> 32 * 2 * 224 * 224
        out_true = np.zeros((x.shape[0], x.shape[1], x.shape[2], 3), dtype=np.uint8)

    if y_pred is not None:
        y_pred = y_pred.permute(1, 0, 2, 3).cpu().numpy()  # 2 * 32 * 224 * 224 -> 32 * 2 * 224 * 224
        out_pred = np.zeros((x.shape[0], x.shape[1], x.shape[2], 3), dtype=np.uint8)

    for i_frame, frame in enumerate(x):
        if y_true is not None:
            true_frame = y_true[i_frame]
            out_true[i_frame] = put_masks_on_img(frame, true_frame, colours, layers_to_visualise,
                                                 raise_preds_to_power=1) * 255
        if y_pred is not None:
            pred_frame = y_pred[i_frame]
            out_pred[i_frame] = put_masks_on_img(frame, pred_frame, colours, layers_to_visualise,
                                                 raise_preds_to_power=raise_preds_to_power_mp4) * 255

    outputdict = {}  # {"-vcodec": "libx264", "-pix_fmt": "yuv420p"}

    if filename_true and y_true is not None:
        skvideo.io.vwrite(filename_true, out_true, outputdict=outputdict)
    if filename_pred and y_pred is not None:
        skvideo.io.vwrite(filename_pred, out_pred, outputdict=outputdict)

    if y_pred is not None:
        out_pred = out_pred.transpose((0, 3, 1, 2))
    if y_true is not None:
        out_true = out_true.transpose((0, 3, 1, 2))

    if y_pred is not None and y_true is not None:
        return out_true, out_pred
    elif y_pred is not None:
        return out_pred
    elif y_true is not None:
        return out_true


def vis_mse(dataset_or_dataloader, model, epoch, cfg):
    if epoch % cfg['output']['vis_every_epoch']:
        return

    # Settings
    device = cfg['training']['device']
    write_probs = cfg['output']['write_probs']
    probs_dir = cfg['paths']['predictions']
    bs, n_batches = cfg['training']['batch_size_test'], cfg['output']['batches_vis']
    normalize = cfg['transforms']['test']['normalize']
    mean, std = cfg['data']['3d']['mean'][0], cfg['data']['3d']['std'][0]
    sigmoid = cfg['training']['sigmoid']
    kldiv = cfg['training'][f'test_criterion'] == 'kldivloss'

    wandb_dict = {'epoch': epoch}
    wandb_videos_true, wandb_videos_pred = [], []

    # Get data & predict
    if type(dataset_or_dataloader) == DataLoader:
        dataloader = dataset_or_dataloader
    else:
        dataloader = DataLoader(dataset_or_dataloader, bs, shuffle=False, num_workers=1, pin_memory=True)
    iter_dl = iter(dataloader)
    for i_batch in range(n_batches):
        batch_x, batch_y_true = next(iter_dl)
        with torch.no_grad():
            batch_y_pred = model(batch_x.to(device))
            if kldiv:
                batch_y_pred = torch.softmax(batch_y_pred, dim=1)[:, 1:]  # Get probabilities and ignore BG channel
                batch_y_true = batch_y_true[:, 1:]
            if sigmoid:
                batch_y_pred = torch.sigmoid(batch_y_pred)

        if write_probs:
            np.savez_compressed(os.path.join(probs_dir, f"{epoch}.npz"),
                                true=batch_y_true.cpu().numpy(),
                                pred=batch_y_pred.cpu().numpy())

        for i, (video, true_video, pred_video) in enumerate(zip(batch_x, batch_y_true, batch_y_pred)):
            if normalize:
                video = video * std + mean
            vid_true, vid_pred = create_video(video, pred_video, true_video)
            wandb_videos_true.append(wandb.Video(vid_true, fps=20, format="mp4"))
            wandb_videos_pred.append(wandb.Video(vid_pred, fps=20, format="mp4"))

    wandb_dict["videos_true"] = wandb_videos_true
    wandb_dict["videos_pred"] = wandb_videos_pred

    wandb.log(wandb_dict)
    print(f"logged: {wandb_dict}")
