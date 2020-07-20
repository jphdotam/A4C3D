import os
import math
import numpy as np
import skvideo.io
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.cuda import empty_cache


def put_masks_on_img(img, masks, colours, layers_to_visualise=None, raise_preds_to_power=1):
    img_rgb = np.concatenate((img, img, img), -1)
    overlays = np.zeros_like(img_rgb)

    for i_mask, mask in enumerate(masks):
        if layers_to_visualise and i_mask not in layers_to_visualise:
            continue

        #mask = mask / max(np.max(mask), 0.1)  # Stops division by a very very small prediction, causing everything to be white
        if raise_preds_to_power > 1:
            mask = np.power(mask, raise_preds_to_power)
        colour_rgb = colours[i_mask]
        overlays = np.maximum(overlays,
                              np.dstack((
                                  colour_rgb[0] / 255 * mask,  # red
                                  colour_rgb[1] / 255 * mask,  # green
                                  colour_rgb[2] / 255 * mask)))  # blue
    return np.maximum(np.minimum((img / 2 + overlays), 1), 0)


def create_video(x, y_pred, y_true=None, filename_pred=None, filename_true=None, colours=None, layers_to_visualise=None, raise_preds_to_power_mp4=1):
    if colours is None:
        colours = [[255,0,0],
                   [0,255,0],
                   [0,0,255],
                   [128,128,0],
                   [0,128,128],
                   [128,0,128],
                   [255,128,0],
                   [255,0,128],
                   [128,255,0],
                   [0,255,128],
                   [0,128,255],
                   [128,0,255]]

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

    if y_pred is not None and y_true is not None:
        return out_true, out_pred
    elif y_pred is not None:
        return out_pred
    elif y_true is not None:
        return out_true


def vis_mse(dataset_or_dataloader, model, epoch, cfg, writer=None):
    if epoch % cfg['output']['vis_every_epoch']:
        return

    # Settings
    output_dir = cfg['paths']['vis']
    n_to_display = cfg['output']['n_visualise']
    n_cols = cfg['output']['n_visualise_cols']
    i_frame_to_display = cfg['output']['i_frame_to_display']
    colours = cfg['output']['colours']
    layers_to_visualise = cfg['output']['layers_to_visualise']
    device = cfg['training']['device']
    write_png_folder = cfg['output']['write_png_folder']
    write_png_tb = cfg['output']['write_png_tb']
    show_png = cfg['output']['show_png']
    write_mp4_folder = cfg['output']['write_mp4_folder']
    write_probs = cfg['output']['write_probs']
    probs_dir = cfg['paths']['predictions']

    # Get data & predict
    if type(dataset_or_dataloader) == DataLoader:
        dataloader = dataset_or_dataloader
    else:
        dataloader = DataLoader(dataset_or_dataloader, n_to_display, shuffle=False, num_workers=1, pin_memory=True)
    batch_x, batch_y_true = next(iter(dataloader))
    with torch.no_grad():
        batch_y_pred = model(batch_x.to(device))

        if write_probs:
            np.savez_compressed(os.path.join(probs_dir, f"{epoch}.npz"),
                                true=batch_y_true.cpu().numpy(),
                                pred=batch_y_pred.cpu().numpy())

    # PNGs
    n_rows = math.ceil(n_to_display/n_cols)
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(20,20))
    for i, (video, true_video, pred_video) in enumerate(zip(batch_x, batch_y_true, batch_y_pred)):

        row = i // n_cols
        col = i % n_cols

        frame = torch.squeeze(video[:, i_frame_to_display], 1).permute(1, 2, 0).cpu().numpy()
        frame = frame * 0.25 + 0.5  # *STD+MEAN
        true_frame = torch.squeeze(true_video[:, i_frame_to_display], 1).cpu().numpy()
        pred_frame = torch.squeeze(pred_video[:, i_frame_to_display], 1).cpu().numpy()

        overlay_true = put_masks_on_img(frame, true_frame, colours, layers_to_visualise)
        overlay_pred = put_masks_on_img(frame, pred_frame, colours, layers_to_visualise)

        axes[row * 2][col].imshow(overlay_true.astype(np.float32))  # In case was float16
        axes[row * 2 + 1][col].imshow(overlay_pred.astype(np.float32))

        if i >= n_to_display - 1:
            break

    plt.suptitle(epoch)
    if write_png_folder:
        plt.savefig(os.path.join(output_dir, f"vis_{epoch}_.png"))
    if show_png:
        plt.show()
    if write_png_tb:
        writer.add_figure(f"Epoch {epoch:03d}", fig, epoch)

    # MP4
    if not write_mp4_folder:
        return None

    for i, (video, true_video, pred_video) in enumerate(zip(batch_x, batch_y_true, batch_y_pred)):
        filename_true = os.path.join(output_dir, f"{epoch}_{i}_true.mp4")
        filename_pred = os.path.join(output_dir, f"{epoch}_{i}_pred.mp4")

        video = video * 0.25 + 0.5

        create_video(video, true_video, pred_video, filename_true, filename_pred)
        if i >= n_to_display - 1:
            break

    del batch_x
    del batch_y_true
    del batch_y_pred
    empty_cache()
