import os
import re
import cv2
import math
import json
import torch
import random
import hashlib
import skimage.io
import skimage.transform
import numpy as np
import albumentations as A

from tqdm import tqdm
from glob import glob

from lib.nets.hrnet147 import gaussian_blur2d_norm

from torch.utils.data import Dataset

MAX_FRAMES = 200  # If you change this you should regenerate the labels


class E32Dataset(Dataset):
    def __init__(self, cfg, data_dir, train_or_test, transforms, label_generation_cnn=None):
        self.cfg = cfg
        self.train_or_test = train_or_test
        self.label_generation_cnn = label_generation_cnn

        # Data
        self.datadir = data_dir
        self.labelname = cfg['data']['labels']['id']
        self.input_dim = cfg['transforms'][self.train_or_test]['img_size']
        self.frames = cfg['data']['3d'][f'{self.train_or_test}ing_frames']

        # Training
        self.mixed_precision = cfg['training']['mixed_precision']
        self.kldiv = cfg['training'][f'{train_or_test}_criterion'] == 'kldivloss'

        # Label generation
        self.label_generation_cnn = label_generation_cnn
        self.label_generation_dim = cfg['data']['2d']['label_generation_dim']
        self.label_generation_modelpath = cfg['paths']['2d_model']
        self.label_generation_device = cfg['data']['2d']['device']
        self.label_generation_output_layer_ids_by_name = self.get_output_layer_ids_by_name()

        # Transforms
        self.video_load_transform = self.get_video_load_transform()
        self.traintime_transforms = transforms

        if self.label_generation_cnn:
            self.generate_labels()
        else:
            self.studies = self.load_studies()
            self.get_labels()

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        """Load the pngs into a video (a random continguous range of len self.frames for train, starting at 0 for test)
        and then augment this video and the labels with the transforms

        Labelpath is directory, from which it pulls an npz or all the PNGs"""
        studyfolder = list(self.studies.keys())[idx]
        pngpaths = self.studies[studyfolder]['pngs']
        labelpath = self.studies[studyfolder]['label']

        if self.train_or_test == "train":
            frame_from = random.randint(0, len(pngpaths) - self.frames)
        else:
            frame_from = 0

        frame_to = frame_from + self.frames

        pngpaths = pngpaths[frame_from:frame_to]

        x = self.load_video(pngpaths)
        y = self.load_label(labelpath, frame_from, frame_to)
        y = self.upsample_label(y)

        x, y = self.transform_video_for_training(video=x, transforms=self.traintime_transforms, label=y,
                                                 seed=studyfolder)

        x = torch.from_numpy(x.transpose([3, 0, 1, 2]))  # n_frames*h*w*3 -> 3*n_frames*h*w
        y = torch.from_numpy(y.transpose([3, 0, 1, 2]))  # n_frames*h*w*2 -> 2*n_frames*h*w

        if self.mixed_precision:
            x = x.half()
            y = y.half()
        else:
            x = x.float()
            y = y.float()

        if self.kldiv:
            with torch.no_grad():
                bg = 1 - torch.sum(y, dim=0).unsqueeze(0)
                y = torch.cat((bg, y), dim=0)

        return x, y

    def load_label(self, labelpath, frame_from, frame_to):
        """Labelpath is either an npz file or a list of PNGs"""
        format = self.cfg['data']['labels']['format']

        if format == 'npz':
            label = np.load(labelpath)['label'][frame_from:frame_to]

        elif format == 'png':
            raise NotImplementedError()

        else:
            return ValueError()

        return label

    def get_keypoint_names_2d(self):
        with open(self.cfg['paths']['keys_json'], "r") as read_file:
            keypoint_names = list(json.load(read_file).keys())
        return keypoint_names

    def get_output_layer_ids_by_name(self):
        keypoint_names = self.get_keypoint_names_2d()
        output_layernames = self.cfg['data']['labels']['names']
        output_layers_ids = {name:keypoint_names.index(name) for name in output_layernames}
        return output_layers_ids

    def get_video_load_transform(self):
        """Load videos at double the resoluton as HRNet downsamples.
        If we aren't generating labels, however, we then need to downsample x2."""
        video_height, video_width = self.label_generation_dim

        return A.Compose([
            A.PadIfNeeded(min_height=video_height,
                          min_width=video_width,
                          border_mode=cv2.BORDER_CONSTANT,
                          mask_value=0),
            A.CenterCrop(height=video_height, width=video_width)],
        )

    def generate_labels(self):
        studies = self.load_studies(all_studies=True)
        print(f"Generating labels for {len(studies)} studies")
        for studypath, studydict in tqdm(studies.items()):
            pngs = studydict['pngs']
            self.generate_label_from_pngs(pngs, studypath)

    def load_studies(self, all_studies=False):
        """
        all_studies should be set to True if we're generating labels so we get cases across all folds
        OR if folds=0, we will use the entire dataset (for train/test splits)

        Studies will eventually be
        {
            './path/to/study001': {
                'pngs': ['./path/to/study001/0001.png', './path/to/study001/0002.png', './path/to/study001/0003.png'],
                'label': './path/to/study001/label_{self.labelname}.npz'
            }
        }
        Here we generate the dictionary and the png paths, but label comes afterwards"""
        studies = {}
        studypaths = sorted([f for f in glob(os.path.join(self.datadir, "*")) if os.path.isdir(f)])

        n_insufficient_frames = 0

        for studypath in studypaths:
            pngpaths = sorted(glob(os.path.join(studypath, "*.png")))
            if len(pngpaths) >= self.frames:
                studies[studypath] = {'pngs': pngpaths[:MAX_FRAMES]}
            else:
                n_insufficient_frames += 1
        print(f"{self.train_or_test.upper()} {len(studies)} studies. "
              f"({len(studypaths)} total; {n_insufficient_frames} excluded due to insufficient frames)")
        return studies

    def get_labels(self):
        format = self.cfg['data']['labels']['format']

        for studypath, studydict in tqdm(self.studies.copy().items()):

            if format == 'npz':
                labelpath = os.path.join(studypath, f"label_{self.labelname}.npz")
                if not os.path.exists(labelpath):
                    del self.studies[studypath]
                    continue
                self.studies[studypath]['label'] = labelpath

            elif format == 'png':
                raise NotImplementedError()

            else:
                raise ValueError()


    def generate_label_from_pngs(self, pngpaths, studypath):
        """Receives list of PNGs and generates labels using the 2D CNNs which is saved as a NPZ array.
        The video needs to be augmented akin to how Matt would augment it to ensure labels are valid.
        He uses x = x/255 - 0.5
        Matt's network expects 9 channels rather than 3 as he feeds in pre/post"""
        savepath = os.path.join(studypath, f"label_{self.labelname}.npz")
        if os.path.exists(savepath):
            print(f"Skipping as {savepath} exists")
            return

        video = self.load_video(pngpaths)
        video = video / 255 - 0.5
        video = np.concatenate((video, video, video), axis=-1)  # -> RGB
        x = torch.from_numpy(video).to(self.label_generation_device).permute(0, 3, 1, 2)

        # append a pre and post frame if needed
        if self.cfg['data']['2d']['pre_post_frames']:
            x = torch.cat((torch.zeros_like(x), x, torch.zeros_like(x)), dim=1).float()

        x = x.float()

        y_batches = []

        frames_per_batch = self.cfg['data']['2d']['frames_per_batch']
        n_frames = x.shape[0]
        n_batches = math.ceil(n_frames/frames_per_batch)

        for i_batch in range(n_batches):
            frame_from = i_batch*frames_per_batch
            frame_to = (i_batch+1)*frames_per_batch
            x_batch = x[frame_from:frame_to]

            with torch.no_grad():
                try:
                    if self.cfg['data']['2d']['multi_res']:
                        curve_sd = self.cfg['data']['2d']['curve_sd']
                        dot_sd = self.cfg['data']['2d']['dot_sd']
                        scale_factors = (4, 2) if self.cfg['data']['2d']['upsample_labels'] else (2, 1)

                        # get gaussian SD for each keypoint depending on if curve or dot - NB matt keypoint names (all 51)
                        keypoint_names = self.get_keypoint_names_2d()
                        keypoint_sds = [curve_sd if 'curve' in keypoint_name else dot_sd for keypoint_name in keypoint_names]
                        keypoint_sds = torch.tensor(keypoint_sds, dtype=torch.float, device=self.label_generation_device)
                        keypoint_sds = keypoint_sds.unsqueeze(1).expand(-1, 2)

                        y_pred_25_clean, y_pred_50_clean = self.label_generation_cnn(x_batch)

                        y_pred_25 = torch.nn.functional.interpolate(y_pred_25_clean, scale_factor=scale_factors[0], mode='bilinear', align_corners=True)
                        y_pred_50 = torch.nn.functional.interpolate(y_pred_50_clean, scale_factor=scale_factors[1], mode='bilinear', align_corners=True)

                        y_batch = (y_pred_25 + y_pred_50) / 2.0

                        y_batch = gaussian_blur2d_norm(y_pred=y_batch, kernel_size=(25, 25), sigma=keypoint_sds)
                    else:
                        y_batch = self.label_generation_cnn(x)[-1]  # Several outputs from Matt's model, we want last

                    y_batches.append(y_batch.cpu().numpy())

                except RuntimeError as e:
                    print(f"CUDA error for {pngpaths[0]} ({len(pngpaths)} images): {e}")
                    return None

        with torch.no_grad():
            y_batches = np.concatenate((y_batches), axis=0)

        # Make channels last before saving so we can more easily augment each image after loading
        output_layer_ids = list(self.label_generation_output_layer_ids_by_name.values())

        label = y_batches[:, output_layer_ids].transpose((0, 2, 3, 1))  # Want channels last for data aug
        np.savez_compressed(savepath, label=label)


    def load_video(self, pngpaths):
        """Loads a video of len(pngs) frames as a numpy arrays equal to Matt's Shape.
        Important to create video of length len(pngs) rather than self.frames so we can generate a label for every
        PNG available when we are generating labels."""
        video_height, video_width = self.label_generation_dim
        video = np.zeros((len(pngpaths), video_height, video_width, 1))
        for i_png, pngpath in enumerate(sorted(pngpaths)):
            try:
                png = skimage.io.imread(pngpath)
            except ValueError as e:
                raise ValueError(f"Error loading {pngpath}: {e}")
            png = self.rgb_to_grayscale(png)
            png = self.video_load_transform(image=png)['image']
            video[i_png] = png
        return video

    def upsample_label(self, label):
        out = np.zeros((label.shape[0], label.shape[1] * 2, label.shape[2] * 2, label.shape[3]))
        for i, lab in enumerate(label):
            out[i] = skimage.transform.rescale(lab, (2, 2), multichannel=True)
        return out

    def transform_video_for_training(self, video, transforms, seed=None, label=None):
        """Take a video (and a mask too, unless we're generating the mask). Use or create a seed to ensure all frames
        in videos are augmented identically. Then augment each frame."""
        height, width = self.input_dim
        if not seed:
            seed = random.random()  # Still need a seed to make sure each frame identically augmented within a video
        out_video = np.zeros((self.frames, height, width, 1))

        if label is not None:
            out_label = np.zeros((self.frames, height, width, label.shape[-1]))

        for i_frame, frame_video in enumerate(video):
            random.seed(seed)
            frame_label = label[i_frame] if label is not None else None
            aug = transforms(image=frame_video, mask=frame_label)
            out_video[i_frame] = aug['image']
            if label is not None:
                out_label[i_frame] = aug['mask']

        # Start with blank frames occasionally and zerod labels
        blank_chance, blanks_maxframes = self.cfg['transforms'][self.train_or_test].get('blankframes_pre', (False, 0))
        if blank_chance:
            if random.random() < blank_chance:
                blankframes_n = random.randint(1, blanks_maxframes)
                blankframes_from, blankframes_to = 1, blankframes_n
                out_video, out_label = self.zero_frames(out_video, blankframes_from, blankframes_to, out_label, zero_label=True)

        # Finish with blank frames occasionally and zerod labels
        blank_chance, blanks_maxframes = self.cfg['transforms'][self.train_or_test].get('blankframes_post', (False, 0))
        if blank_chance:
            if random.random() < blank_chance:
                blankframes_n = random.randint(1, blanks_maxframes)
                blankframes_from, blankframes_to = len(out_video) - blankframes_n, len(out_video)
                out_video, out_label = self.zero_frames(out_video, blankframes_from, blankframes_to, out_label, zero_label=True)

        if label is not None:
            return out_video, out_label
        else:
            return out_video

    @staticmethod
    def zero_frames(video, frame_from, frame_to, label=None, zero_label=True):
        video[frame_from:frame_to+1] = 0
        if zero_label:
            assert label is not None, "need to supply a label if zero_label is True"
            label[frame_from:frame_to+1] = 0
        return video, label

    @staticmethod
    def rgb_to_grayscale(img):
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                return np.expand_dims(np.dot(img[..., :3], [0.2989, 0.5870, 0.1440]), -1)
            elif img.shape[2] == 1:
                return img
            else:
                raise ValueError()
        elif len(img.shape) == 2:
            return np.expand_dims(img, -1)
        else:
            raise ValueError()