import os
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

from torch.utils.data import Dataset

MAX_FRAMES = 200  # If you change this you should regenerate the labels


class E32Dataset(Dataset):
    def __init__(self, cfg, train_or_test, transforms, label_generation_cnn=None, fold=1, check_labels=False):
        self.cfg = cfg
        self.train_or_test = train_or_test
        self.fold = fold
        self.label_generation_cnn = label_generation_cnn

        # Data
        self.datadir = cfg['paths']['data']
        self.labelname = cfg['data']['labels']['id']
        self.input_dim = cfg['transforms'][self.train_or_test]['img_size']
        self.frames = cfg['data']['3d'][f'{self.train_or_test}ing_frames']
        self.n_folds = cfg['data']['n_folds']
        self.excluded_folds = cfg['data']['excluded_folds']

        # Training
        self.mixed_precision = cfg['training']['mixed_precision']

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
            self.check_labels()

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        """Load the pngs into a video (a random continguous range of len self.frames for train, starting at 0 for test)
        and then augment this video and the labels with the transforms"""
        studyfolder = list(self.studies.keys())[idx]
        pngpaths = self.studies[studyfolder]['pngs']
        labelpath = self.studies[studyfolder]['label']

        if self.train_or_test == "train":
            frame_from = random.randint(0, len(pngpaths) - self.frames)
        else:
            frame_from = 0

        pngpaths = pngpaths[frame_from:frame_from + self.frames]

        x = self.load_video(pngpaths)
        y = np.load(labelpath)['label'][frame_from:frame_from + self.frames]
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

        return x, y

    def get_output_layer_ids_by_name(self):
        with open(self.cfg['paths']['keys_json'], "r") as read_file:
            keypoint_names = list(json.load(read_file).keys())
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
            labelpath = os.path.join(studypath, f"label_{self.labelname}.npz")
            self.generate_label_from_pngs(pngs, labelpath)

    def load_studies(self, all_studies=False):
        """
        all_studies should be set to True if we're generating labels so we get cases across all folds

        Studies will eventually be
        {
            './path/to/study001': {
                'pngs': ['./path/to/study001/0001.png', './path/to/study001/0002.png', './path/to/study001/0003.png'],
                'label': './path/to/study001/label_{self.labelname}.npz'
            }
        }
        Here we generate the dictionary and the png paths, but label comes afterwards"""

        def get_train_test_exclude_for_study(study):
            randnum = int(hashlib.md5(str.encode(study)).hexdigest(), 16) / 16**32
            test_fold = math.floor(randnum * self.n_folds)
            if test_fold == self.fold:
                return 'test'
            elif test_fold in self.excluded_folds:
                return 'excluded'
            else:
                return 'train'

        assert 1 <= self.fold <= self.n_folds, f"Fold should be between 1 and {self.n_folds}, not {self.fold}"
        studies = {}
        studypaths_all = [f for f in glob(os.path.join(self.datadir, "*")) if os.path.isdir(f)]

        if self.label_generation_cnn:  # If including all studies so we can generate labels
            studypaths = studypaths_all
        else:
            studypaths = [f for f in studypaths_all if get_train_test_exclude_for_study(f) == self.train_or_test]

        n_insufficient_frames = 0

        for studypath in studypaths:
            pngpaths = sorted(glob(os.path.join(studypath, "*.png")))
            if len(pngpaths) >= self.frames:
                studies[studypath] = {'pngs': pngpaths[:MAX_FRAMES]}
            else:
                n_insufficient_frames += 1
        print(f"{self.train_or_test.upper()} {len(studies)} studies. "
              f"({len(studypaths_all)} total; {n_insufficient_frames} excluded due to insufficient frames)")
        return studies

    def check_labels(self):
        """Check each study has a label.npz file created from 2D inference. If not, generate it.
        This will throw an error if a label is missing and a 2D CNN isn't found.
        Finally, add the path to label.npz to the studies dictionary under the ['label'] key"""
        for studypath, studydict in tqdm(self.studies.copy().items()):
            labelpath = os.path.join(studypath, f"label_{self.labelname}.npz")
            if not os.path.exists(labelpath):
                del self.studies[studypath]
                continue
                #raise FileNotFoundError(f"Missing label for study {studypath}")
            self.studies[studypath]['label'] = labelpath

    def generate_label_from_pngs(self, pngpaths, savepath):
        """Receives list of PNGs and generates labels using the 2D CNNs which is saved as a NPZ array.
        The video needs to be augmented akin to how Matt would augment it to ensure labels are valid.
        He uses x = x/255 - 0.5
        Matt's network expects 9 channels rather than 3 as he feeds in pre/post"""
        video = self.load_video(pngpaths)
        video = video / 255 - 0.5
        video = np.concatenate((video, video, video), axis=-1)  # -> RGB
        x = torch.from_numpy(video).to(self.label_generation_device).permute(0, 3, 1, 2)
        x = torch.cat((torch.zeros_like(x), x, torch.zeros_like(x)), dim=1).float()
        with torch.no_grad():
            try:
                y = self.label_generation_cnn(x)[-1]  # Several outputs from Matt's model, we want last
            except RuntimeError as e:
                print(f"CUDA memory error for {pngpaths[0]} ({len(pngpaths)} images): {e}")
                return None
        # Make channels last before saving so we can more easily augment each image after loading
        output_layer_ids = list(self.label_generation_output_layer_ids_by_name.values())
        label = y[:, output_layer_ids].permute((0, 2, 3, 1)).cpu().numpy()
        np.savez_compressed(savepath, label=label)

    def load_video(self, pngpaths):
        """Loads a video of len(pngs) frames as a numpy arrays equal to Matt's Shape.
        Important to create video of length len(pngs) rather than self.frames so we can generate a label for every
        PNG available when we are generating labels."""
        video_height, video_width = self.label_generation_dim
        video = np.zeros((len(pngpaths), video_height, video_width, 1))
        for i_png, pngpath in enumerate(sorted(pngpaths)):
            png = skimage.io.imread(pngpath)
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

        if label is not None:
            return out_video, out_label
        else:
            return out_video

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