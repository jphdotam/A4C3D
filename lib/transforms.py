import cv2
import albumentations as A

def load_transforms(cfg):
    input_height, input_width = cfg['data']['2d']['label_generation_dim']

    def get_transforms(transcfg):
        img_height, img_width = transcfg['img_size']

        transforms = []

        if transcfg.get("randomresizedcrop", False):
            scale = transcfg.get("randomresizedcrop")
            transforms.append(A.RandomResizedCrop(height=img_height, width=img_width, scale=scale, ratio=(0.8, 1.2), p=1))
        elif img_height != input_height or img_width != input_width:
            transforms.append(A.Resize(height=img_height, width=img_width))

        if transcfg.get("shiftscalerotate", False):
            transforms.append(A.ShiftScaleRotate(rotate_limit=(-10, 10), border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5))

        if transcfg.get("normalize", True):
            mean, std = cfg['data']['3d']['mean'], cfg['data']['3d']['std']
            transforms.append(A.Normalize(mean=mean, std=std))

        return A.Compose(transforms)

    train_transforms = get_transforms(cfg['transforms']['train'])
    test_transforms = get_transforms(cfg['transforms']['test'])

    return train_transforms, test_transforms
