import os
import numpy as np
from glob import glob


def get_data_and_points_paths(input_dir):

    def has_points(path):
        point_paths = sorted(glob(os.path.join(os.path.dirname(path), "*pts.npy")))
        if len(point_paths) == 1:
            return point_paths[0]
        else:
            return False

    data_paths = glob(os.path.join(input_dir, "**/data.npy"), recursive=True)
    data_with_points = {}
    for data_path in data_paths:
        points_path = has_points(data_path)
        if points_path:
            data_with_points[data_path] = points_path

    return data_with_points


def export_cine_and_points(cinepath, pointpath, cfg):
    export_method = cfg['data']['export']['method']
    export_height, export_width = cfg['data']['export']['output_res']
    export_dir = cfg['paths']['data']
    export_format = cfg['data']['export']['format']
    gaussian_sigma = cfg['data']['gaussian_sigma']
    study_type = os.path.basename(os.path.dirname(cinepath))
    patient_id = os.path.basename(os.path.dirname(os.path.dirname(cinepath)))
    date = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(cinepath))))

    cine = np.load(cinepath)

    points = np.load(pointpath)

    data_export_path = os.path.join(export_dir, f"{date}_{patient_id}_{study_type}_data.npz")
    label_export_path = os.path.join(export_dir, f"{date}_{patient_id}_{study_type}_label.npz")

    if export_method == 'centre_crop':
        cine, top_left = center_crop(cine, export_height, export_width)
        if export_format == 'probs':
            probs = create_probability_map(points, export_height, export_width, top_left, gaussian_sigma)
            save_probs(probs, label_export_path)
        elif export_format == 'points':
            points = create_points(points, top_left)
            save_points(points, label_export_path)

        save_cine(cine, data_export_path)

    else:
        raise ValueError(f"Unknown export method {export_method}")


def save_cine(cine, export_path):
    if len(cine.shape) == 3:
        # Add a stack dimension (to make 2/3/4CH like SAX)
        cine = np.expand_dims(cine, axis=-1)
    elif len(cine.shape) != 4:
        raise ValueError(f"cine must have 3 or 4 dimensions, not {cine.shape}")

    cine = cine.transpose((3, 2, 0, 1))  # -> STACK_DIM, TIME, HEIGHT, WIDTH
    np.savez_compressed(export_path, data=cine)


def save_probs(probs, export_path):
    assert len(probs.shape) == 5, f"Probability map should have 5 dimensions, not {probs.shape}"
    np.savez_compressed(export_path, probs=probs)


def save_points(points, export_path):
    assert len(points.shape) == 4, f"Points should have 4 dimensions, not {points.shape}"
    np.savez_compressed(export_path, points=points)


def create_probability_map(points, height, width, top_left, gaussian_sigma):
    """This creates a map of gaussian blobs for each points.
    Remember, the points are in native co-ords, but we want an output which
    is height*width, and where 0,0 is the top left co-ord of the cine"""
    if len(points.shape) == 3:
        # Add a stack dimension (to make 2/3/4CH like SAX)
        points = np.expand_dims(points, axis=-1)
    elif len(points.shape) != 4:
        raise ValueError(f"points data must have 3 or 4 dimensions, not {points.shape}")
    points = points.transpose((3, 2, 0, 1))  # -> STACK_DIM, TIME, N_POINTS, 2
    n_stacks, n_frames, n_points, _ = points.shape

    mask = np.zeros((n_stacks, n_frames, height, width, n_points))
    gauss_kernel = get_gauss_kernel(height, width, gaussian_sigma)

    for i_stack, stack in enumerate(points):
        for i_frame, frame in enumerate(stack):
            for i_point, point in enumerate(frame):
                point = point - (top_left[1], top_left[0])  # point is x, y; top_left is row, col in numpy format
                mask = put_point_on_mask(mask, point, gauss_kernel, i_stack, i_frame, i_point)

    return mask


def create_points(points, top_left):
    """Creates a numpy array of co-ordinates.
    We replace the out of bounds (-1, -1) with (-1000, -1000) to ensure out of bounds"""
    if len(points.shape) == 3:
        # Add a stack dimension (to make 2/3/4CH like SAX)
        points = np.expand_dims(points, axis=-1)
    elif len(points.shape) != 4:
        raise ValueError(f"points data must have 3 or 4 dimensions, not {points.shape}")
    points = points.transpose((3, 2, 0, 1))  # -> STACK_DIM, TIME, N_POINTS, 2
    points = np.where(points == -1, -1000, points)
    points = points - (top_left[1], top_left[0])
    return points


def put_point_on_mask(mask, point, gauss_kernel, i_stack, i_frame, i_point):
    x, y = int(round(point[0])), int(round(point[1]))
    n_stacks, n_frames, mask_height, mask_width, n_labels = mask.shape
    gauss_max_displacement = max(mask_height, mask_width)

    if x < 0 or y < 0 or y > mask_height or x > mask_width:  # Point off screen
        return mask

    row_from, row_to = y - gauss_max_displacement, y + gauss_max_displacement + 1
    col_from, col_to = x - gauss_max_displacement, x + gauss_max_displacement + 1

    gauss_row_from, gauss_row_to = 0, gauss_max_displacement * 2 + 1
    gauss_col_from, gauss_col_to = 0, gauss_max_displacement * 2 + 1

    if row_from < 0:
        gauss_row_from += abs(row_from)
        row_from = 0
    if row_to < 0:
        gauss_row_to += abs(row_to)
        row_to = 0
    if col_from < 0:
        gauss_col_from += abs(col_from)
        col_from = 0
    if col_to < 0:
        gauss_col_to += abs(col_to)
        col_to = 0
    if row_from > mask_height:
        gauss_row_from -= (row_from - mask_height)
        row_from = mask_height
    if row_to > mask_height:
        gauss_row_to -= (row_to - mask_height)
        row_to = mask_height
    if col_from > mask_width:
        gauss_col_from -= (col_from - mask_width)
        col_from = mask_width
    if col_to > mask_width:
        gauss_col_to -= (col_to - mask_width)
        col_to = mask_width

    mask[i_stack, i_frame, row_from:row_to, col_from:col_to, i_point] += gauss_kernel[
                                                                gauss_row_from:gauss_row_to,
                                                                gauss_col_from:gauss_col_to]

    return mask


def get_gauss_kernel(height, width, sigma):
    def calc_gauss_on_a_scalar_or_matrix(dist, sig):
        return 0.8 * np.exp(-(dist ** 2) / (2 * sig ** 2)) + 0.2 * (1 / (1 + dist))

    max_image_dim = max(height, width)
    center, kernel_dimension = max_image_dim, max_image_dim * 2
    kernel = np.zeros((kernel_dimension, kernel_dimension)).astype("float32")
    for i in range(kernel_dimension):
        for j in range(kernel_dimension):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel[i, j] = calc_gauss_on_a_scalar_or_matrix(distance, sigma)
    return kernel / np.max(kernel)


def center_crop(img_or_cine, crop_height, crop_width, centre=None, pad_first=True):
    """Either crops by the center of the image, or around a supplied point.

    If pad_first==False, does not pad; if the supplied centre is towards the egde of the image, the padded
    area is shifted so crops start at 0 and only go up to the max row/col

    We also need to track the co-ord offset due to the padding (0,0 if not padded)

    Returns both the new crop, and the top-left coords as a row,col tuple"""
    input_height, input_width = img_or_cine.shape[:2]
    pad_top_left = 0, 0

    if pad_first:
        img_or_cine, pad_top_left = pad_if_needed(img_or_cine, crop_height, crop_width)
        input_height, input_width = img_or_cine.shape[:2]

    if centre is None:
        row_from = (input_height - crop_height)//2
        col_from = (input_width - crop_width)//2
    else:
        row_centre, col_centre = centre
        row_from = max(row_centre - (crop_height//2), 0)
        if (row_from + crop_height) > input_height:
            row_from -= (row_from + crop_height - input_height)
        col_from = max(col_centre - (crop_width//2), 0)
        if (col_from + crop_width) > input_width:
            col_from -= (col_from + crop_width - input_width)

    img_or_cine = img_or_cine[row_from:row_from + crop_height, col_from:col_from + crop_width]

    top_left = (row_from - pad_top_left[0], col_from - pad_top_left[1])

    return img_or_cine, top_left


def pad_if_needed(img_or_cine, min_height, min_width):
    input_height, input_width = img_or_cine.shape[:2]
    new_shape = list(img_or_cine.shape)
    new_shape[0] = max(input_height, min_height)
    new_shape[1] = max(input_width, min_width)
    row_from, col_from = 0, 0
    if input_height < min_height:
        row_from = (min_height - input_height) // 2
    if input_width < min_width:
        col_from = (min_width - input_width) // 2
    out = np.zeros(new_shape, dtype=img_or_cine.dtype)
    out[row_from:row_from+input_height, col_from:col_from+input_width] = img_or_cine
    return out, (row_from, col_from)
