# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).

import hashlib
import os
import re
import shutil
from copy import deepcopy
from functools import cmp_to_key
from pathlib import Path
import logging
import platform
from typing import Union, List, Any, Dict
from collections import Counter, defaultdict

from PIL import Image, ImageOps
import numpy as np
from numpy import random
import torch
from torchvision.utils import save_image

from .consts import MODEL_VERSION

fmt = '[%(levelname)s %(asctime)s %(funcName)s:%(lineno)d] %(' 'message)s '
logging.basicConfig(format=fmt)
logging.captureWarnings(True)
logger = logging.getLogger()


def set_logger(log_file=None, log_level=logging.INFO, log_file_level=logging.NOTSET):
    """
    Example:
        >>> set_logger(log_file)
        >>> logger.info("abc'")
    """
    log_format = logging.Formatter(fmt)
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        if not Path(log_file).parent.exists():
            os.makedirs(Path(log_file).parent)
        if isinstance(log_file, Path):
            log_file = str(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def select_device(device) -> str:
    if device is not None:
        return device

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    return device


def data_dir_default():
    """

    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'pix2text')
    else:
        return os.path.join(os.path.expanduser("~"), '.pix2text')


def data_dir():
    """

    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('PIX2TEXT_HOME', data_dir_default())


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]


def read_tsv_file(fp, sep='\t', img_folder=None, mode='eval'):
    img_fp_list, labels_list = [], []
    num_fields = 2 if mode != 'test' else 1
    with open(fp) as f:
        for line in f:
            fields = line.strip('\n').split(sep)
            assert len(fields) == num_fields
            img_fp = (
                os.path.join(img_folder, fields[0])
                if img_folder is not None
                else fields[0]
            )
            img_fp_list.append(img_fp)

            if mode != 'test':
                labels = fields[1].split(' ')
                labels_list.append(labels)

    return (img_fp_list, labels_list) if mode != 'test' else (img_fp_list, None)


def read_img(
    path: Union[str, Path], return_type='Tensor'
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """

    Args:
        path (str): image file path
        return_type (str): 返回类型；
            支持 `Tensor`return torch.Tensor；`ndarray`return np.ndarray；`Image`return `Image.Image`

    Returns: RGB Image.Image, or np.ndarray / torch.Tensor, with shape [Channel, Height, Width]
    """
    assert return_type in ('Tensor', 'ndarray', 'Image')
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert('RGB')  # Recognize the rotated image (pillow will not automatically recognize it)
    if return_type == 'Image':
        return img
    img = np.array(img)
    if return_type == 'ndarray':
        return img
    return torch.tensor(img.transpose((2, 0, 1)))


def save_img(img: Union[torch.Tensor, np.ndarray], path):
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    # img *= 255
    # img = img.to(dtype=torch.uint8)
    save_image(img, path)

    # Image.fromarray(img).save(path)


def get_background_color(image: Image.Image, margin=2):
    width, height = image.size

    # Pixel sampling in the edge area
    edge_pixels = []
    for x in range(width):
        for y in range(height):
            if (
                x <= margin
                or y <= margin
                or x >= width - margin
                or y >= height - margin
            ):
                edge_pixels.append(image.getpixel((x, y)))

    # Count the color frequency of edge pixels
    color_counter = Counter(edge_pixels)

    # Get the most frequent colors
    background_color = color_counter.most_common(1)[0][0]

    return background_color


def add_img_margin(
    image: Image.Image, left_right_margin, top_bottom_margin, background_color=None
):
    if background_color is None:
        background_color = get_background_color(image)

    # Get the original image size
    width, height = image.size

    # Calculate the size of the new image
    new_width = width + left_right_margin * 2
    new_height = height + top_bottom_margin * 2

    # Create a new image object and fill it with the specified background color
    new_image = Image.new("RGB", (new_width, new_height), background_color)

    # Paste the original image into the center of the new image
    new_image.paste(image, (left_right_margin, top_bottom_margin))

    return new_image


def prepare_imgs(imgs: List[Union[str, Path, Image.Image]]) -> List[Image.Image]:
    output_imgs = []
    for img in imgs:
        if isinstance(img, (str, Path)):
            img = read_img(img, return_type='Image')
        elif isinstance(img, Image.Image):
            img = img.convert('RGB')
        else:
            raise ValueError(f'Unsupported image type: {type(img)}')
        output_imgs.append(img)

    return output_imgs


COLOR_LIST = [
    [0, 140, 255],  # Dark orange
    [127, 255, 0],  # Spring green
    [255, 144, 30],  # Dodge Blue
    [180, 105, 255],  # Pink
    [128, 0, 128],  # Purple
    [0, 255, 255],  # Yellow
    [255, 191, 0],  # Deep sky blue
    [50, 205, 50],  # Lime green
    [60, 20, 220],  # Scarlet
    [130, 0, 75],  # Indigo
    [255, 0, 0],  # Red
    [0, 255, 0],  # Green
    [0, 0, 255],  # Blue
]


def save_layout_img(img0, categories, one_out, save_path, key='position'):
    import cv2
    from cnstd.yolov7.plots import plot_one_box

    """Visualize the results of layout analysis."""
    if isinstance(img0, Image.Image):
        img0 = cv2.cvtColor(np.asarray(img0.convert('RGB')), cv2.COLOR_RGB2BGR)

    if len(categories) > 13:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in categories]
    else:
        colors = COLOR_LIST
    for one_box in one_out:
        _type = one_box.get('type', 'text')
        box = one_box[key]
        xyxy = [box[0, 0], box[0, 1], box[2, 0], box[2, 1]]
        label = str(_type)
        plot_one_box(
            xyxy,
            img0,
            label=label,
            color=colors[categories.index(_type)],
            line_thickness=1,
        )

    cv2.imwrite(str(save_path), img0)
    logger.info(f" The image with the result is saved in: {save_path}")


def rotated_box_to_horizontal(box):
    """Converts the rotation box to a horizontal rectangle.

    :param box: [4, 2]，Coordinates of the top left, top right, bottom right, and bottom left corner
    """
    xmin = min(box[:, 0])
    xmax = max(box[:, 0])
    ymin = min(box[:, 1])
    ymax = max(box[:, 1])
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


def is_valid_box(box, min_height=8, min_width=2) -> bool:
    """判断box是否有效。
    :param box: [4, 2]，Coordinates of the top left, top right, bottom right, and bottom left corner
    :param min_height: 最小高度
    :param min_width: 最小宽度
    :return: bool, 是否有效
    """
    return (
        box[0, 0] + min_width <= box[1, 0]
        and box[1, 1] + min_height <= box[2, 1]
        and box[2, 0] >= box[3, 0] + min_width
        and box[3, 1] >= box[0, 1] + min_height
    )


def list2box(xmin, ymin, xmax, ymax, dtype=float):
    return np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=dtype
    )


def box2list(bbox):
    return [int(bbox[0, 0]), int(bbox[0, 1]), int(bbox[2, 0]), int(bbox[2, 1])]


def clipbox(box, img_height, img_width):
    new_box = np.zeros_like(box)
    new_box[:, 0] = np.clip(box[:, 0], 0, img_width - 1)
    new_box[:, 1] = np.clip(box[:, 1], 0, img_height - 1)
    return new_box


def y_overlap(box1, box2, key='position'):
    # 计算它们在y轴上的IOU: Interaction / min(height1, height2)
    if key:
        box1 = [box1[key][0][0], box1[key][0][1], box1[key][2][0], box1[key][2][1]]
        box2 = [box2[key][0][0], box2[key][0][1], box2[key][2][0], box2[key][2][1]]
    else:
        box1 = [box1[0][0], box1[0][1], box1[2][0], box1[2][1]]
        box2 = [box2[0][0], box2[0][1], box2[2][0], box2[2][1]]
    # Determine whether there is an intersection
    if box1[3] <= box2[1] or box2[3] <= box1[1]:
        return 0
    # Calculate the height of the intersection
    y_min = max(box1[1], box2[1])
    y_max = min(box1[3], box2[3])
    return (y_max - y_min) / max(1, min(box1[3] - box1[1], box2[3] - box2[1]))


def x_overlap(box1, box2, key='position'):
    # 计算它们在x轴上的IOU: Interaction / min(width1, width2)
    if key:
        box1 = [box1[key][0][0], box1[key][0][1], box1[key][2][0], box1[key][2][1]]
        box2 = [box2[key][0][0], box2[key][0][1], box2[key][2][0], box2[key][2][1]]
    else:
        box1 = [box1[0][0], box1[0][1], box1[2][0], box1[2][1]]
        box2 = [box2[0][0], box2[0][1], box2[2][0], box2[2][1]]
    # Determine whether there is an intersection
    if box1[2] <= box2[0] or box2[2] <= box1[0]:
        return 0
    # Calculate the width of the intersection
    x_min = max(box1[0], box2[0])
    x_max = min(box1[2], box2[2])
    return (x_max - x_min) / max(1, min(box1[2] - box1[0], box2[2] - box2[0]))


def overlap(box1, box2, key='position'):
    return x_overlap(box1, box2, key) * y_overlap(box1, box2, key)


def get_same_line_boxes(anchor, total_boxes):
    line_boxes = [anchor]
    for box in total_boxes:
        if box['line_number'] >= 0:
            continue
        if max([y_overlap(box, l_box) for l_box in line_boxes]) > 0.1:
            line_boxes.append(box)
    return line_boxes


def _compare_box(box1, box2, anchor, key, left_best: bool = True):
    over1 = y_overlap(box1, anchor, key)
    over2 = y_overlap(box2, anchor, key)
    if box1[key][2, 0] < box2[key][0, 0] - 3:
        return -1
    elif box2[key][2, 0] < box1[key][0, 0] - 3:
        return 1
    else:
        if max(over1, over2) >= 3 * min(over1, over2):
            return over2 - over1 if left_best else over1 - over2
        return box1[key][0, 0] - box2[key][0, 0]


def sort_and_filter_line_boxes(line_boxes, key):
    if len(line_boxes) <= 1:
        return line_boxes

    allowed_max_overlay_x = 20

    def find_right_box(anchor):
        anchor_width = anchor[key][2, 0] - anchor[key][0, 0]
        allowed_max = min(
            max(allowed_max_overlay_x, anchor_width * 0.5), anchor_width * 0.95
        )
        right_boxes = [
            l_box
            for l_box in line_boxes[1:]
            if l_box['line_number'] < 0
            and l_box[key][0, 0] >= anchor[key][2, 0] - allowed_max
        ]
        if not right_boxes:
            return None
        right_boxes = sorted(
            right_boxes,
            key=cmp_to_key(
                lambda x, y: _compare_box(x, y, anchor, key, left_best=True)
            ),
        )
        return right_boxes[0]

    def find_left_box(anchor):
        anchor_width = anchor[key][2, 0] - anchor[key][0, 0]
        allowed_max = min(
            max(allowed_max_overlay_x, anchor_width * 0.5), anchor_width * 0.95
        )
        left_boxes = [
            l_box
            for l_box in line_boxes[1:]
            if l_box['line_number'] < 0
            and l_box[key][2, 0] <= anchor[key][0, 0] + allowed_max
        ]
        if not left_boxes:
            return None
        left_boxes = sorted(
            left_boxes,
            key=cmp_to_key(
                lambda x, y: _compare_box(x, y, anchor, key, left_best=False)
            ),
        )
        return left_boxes[-1]

    res_boxes = [line_boxes[0]]
    anchor = res_boxes[0]
    line_number = anchor['line_number']

    while True:
        right_box = find_right_box(anchor)
        if right_box is None:
            break
        right_box['line_number'] = line_number
        res_boxes.append(right_box)
        anchor = right_box

    anchor = res_boxes[0]
    while True:
        left_box = find_left_box(anchor)
        if left_box is None:
            break
        left_box['line_number'] = line_number
        res_boxes.insert(0, left_box)
        anchor = left_box

    return res_boxes


def merge_boxes(bbox1, bbox2):
    """
    Merge two bounding boxes to get a bounding box that encompasses both.

    Parameters:
    - bbox1, bbox2: The bounding boxes to merge. Each box is np.ndarray, with shape of [4, 2]

    Returns: new merged box, with shape of [4, 2]
    """
    # Unwrap the coordinates of the two bounding boxes
    x_min1, y_min1, x_max1, y_max1 = box2list(bbox1)
    x_min2, y_min2, x_max2, y_max2 = box2list(bbox2)

    # Calculate the coordinates of the merged bounding box
    x_min = min(x_min1, x_min2)
    y_min = min(y_min1, y_min2)
    x_max = max(x_max1, x_max2)
    y_max = max(y_max1, y_max2)

    # Return the merged bounding box
    return list2box(x_min, y_min, x_max, y_max)


def sort_boxes(boxes: List[dict], key='position') -> List[List[dict]]:
    # Sort all boxes by y coordinates
    boxes.sort(key=lambda box: box[key][0, 1])
    for box in boxes:
        box['line_number'] = -1  # The line number of the line, -1 indicates that it is not assigned

    def get_anchor():
        anchor = None
        for box in boxes:
            if box['line_number'] == -1:
                anchor = box
                break
        return anchor

    lines = []
    while True:
        anchor = get_anchor()
        if anchor is None:
            break
        anchor['line_number'] = len(lines)
        line_boxes = get_same_line_boxes(anchor, boxes)
        line_boxes = sort_and_filter_line_boxes(line_boxes, key)
        lines.append(line_boxes)

    return lines


def merge_adjacent_bboxes(line_bboxes):
    """
    Merge bounding boxes (bboxes) that are adjacent and close enough to each other in the same row.
    If the distance between the two bounding boxes in the horizontal direction is less than the height of the row, they are merged into a single bounding box.

    :param line_bboxes: A list of bounding box information, each containing the line number, location (coordinates of the four corners), and type.
    :return: A list of merged bounding boxes.
    """
    merged_bboxes = []
    current_bbox = None

    for bbox in line_bboxes:
        # If it is the first bounding box of the current line, or it is not on the same line as the previous bounding box
        if current_bbox is None:
            current_bbox = bbox
            continue

        line_number = bbox['line_number']
        position = bbox['position']
        bbox_type = bbox['type']

        # Calculate the height and width of the bounding box
        height = position[2, 1] - position[0, 1]

        # Check the distance between the current bounding box and the previous bounding box
        distance = position[0, 0] - current_bbox['position'][1, 0]
        if (
            current_bbox['type'] == 'text'
            and bbox_type == 'text'
            and distance <= height
        ):
            # Merge bounding boxes: ymin takes the smaller value of the corresponding value of the two boxes, and ymax takes the larger value of the corresponding value of the two boxes
            # [text]_[text] -> [text_text]
            ymin = min(position[0, 1], current_bbox['position'][0, 1])
            ymax = max(position[2, 1], current_bbox['position'][2, 1])
            xmin = current_bbox['position'][0, 0]
            xmax = position[2, 0]
            current_bbox['position'] = list2box(xmin, ymin, xmax, ymax)
        else:
            if (
                current_bbox['type'] == 'text'
                and bbox_type != 'text'
                and 0 < distance <= height
            ):
                # [text]_[embedding] -> [text_][embedding]
                current_bbox['position'][1, 0] = position[0, 0]
                current_bbox['position'][2, 0] = position[0, 0]
            elif (
                current_bbox['type'] != 'text'
                and bbox_type == 'text'
                and 0 < distance <= height
            ):
                # [embedding]_[text] -> [embedding][_text]
                position[0, 0] = current_bbox['position'][1, 0]
                position[3, 0] = current_bbox['position'][1, 0]
            # Add the current bounding box and start a new merge
            merged_bboxes.append(current_bbox)
            current_bbox = bbox

    if current_bbox is not None:
        merged_bboxes.append(current_bbox)

    return merged_bboxes


def adjust_line_height(bboxes, img_height, max_expand_ratio=0.2):
    """
    Based on the gap between adjacent lines, the height of the box is slightly higher (the detected box can be very close to the text).
    Args:
        bboxes (List[List[dict]]): A list of bounding box information, each containing the line number, location (coordinates of the four corners), and type.
        img_height (int): The height of the original image.
        max_expand_ratio (float): The maximum up-and-down expansion ratio relative to the height of the box

    Returns:

    """

    def get_max_text_ymax(line_bboxes):
        return max([bbox['position'][2, 1] for bbox in line_bboxes])

    def get_min_text_ymin(line_bboxes):
        return min([bbox['position'][0, 1] for bbox in line_bboxes])

    if len(bboxes) < 1:
        return bboxes

    for line_idx, line_bboxes in enumerate(bboxes):
        next_line_ymin = (
            get_min_text_ymin(bboxes[line_idx + 1])
            if line_idx < len(bboxes) - 1
            else img_height
        )
        above_line_ymax = get_max_text_ymax(bboxes[line_idx - 1]) if line_idx > 0 else 0
        for box in line_bboxes:
            if box['type'] != 'text':
                continue
            box_height = box['position'][2, 1] - box['position'][0, 1]
            if box['position'][0, 1] > above_line_ymax:
                expand_size = min(
                    (box['position'][0, 1] - above_line_ymax) // 3,
                    int(max_expand_ratio * box_height),
                )
                box['position'][0, 1] -= expand_size
                box['position'][1, 1] -= expand_size
            if box['position'][2, 1] < next_line_ymin:
                expand_size = min(
                    (next_line_ymin - box['position'][2, 1]) // 3,
                    int(max_expand_ratio * box_height),
                )
                box['position'][2, 1] += expand_size
                box['position'][3, 1] += expand_size
    return bboxes


def adjust_line_width(
    text_box_infos, formula_box_infos, img_width, max_expand_ratio=0.2
):
    """
    If it doesn't overlap with other boxes, expand the text box slightly to the left and right (the detected text box may cut off part of the border character on the border).
        Args:
            text_box_infos (List[dict]): Textbox information where the 'box' field contains the coordinates of the four corners.
            formula_box_infos (List[dict]): Formula box information, where the 'position' field contains the coordinates of the four corners.
            img_width (int): The width of the original image.
            max_expand_ratio (float): The maximum left-right expansion ratio relative to the height of the box.

    Returns: Extended text_box_infos.
    """

    def _expand_left_right(box):
        expanded_box = box.copy()
        xmin, xmax = box[0, 0], box[2, 0]
        box_height = box[2, 1] - box[0, 1]
        expand_size = int(max_expand_ratio * box_height)
        expanded_box[3, 0] = expanded_box[0, 0] = max(xmin - expand_size, 0)
        expanded_box[2, 0] = expanded_box[1, 0] = min(xmax + expand_size, img_width - 1)
        return expanded_box

    def _is_adjacent(anchor_box, text_box):
        if overlap(anchor_box, text_box, key=None) < 1e-6:
            return False
        anchor_xmin, anchor_xmax = anchor_box[0, 0], anchor_box[2, 0]
        text_xmin, text_xmax = text_box[0, 0], text_box[2, 0]
        if (
            text_xmin < anchor_xmin < text_xmax < anchor_xmax
            or anchor_xmin < text_xmin < anchor_xmax < text_xmax
        ):
            return True
        return False

    for idx, text_box in enumerate(text_box_infos):
        expanded_box = _expand_left_right(text_box['position'])
        overlapped = False
        cand_boxes = [
            _text_box['position']
            for _idx, _text_box in enumerate(text_box_infos)
            if _idx != idx
        ]
        cand_boxes.extend(
            [_formula_box['position'] for _formula_box in formula_box_infos]
        )
        for cand_box in cand_boxes:
            if _is_adjacent(expanded_box, cand_box):
                overlapped = True
                break
        if not overlapped:
            text_box_infos[idx]['position'] = expanded_box

    return text_box_infos


def crop_box(text_box, formula_box, min_crop_width=2) -> List[np.ndarray]:
    """
    Crop out the part where text_box and formula_box intersect
    Args:
        text_box ():
        formula_box ():
        min_crop_width (int): The minimum width of the new text box that will be retained after cropping, and text boxes below this width will be deleted.

    Returns:

    """
    text_xmin, text_xmax = text_box[0, 0], text_box[2, 0]
    text_ymin, text_ymax = text_box[0, 1], text_box[2, 1]
    formula_xmin, formula_xmax = formula_box[0, 0], formula_box[2, 0]

    cropped_boxes = []
    if text_xmin < formula_xmin:
        new_text_xmax = min(text_xmax, formula_xmin)
        if new_text_xmax - text_xmin >= min_crop_width:
            cropped_boxes.append((text_xmin, text_ymin, new_text_xmax, text_ymax))

    if text_xmax > formula_xmax:
        new_text_xmin = max(text_xmin, formula_xmax)
        if text_xmax - new_text_xmin >= min_crop_width:
            cropped_boxes.append((new_text_xmin, text_ymin, text_xmax, text_ymax))

    return [list2box(*box, dtype=None) for box in cropped_boxes]


def remove_overlap_text_bbox(text_box_infos, formula_box_infos):
    """
    If a text box intersects with a formula_box, crop the text box。
    Args:
        text_box_infos ():
        formula_box_infos ():

    Returns:

    """

    new_text_box_infos = []
    for idx, text_box in enumerate(text_box_infos):
        max_overlap_val = 0
        max_overlap_fbox = None

        for formula_box in formula_box_infos:
            cur_val = overlap(text_box['position'], formula_box['position'], key=None)
            if cur_val > max_overlap_val:
                max_overlap_val = cur_val
                max_overlap_fbox = formula_box

        if max_overlap_val < 0.1:  # If there is too little overlap, do not do anything
            new_text_box_infos.append(text_box)
            continue
        # if max_overlap_val > 0.8:  # overlap 太多的情况，直接扔掉 text box
        #     continue

        cropped_text_boxes = crop_box(
            text_box['position'], max_overlap_fbox['position']
        )
        if cropped_text_boxes:
            for _box in cropped_text_boxes:
                new_box = deepcopy(text_box)
                new_box['position'] = _box
                new_text_box_infos.append(new_box)

    return new_text_box_infos


def is_chinese(ch):
    """
    Determine whether a character is a Chinese character
    """
    return '\u4e00' <= ch <= '\u9fff'


def find_first_punctuation_position(text):
    # Regular expressions that match common punctuation marks
    pattern = re.compile(r'[,.!?;:()\[\]{}\'\"\\/-]')
    match = pattern.search(text)
    if match:
        return match.start()
    else:
        return len(text)


def smart_join(str_list, spellchecker=None):
    """
    Concatenate the list of strings, and do not add spaces if the two adjacent strings are both 
    Chinese or contain blank symbols; In other cases, spaces are added
    """

    def contain_whitespace(s):
        if re.search(r'\s', s):
            return True
        else:
            return False

    str_list = [s for s in str_list if s]
    if not str_list:
        return ''
    res = str_list[0]
    for i in range(1, len(str_list)):
        if (is_chinese(res[-1]) and is_chinese(str_list[i][0])) or contain_whitespace(
            res[-1] + str_list[i][0]
        ):
            res += str_list[i]
        elif spellchecker is not None and res.endswith('-'):
            fields = res.rsplit(' ', maxsplit=1)
            if len(fields) > 1:
                new_res, prev_word = fields[0], fields[1]
            else:
                new_res, prev_word = '', res

            fields = str_list[i].split(' ', maxsplit=1)
            if len(fields) > 1:
                next_word, new_next = fields[0], fields[1]
            else:
                next_word, new_next = str_list[i], ''

            punct_idx = find_first_punctuation_position(next_word)
            next_word = next_word[:punct_idx]
            new_next = str_list[i][len(next_word) :]
            new_word = prev_word[:-1] + next_word
            if (
                next_word
                and spellchecker.unknown([prev_word + next_word])
                and spellchecker.known([new_word])
            ):
                res = new_res + ' ' + new_word + new_next
            else:
                new_word = prev_word + next_word
                res = new_res + ' ' + new_word + new_next
        else:
            res += ' ' + str_list[i]
    return res


def cal_block_xmin_xmax(lines, indentation_thrsh):
    total_min_x, total_max_x = min(lines[:, 0]), max(lines[:, 1])
    if lines.shape[0] < 2:
        return total_min_x, total_max_x

    min_x, max_x = min(lines[1:, 0]), max(lines[1:, 1])
    first_line_is_full = total_max_x > max_x - indentation_thrsh
    if first_line_is_full:
        return min_x, total_max_x

    return total_min_x, total_max_x


def merge_line_texts(
    outs: List[Dict[str, Any]],
    auto_line_break: bool = True,
    line_sep='\n',
    embed_sep=(' $', '$ '),
    isolated_sep=('$$\n', '\n$$'),
    spellchecker=None,
) -> str:
    """
    handful Pix2Text.recognize_by_mfd() and combine them into a single string
    Args:
        outs (List[Dict[str, Any]]):
        auto_line_break: Automatically determines whether it is time to wrap based on the position of the box
        line_sep: Separator between rows
        embed_sep (tuple): Prefix and suffix for embedding latex; default value is `(' $', '$ ')`
        isolated_sep (tuple): Prefix and suffix for isolated latex; default value is `('$$\n', '\n$$')`
        spellchecker: Spell Checker

    Returns: The merged string

    """
    if not outs:
        return ''
    out_texts = []
    line_margin_list = []  # The leftmost and rightmost x coordinates of each row
    isolated_included = []  # Whether each row contains a mathematical formula of type 'isolated'
    line_height_dict = defaultdict(list)  # The height of each block in each row
    line_ymin_ymax_list = []  # The y-coordinates of the topmost and bottommost edges of each row
    for _out in outs:
        line_number = _out.get('line_number', 0)
        while len(out_texts) <= line_number:
            out_texts.append([])
            line_margin_list.append([100000, 0])
            isolated_included.append(False)
            line_ymin_ymax_list.append([100000, 0])
        cur_text = _out['text']
        cur_type = _out.get('type', 'text')
        box = _out['position']
        if cur_type in ('embedding', 'isolated'):
            sep = isolated_sep if _out['type'] == 'isolated' else embed_sep
            cur_text = sep[0] + cur_text + sep[1]
        if cur_type == 'isolated':
            isolated_included[line_number] = True
            cur_text = line_sep + cur_text + line_sep
        out_texts[line_number].append(cur_text)
        line_margin_list[line_number][1] = max(
            line_margin_list[line_number][1], float(box[2, 0])
        )
        line_margin_list[line_number][0] = min(
            line_margin_list[line_number][0], float(box[0, 0])
        )
        if cur_type == 'text':
            line_height_dict[line_number].append(box[2, 1] - box[1, 1])
            line_ymin_ymax_list[line_number][0] = min(
                line_ymin_ymax_list[line_number][0], float(box[0, 1])
            )
            line_ymin_ymax_list[line_number][1] = max(
                line_ymin_ymax_list[line_number][1], float(box[2, 1])
            )

    line_text_list = [smart_join(o) for o in out_texts]

    for _line_number in line_height_dict.keys():
        if line_height_dict[_line_number]:
            line_height_dict[_line_number] = np.mean(line_height_dict[_line_number])
    _line_heights = list(line_height_dict.values())
    mean_height = np.mean(_line_heights) if _line_heights else None

    default_res = re.sub(rf'{line_sep}+', line_sep, line_sep.join(line_text_list))
    if not auto_line_break:
        return default_res

    line_lengths = [rx - lx for lx, rx in line_margin_list]
    line_length_thrsh = max(line_lengths) * 0.3
    if line_length_thrsh < 1:
        return default_res

    lines = np.array(
        [
            margin
            for idx, margin in enumerate(line_margin_list)
            if isolated_included[idx] or line_lengths[idx] >= line_length_thrsh
        ]
    )
    if lines.shape[0] < 1:
        return default_res
    min_x, max_x = min(lines[:, 0]), max(lines[:, 1])

    indentation_thrsh = (max_x - min_x) * 0.1
    if mean_height is not None:
        indentation_thrsh = 1.5 * mean_height

    min_x, max_x = cal_block_xmin_xmax(lines, indentation_thrsh)

    res_line_texts = [''] * len(line_text_list)
    line_text_list = [(idx, txt) for idx, txt in enumerate(line_text_list) if txt]
    for idx, (line_number, txt) in enumerate(line_text_list):
        if isolated_included[line_number]:
            res_line_texts[line_number] = line_sep + txt + line_sep
            continue

        tmp = txt
        if line_margin_list[line_number][0] > min_x + indentation_thrsh:
            tmp = line_sep + txt
        if line_margin_list[line_number][1] < max_x - indentation_thrsh:
            tmp = tmp + line_sep
        if idx < len(line_text_list) - 1:
            cur_height = line_ymin_ymax_list[line_number][1] - line_ymin_ymax_list[line_number][0]
            next_line_number = line_text_list[idx + 1][0]
            if (
                cur_height > 0
                and line_ymin_ymax_list[next_line_number][0] < line_ymin_ymax_list[next_line_number][1]
                and line_ymin_ymax_list[next_line_number][0] - line_ymin_ymax_list[line_number][1]
                > cur_height
            ):  # If the distance between the current line and the next line exceeds the line height of the line, it is considered that they should be different paragraphs
                tmp = tmp + line_sep
        res_line_texts[idx] = tmp

    outs = smart_join([c for c in res_line_texts if c], spellchecker)
    return re.sub(rf'{line_sep}+', line_sep, outs)  # Replace multiple '\n' with '\n'


def prepare_model_files(root, model_info) -> Path:
    model_root_dir = Path(root) / MODEL_VERSION
    model_dir = model_root_dir / model_info['local_model_id']
    if model_dir.is_dir() and list(model_dir.glob('**/[!.]*')):
        return model_dir
    assert 'hf_model_id' in model_info
    model_dir.mkdir(parents=True)
    download_cmd = f'huggingface-cli download --repo-type model --resume-download --local-dir-use-symlinks False {model_info["hf_model_id"]} --local-dir {model_dir}'
    os.system(download_cmd)
    # If there is no file in the current directory, download it from Huggingface
    if not list(model_dir.glob('**/[!.]*')):
        if model_dir.exists():
            shutil.rmtree(str(model_dir))
        os.system('HF_ENDPOINT=https://hf-mirror.com ' + download_cmd)
    return model_dir
