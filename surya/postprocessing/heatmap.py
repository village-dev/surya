from time import time
from typing import List, Tuple, Union

import numpy as np
import cv2
from PIL import ImageDraw, ImageFont

from surya.postprocessing.fonts import get_font_path
from surya.schema import PolygonBox
from surya.settings import settings
from surya.postprocessing.text import get_text_size
from loguru import logger
from shapely.geometry import Polygon
from doctr.models.detection.fast.base import FASTPostProcessor


def keep_largest_boxes(boxes: List[PolygonBox]) -> List[PolygonBox]:
    new_boxes = []
    for box_obj in boxes:
        box = box_obj.bbox
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        contained = False
        for other_box_obj in boxes:
            if other_box_obj.polygon == box_obj.polygon:
                continue

            other_box = other_box_obj.bbox
            other_box_area = (other_box[2] - other_box[0]) * (
                other_box[3] - other_box[1]
            )
            if box == other_box:
                continue
            # find overlap percentage
            overlap = box_obj.intersection_pct(other_box_obj)
            if overlap > 0.9 and box_area < other_box_area:
                contained = True
                break
        if not contained:
            new_boxes.append(box_obj)
    return new_boxes


def clean_boxes(boxes: List[PolygonBox]) -> List[PolygonBox]:
    new_boxes = []
    for box_obj in boxes:
        xs = [point[0] for point in box_obj.polygon]
        ys = [point[1] for point in box_obj.polygon]
        if max(xs) == min(xs) or max(ys) == min(ys):
            continue

        box = box_obj.bbox
        contained = False
        for other_box_obj in boxes:
            if other_box_obj.polygon == box_obj.polygon:
                continue

            other_box = other_box_obj.bbox
            if box == other_box:
                continue
            if (
                box[0] >= other_box[0]
                and box[1] >= other_box[1]
                and box[2] <= other_box[2]
                and box[3] <= other_box[3]
            ):
                contained = True
                break
        if not contained:
            new_boxes.append(box_obj)
    return new_boxes


def get_dynamic_thresholds(
    linemap: np.ndarray, text_threshold, low_text, typical_top10_avg=0.7
):
    # Find average intensity of top 10% pixels
    flat_map = linemap.ravel()
    top_10_count = int(len(flat_map) * 0.9)
    avg_intensity = np.mean(np.partition(flat_map, top_10_count)[top_10_count:])
    scaling_factor = np.clip(avg_intensity / typical_top10_avg, 0, 1) ** (1 / 2)

    low_text = np.clip(low_text * scaling_factor, 0.1, 0.6)
    text_threshold = np.clip(text_threshold * scaling_factor, 0.15, 0.8)

    return text_threshold, low_text


processor = FASTPostProcessor(
    bin_thresh=0.5, box_thresh=0.5, assume_straight_pages=True
)


def detect_boxes(
    linemap: np.ndarray, text_threshold: float, low_text: float
) -> Tuple[List[np.ndarray], List[float]]:
    # init FAST processor

    # binarize the map
    bitmap = (linemap > 0.5).astype(np.uint8)

    # get boxes and scores
    boxes = processor.bitmap_to_boxes(linemap, bitmap)

    # convert relative coords to absolute
    h, w = linemap.shape
    det = []
    confidences = []

    for box in boxes:
        # extract score
        confidence = float(box[4])

        # convert relative to absolute coords
        x1, y1, x2, y2 = box[:4]
        abs_box = np.array(
            [[x1 * w, y1 * h], [x2 * w, y1 * h], [x2 * w, y2 * h], [x1 * w, y2 * h]]
        )

        det.append(abs_box)
        confidences.append(confidence)

    if len(confidences) > 0:
        max_confidence = max(confidences)
        confidences = [c / max_confidence for c in confidences]

    return det, confidences


def get_detected_boxes(
    textmap: np.ndarray, text_threshold=None, low_text=None
) -> List[PolygonBox]:
    if text_threshold is None:
        text_threshold = settings.DETECTOR_TEXT_THRESHOLD

    if low_text is None:
        low_text = settings.DETECTOR_BLANK_THRESHOLD

    textmap = textmap.copy()
    # textmap = textmap.astype(np.float32)
    boxes, confidences = detect_boxes(textmap, text_threshold, low_text)

    # From point form to box form
    boxes = [
        PolygonBox(polygon=box, confidence=confidence)
        for box, confidence in zip(boxes, confidences)
    ]
    return boxes


def get_and_clean_boxes(
    textmap: np.ndarray, processor_size, image_size, text_threshold=None, low_text=None
) -> List[PolygonBox]:
    bboxes = get_detected_boxes(textmap, text_threshold, low_text)
    for bbox in bboxes:
        bbox.rescale(processor_size, image_size)
        bbox.fit_to_bounds([0, 0, image_size[0], image_size[1]])

    bboxes = clean_boxes(bboxes)
    return bboxes


def draw_bboxes_on_image(
    bboxes, image, labels=None, label_font_size=10, color: str | list = "red"
):
    polys = []
    for bb in bboxes:
        # Clockwise polygon
        poly = [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]
        polys.append(poly)

    return draw_polys_on_image(
        polys, image, labels, label_font_size=label_font_size, color=color
    )


def draw_polys_on_image(
    corners,
    image,
    labels=None,
    box_padding=-1,
    label_offset=1,
    label_font_size=10,
    color: str | list = "red",
):
    draw = ImageDraw.Draw(image)
    font_path = get_font_path()
    label_font = ImageFont.truetype(font_path, label_font_size)

    for i in range(len(corners)):
        poly = corners[i]
        poly = [(int(p[0]), int(p[1])) for p in poly]
        draw.polygon(
            poly, outline=color[i] if isinstance(color, list) else color, width=1
        )

        if labels is not None:
            label = labels[i]
            text_position = (
                min([p[0] for p in poly]) + label_offset,
                min([p[1] for p in poly]) + label_offset,
            )
            text_size = get_text_size(label, label_font)
            box_position = (
                text_position[0] - box_padding + label_offset,
                text_position[1] - box_padding + label_offset,
                text_position[0] + text_size[0] + box_padding + label_offset,
                text_position[1] + text_size[1] + box_padding + label_offset,
            )
            draw.rectangle(box_position, fill="white")
            draw.text(
                text_position,
                label,
                fill=color[i] if isinstance(color, list) else color,
                font=label_font,
            )

    return image
