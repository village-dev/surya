import torch
from typing import List, Tuple, Generator

import numpy as np
from PIL import Image

from surya.model.detection.model import EfficientViTForSemanticSegmentation
from surya.postprocessing.heatmap import get_and_clean_boxes
from surya.postprocessing.affinity import get_vertical_lines
from surya.input.processing import prepare_image_detection, split_image, get_total_splits
from surya.schema import TextDetectionResult
from surya.settings import settings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F

from surya.util.parallel import FakeExecutor


def get_batch_size():
    batch_size = settings.DETECTOR_BATCH_SIZE
    if batch_size is None:
        batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 36
    return batch_size

def pad_to_batch_size(tensor, batch_size):
    current_batch_size = tensor.shape[0]
    if current_batch_size >= batch_size:
        return tensor

    pad_size = batch_size - current_batch_size
    padding = (0, 0) * (tensor.dim() - 1) + (0, pad_size)

    return F.pad(tensor, padding, mode='constant', value=0)

def batch_detection(
    images: List,
    model: EfficientViTForSemanticSegmentation,
    processor,
    batch_size=None,
    static_cache=False
) -> Generator[Tuple[List[List[np.ndarray]], List[Tuple[int, int]]], None, None]:
    assert all([isinstance(image, Image.Image) for image in images])
    if batch_size is None:
        batch_size = get_batch_size()
    heatmap_count = model.config.num_labels

    orig_sizes = [image.size for image in images]
    splits_per_image = [get_total_splits(size, processor) for size in orig_sizes]

    batches = []
    current_batch_size = 0
    current_batch = []
    for i in range(len(images)):
        if current_batch_size + splits_per_image[i] > batch_size:
            if len(current_batch) > 0:
                batches.append(current_batch)
            current_batch = []
            current_batch_size = 0
        current_batch.append(i)
        current_batch_size += splits_per_image[i]

    if len(current_batch) > 0:
        batches.append(current_batch)

    for batch_idx in tqdm(range(len(batches)), desc="Detecting bboxes"):
        batch_image_idxs = batches[batch_idx]
        batch_images = [images[j].convert("RGB") for j in batch_image_idxs]

        split_index = []
        split_heights = []
        image_splits = []
        for image_idx, image in enumerate(batch_images):
            image_parts, split_height = split_image(image, processor)
            image_splits.extend(image_parts)
            split_index.extend([image_idx] * len(image_parts))
            split_heights.extend(split_height)

        image_splits = [prepare_image_detection(image, processor, model.device) for image in image_splits]
        # Batch images in dim 0
        batch = torch.stack(image_splits, dim=0).to(model.dtype).to(model.device)
        if static_cache:
            batch = pad_to_batch_size(batch, batch_size)

        with torch.inference_mode():
            pred = model(pixel_values=batch)

        logits = pred.logits
        correct_shape = [processor.size["height"], processor.size["width"]]
        current_shape = list(logits.shape[2:])
        if current_shape != correct_shape:
            logits = F.interpolate(logits, size=correct_shape, mode='bilinear', align_corners=False)

        preds = []
        pred_p90s: list[torch.Tensor] = []
        
        for i, (idx, height) in enumerate(zip(split_index, split_heights)):
            # If our current prediction length is below the image idx, that means we have a new image
            # Otherwise, we need to add to the current image
            pred_heatmaps = logits[i][range(heatmap_count)]

            if len(preds) <= idx:
                preds.append(pred_heatmaps)
                pred_p90s.append(
                    torch.quantile(
                        pred_heatmaps.view(pred_heatmaps.shape[0], -1).to(
                            torch.float32
                        ),
                        0.9,
                        dim=1,
                    )
                )
            else:
                heatmaps = preds[idx]

                if height < processor.size["height"]:
                    # Cut off padding to get original height
                    pred_heatmaps = pred_heatmaps[:, :height, :]

                heatmaps = torch.cat([heatmaps, pred_heatmaps], dim=1)

                preds[idx] = heatmaps
                pred_p90s[idx] = torch.quantile(
                    heatmaps.view(len(heatmaps), -1).to(torch.float32),
                    0.9,
                    dim=1,
                )
                
        preds = [x.float().cpu().detach().numpy() for x in preds]
        pred_p90s = [x.float().cpu().detach().numpy() for x in pred_p90s]

        yield preds, pred_p90s, [orig_sizes[j] for j in batch_image_idxs]


def parallel_get_lines(preds, pred_p90s, orig_sizes, include_maps=False):
    heatmap_p90 = pred_p90s[0]
    heatmap, affinity_map = preds
    heat_img, aff_img = None, None
    if include_maps:
        heat_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        aff_img = Image.fromarray((affinity_map * 255).astype(np.uint8))
    affinity_size = list(reversed(affinity_map.shape))
    heatmap_size = list(reversed(heatmap.shape))
    bboxes = get_and_clean_boxes(heatmap, heatmap_p90, heatmap_size, orig_sizes)
    vertical_lines = get_vertical_lines(affinity_map, affinity_size, orig_sizes)

    result = TextDetectionResult(
        bboxes=bboxes,
        vertical_lines=vertical_lines,
        heatmap=heat_img,
        affinity_map=aff_img,
        image_bbox=[0, 0, orig_sizes[0], orig_sizes[1]]
    )
    return result


def batch_text_detection(images: List, model, processor, batch_size=None, include_maps=False) -> List[TextDetectionResult]:
    detection_generator = batch_detection(images, model, processor, batch_size=batch_size, static_cache=settings.DETECTOR_STATIC_CACHE)

    postprocessing_futures = []
    max_workers = min(settings.DETECTOR_POSTPROCESSING_CPU_WORKERS, len(images))
    parallelize = not settings.IN_STREAMLIT and len(images) >= settings.DETECTOR_MIN_PARALLEL_THRESH
    executor = ThreadPoolExecutor if parallelize else FakeExecutor
    with executor(max_workers=max_workers) as e:
        for preds, pred_p90s, orig_sizes in detection_generator:
            for pred, pred_p90, orig_size in zip(preds, pred_p90s, orig_sizes):
                postprocessing_futures.append(e.submit(parallel_get_lines, pred, pred_p90, orig_size, include_maps))

    return [future.result() for future in postprocessing_futures]
