import torch
from typing import List, Tuple, Generator

from PIL import Image

from surya.model.detection.model import EfficientViTForSemanticSegmentation
from surya.model.detection.processor import SegformerImageProcessor
from surya.postprocessing.heatmap import get_and_clean_boxes
from surya.postprocessing.affinity import get_vertical_lines
from surya.input.processing import (
    prepare_image_detection,
    split_image,
    get_total_splits,
)
from surya.schema import TextDetectionResult
from surya.settings import settings
from tqdm import tqdm
import torch.nn.functional as F
from loguru import logger
from time import time


def get_batch_size():
    batch_size = settings.DETECTOR_BATCH_SIZE
    if batch_size is None:
        batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 36
    return batch_size


def batch_detection(
    images: List,
    model: EfficientViTForSemanticSegmentation,
    processor: SegformerImageProcessor,
    batch_size=None,
) -> Generator[
    Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[Tuple[int, int]],
    ],
    None,
    None,
]:
    start_time = time()
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
    logger.debug(f"Time to batch images: {time() - start_time:.3f}s")

    for batch_idx in tqdm(range(len(batches)), desc="Detecting bboxes"):
        start_time = time()
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
        logger.debug(f"Time to split images: {time() - start_time:.3f}s")

        start_time = time()
        image_splits = [
            prepare_image_detection(image, processor) for image in image_splits
        ]
        logger.debug(f"Time to prepare images: {time() - start_time:.3f}s")

        start_time = time()
        # Batch images in dim 0
        batch = torch.stack(image_splits, dim=0).to(model.dtype).to(model.device)
        logger.debug(f"Time to stack images: {time() - start_time:.3f}s")

        start_time = time()
        with torch.inference_mode():
            pred = model(pixel_values=batch)
        logger.debug(f"Time to run model: {time() - start_time:.3f}s")

        start_time = time()
        logits = pred.logits
        correct_shape = [processor.size["height"], processor.size["width"]]
        current_shape = list(logits.shape[2:])
        if current_shape != correct_shape:
            logits = F.interpolate(
                logits, size=correct_shape, mode="bilinear", align_corners=False
            )

        logger.debug(f"Time to interpolate: {time() - start_time:.3f}s")

        preds: List[torch.Tensor] = []
        pred_p90s: List[torch.Tensor] = []

        start_time = time()
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
                    pred_heatmaps = pred_heatmaps[:, :height, :]

                heatmaps.extend(pred_heatmaps)

                preds[idx] = heatmaps
                pred_p90s[idx] = torch.quantile(
                    torch.cat(heatmaps).view(len(heatmaps), -1).to(torch.float32),
                    0.9,
                    dim=1,
                )

        logger.debug(f"Time to process batch: {time() - start_time:.3f}s")

        segment_assignments = [
            heatmaps.cuda().argmax(dim=0).cpu().detach() for heatmaps in preds
        ]
        preds = [x.cpu().detach() for x in preds]
        pred_p90s = [x.cpu().detach() for x in pred_p90s]

        yield (
            preds,
            segment_assignments,
            pred_p90s,
            [orig_sizes[j] for j in batch_image_idxs],
        )


def parallel_get_lines(
    preds: Tuple[torch.Tensor, torch.Tensor],
    pred_p90s: Tuple[torch.Tensor, torch.Tensor],
    orig_sizes: Tuple[int, int],
):
    heatmap, affinity_map = preds[0], preds[1]
    heatmap_p90, affinity_map_p90 = pred_p90s[0], pred_p90s[1]

    heat_img = Image.fromarray((heatmap * 255).to(torch.uint8).numpy())
    aff_img = Image.fromarray((affinity_map * 255).to(torch.uint8).numpy())

    affinity_size = list(reversed(affinity_map.shape))
    heatmap_size = list(reversed(heatmap.shape))

    bboxes = get_and_clean_boxes(
        heatmap.numpy(), heatmap_p90.item(), heatmap_size, orig_sizes
    )

    vertical_lines = get_vertical_lines(affinity_map, affinity_size, orig_sizes)

    result = TextDetectionResult(
        bboxes=bboxes,
        vertical_lines=vertical_lines,
        heatmap=heat_img,
        affinity_map=aff_img,
        image_bbox=[0, 0, orig_sizes[0], orig_sizes[1]],
    )
    return result


def batch_text_detection(
    images: List[Image.Image],
    model: EfficientViTForSemanticSegmentation,
    processor: SegformerImageProcessor,
    batch_size=None,
) -> List[TextDetectionResult]:
    detection_generator = batch_detection(
        images, model, processor, batch_size=batch_size
    )

    postprocessing_futures = []
    max_workers = min(settings.DETECTOR_POSTPROCESSING_CPU_WORKERS, len(images))
    parallelize = (
        not settings.IN_STREAMLIT
        and len(images) >= settings.DETECTOR_MIN_PARALLEL_THRESH
    )

    for preds, segment_assignments, pred_p90s, orig_sizes in detection_generator:
        start_time = time()
        for pred, segment_assignment, pred_p90, orig_size in zip(
            preds, segment_assignments, pred_p90s, orig_sizes
        ):
            postprocessing_futures.append(
                parallel_get_lines(
                    (pred[0], pred[1]), (pred_p90[0], pred_p90[1]), orig_size
                )
            )
        logger.debug(f"Time to postprocess: {time() - start_time:.3f}s")
    return postprocessing_futures

    # with ProcessPoolExecutor(
    #     max_workers=max_workers,
    # ) if parallelize else contextlib.nullcontext() as executor:
    #     func = executor.submit if parallelize else FakeParallel
    #     for preds, orig_sizes in detection_generator:
    #         for pred, orig_size in zip(preds, orig_sizes):
    #             postprocessing_futures.append(
    #                 func(parallel_get_lines, (pred[0], pred[1]), orig_size)
    #             )

    # return [future.result() for future in postprocessing_futures]
