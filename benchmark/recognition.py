from collections import defaultdict
from typing import cast

import torch

from benchmark.scoring import overlap_score
from surya.input.processing import convert_if_not_rgb
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import (
    SuryaImageProcessor,
    load_processor as load_recognition_processor,
)
from torch.profiler import profile, ProfilerActivity
from surya.ocr import run_recognition
from surya.postprocessing.text import draw_text_on_image
from surya.settings import settings
from surya.languages import CODE_TO_LANGUAGE
import os
import datasets
import json
import time
from tabulate import tabulate
import click
from loguru import logger

KEY_LANGUAGES = [
    "Chinese",
    "Spanish",
    "English",
    "Arabic",
    "Hindi",
    "Bengali",
    "Russian",
    "Japanese",
]


@click.command()
@click.option(
    "--results_dir",
    default=os.path.join(settings.RESULT_DIR, "benchmark"),
    help="path to json file with ocr results",
)
@click.option("--max-pages", type=int, help="maximum number of pdf pages to ocr")
@click.option(
    "--debug",
    type=int,
    default=0,
    help="debug level - 1 dumps bad detection info, 2 writes out images",
)
@click.option("--tesseract", is_flag=True, help="run tesseract instead of surya")
@click.option("--langs", help="specify certain languages to benchmark")
@click.option(
    "--tess_cpus", type=int, default=28, help="number of cpus to use for tesseract"
)
@click.option("--compile", is_flag=True, help="compile the model")
@click.option("--profile-torch", is_flag=True, help="profile the model")
@click.option(
    "--specify_language", is_flag=True, help="pass language codes into the model"
)
def main(
    results_dir: str,
    max_pages: int,
    debug: int,
    tesseract: bool,
    langs: str,
    tess_cpus: int,
    compile: bool,
    profile_torch: bool,
    specify_language: bool,
):
    if compile:
        assert (
            settings.RECOGNITION_STATIC_CACHE
        ), "You must set RECOGNITION_STATIC_CACHE to compile the model."

    rec_model = load_recognition_model()
    rec_processor = cast(SuryaImageProcessor, load_recognition_processor())

    split = "train"
    if max_pages:
        split = f"train[:{max_pages}]"

    dataset = cast(
        datasets.Dataset,
        datasets.load_dataset(settings.RECOGNITION_BENCH_DATASET_NAME, split=split),
    )

    if langs:
        split_langs = langs.split(",")
        dataset = dataset.filter(lambda x: x["language"] in split_langs, num_proc=4)

    images = list(dataset["image"])
    images = convert_if_not_rgb(images)
    bboxes = list(dataset["bboxes"])
    line_text = list(dataset["text"])
    languages = list(dataset["language"])

    print(f"Loaded {len(images)} images. Running OCR...")

    lang_list: list[list[str]] = []
    for lang in languages:
        if not isinstance(lang, list):
            lang_list.append([lang])
        else:
            lang_list.append(lang)
    n_list = [None] * len(images)

    if compile:
        torch.set_float32_matmul_precision("high")
        torch._dynamo.config.cache_size_limit = 64
        rec_model.decoder.model = torch.compile(rec_model.decoder.model)
        # Run through one batch to compile the model
        run_recognition(
            images[:1], lang_list[:1], rec_model, rec_processor, bboxes=bboxes[:1]
        )

    start = time.time()

    if profile_torch:
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            profile_memory=False,
            record_shapes=False,
        ) as prof:
            predictions_by_image = run_recognition(
                images,
                lang_list if specify_language else n_list,
                rec_model,
                rec_processor,
                bboxes=bboxes,
            )
        prof.export_chrome_trace("trace.json")
    else:
        predictions_by_image = run_recognition(
            images,
            lang_list if specify_language else n_list,
            rec_model,
            rec_processor,
            bboxes=bboxes,
        )

    surya_time = time.time() - start

    logger.info(f"Time to process batch: {surya_time:.3f}s")

    surya_scores = defaultdict(list)
    img_surya_scores = []
    for idx, (pred, ref_text, lang) in enumerate(
        zip(predictions_by_image, line_text, lang_list)
    ):
        pred_text = [line.text for line in pred.text_lines]
        image_score = overlap_score(pred_text, ref_text)
        img_surya_scores.append(image_score)
        for lang in lang:
            surya_scores[CODE_TO_LANGUAGE[lang]].append(image_score)

    flat_surya_scores = [s for lang in surya_scores for s in surya_scores[lang]]
    benchmark_stats = {
        "surya": {
            "avg_score": sum(flat_surya_scores) / max(1, len(flat_surya_scores)),
            "lang_scores": {
                lang: sum(scores) / max(1, len(scores))
                for lang, scores in surya_scores.items()
            },
            "time_per_img": surya_time / max(1, len(images)),
        }
    }

    result_path = os.path.join(results_dir, "rec_bench")
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, "surya_scores.json"), "w+") as f:
        json.dump(surya_scores, f)

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(benchmark_stats, f)

    key_languages = [k for k in KEY_LANGUAGES if k in surya_scores]
    table_headers = ["Model", "Time per page (s)", "Avg Score"] + key_languages
    table_data = [
        [
            "surya",
            benchmark_stats["surya"]["time_per_img"],
            benchmark_stats["surya"]["avg_score"],
        ]
        + [benchmark_stats["surya"]["lang_scores"][lang] for lang in key_languages],
    ]

    print(tabulate(table_data, headers=table_headers, tablefmt="github"))
    print(
        "Only a few major languages are displayed. See the result path for additional languages."
    )

    if debug >= 1:
        bad_detections = []
        for idx, (score, lang) in enumerate(zip(flat_surya_scores, lang_list)):
            if score < 0.8:
                bad_detections.append((idx, lang, score))
        print(f"Found {len(bad_detections)} bad detections. Writing to file...")
        with open(os.path.join(result_path, "bad_detections.json"), "w+") as f:
            json.dump(bad_detections, f)

    if debug == 2:
        for idx, (image, pred, ref_text, bbox, lang) in enumerate(
            zip(images, predictions_by_image, line_text, bboxes, lang_list)
        ):
            pred_image_name = f"{'_'.join(lang)}_{idx}_pred.png"
            ref_image_name = f"{'_'.join(lang)}_{idx}_ref.png"
            pred_text = [line.text for line in pred.text_lines]
            pred_image = draw_text_on_image(bbox, pred_text, image.size, lang)
            pred_image.save(os.path.join(result_path, pred_image_name))
            ref_image = draw_text_on_image(bbox, ref_text, image.size, lang)
            ref_image.save(os.path.join(result_path, ref_image_name))
            image.save(os.path.join(result_path, f"{'_'.join(lang)}_{idx}_image.png"))

    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
