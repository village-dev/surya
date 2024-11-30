from typing import Dict, Union, Optional, List, Iterable

import cv2
from torch import TensorType
from transformers import DonutImageProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import pad, normalize
from transformers.image_utils import PILImageResampling, ImageInput, ChannelDimension, make_list_of_images, get_image_size
import numpy as np
from PIL import Image
import PIL
from surya.settings import settings
import torchvision
import torch

class SuryaEncoderImageProcessor(DonutImageProcessor):
    def __init__(self, *args, max_size=None, align_long_axis=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.patch_size = kwargs.get("patch_size", (4, 4))
        self.max_size = max_size
        self.do_align_long_axis = align_long_axis

    @classmethod
    def numpy_resize(cls, image: np.ndarray, size, interpolation=cv2.INTER_LANCZOS4):
        max_width, max_height = size["width"], size["height"]

        resized_image = cv2.resize(image, (max_width, max_height), interpolation=interpolation)
        resized_image = resized_image.transpose(2, 0, 1)

        return resized_image

    def process_inner(self, images: List[Image.Image], model_device: torch.device):
        images = [
            SuryaEncoderImageProcessor.align_long_axis(image, size=self.max_size)  # type: ignore
            for image in images
        ]

        image_arrays = [
            torchvision.transforms.functional.pil_to_tensor(img).to(model_device)
            for img in images
        ]

        image_arrays = [
            torchvision.transforms.functional.resize(
                img,
                [self.max_size["height"], self.max_size["width"]],  # type: ignore
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
            for img in image_arrays
        ]

        # Rescale and normalize
        for idx in range(len(image_arrays)):
            image_arrays[idx] = image_arrays[idx] * self.rescale_factor

        image_arrays = [
            torchvision.transforms.functional.normalize(
                img,
                mean=self.image_mean,  # type: ignore
                std=self.image_std,  # type: ignore
            )
            for img in image_arrays
        ]

        return image_arrays

    def preprocess(
        self,
        images: ImageInput,
        model_device: torch.device = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_pad: bool = None,
        random_padding: bool = False,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> PIL.Image.Image:
        # Convert to numpy for later processing steps
        images = self.process_inner(images, model_device)

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    @classmethod
    def pad_image(
        cls,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        output_height, output_width = size["height"], size["width"]
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        assert delta_width >= 0 and delta_height >= 0

        pad_top = delta_height // 2
        pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        padding = [pad_left, pad_top, pad_right, pad_bottom]
        return torchvision.transforms.functional.pad(image, padding, pad_value)

    @classmethod
    def align_long_axis(  # type: ignore
        cls,
        image: Image.Image,
        size: dict[str, int],
    ) -> Image.Image:
        input_width, input_height = image.size
        output_height, output_width = size["height"], size["width"]

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = image.rotate(270, expand=True)
        return image
