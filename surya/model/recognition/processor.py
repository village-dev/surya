from typing import Dict, Union, Optional, List, cast

from torch import TensorType
from transformers import DonutImageProcessor, DonutProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import (
    PILImageResampling,
    ChannelDimension,
)
from PIL import Image
from surya.model.recognition.tokenizer import Byt5LangTokenizer
from surya.settings import settings
import torch
import torchvision
import torchvision.transforms.functional


def load_processor():
    processor = SuryaProcessor()
    processor.image_processor.train = False
    processor.image_processor.max_size = settings.RECOGNITION_IMAGE_SIZE
    processor.tokenizer.model_max_length = settings.RECOGNITION_MAX_TOKENS
    return processor


class SuryaImageProcessor(DonutImageProcessor):
    tokenizer: Byt5LangTokenizer

    def __init__(self, *args, max_size=None, train=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.patch_size = kwargs.get("patch_size", (4, 4))
        self.max_size = cast(dict[str, int], max_size)
        self.train = train

    # @classmethod
    # def numpy_resize(cls, image: np.ndarray, size, interpolation=cv2.INTER_LANCZOS4):
    #     max_width, max_height = size["width"], size["height"]

    #     resized_image = cv2.resize(
    #         image, (max_width, max_height), interpolation=interpolation
    #     )
    #     resized_image = resized_image.transpose(2, 0, 1)

    #     return resized_image

    def process_inner(self, images: List[Image.Image]):
        images = [
            SuryaImageProcessor.align_long_axis(image, size=self.max_size)
            for image in images
        ]
        # images = [
        #     x.resize(
        #         (self.max_size["width"], self.max_size["height"]),
        #         Image.Resampling.BICUBIC,
        #     )
        #     for x in images
        # ]

        image_arrays = [
            torchvision.transforms.functional.pil_to_tensor(img).cuda()
            for img in images
        ]

        image_arrays = [
            torchvision.transforms.functional.resize(
                img,
                [self.max_size["height"], self.max_size["width"]],
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
                mean=self.image_mean,
                std=self.image_std,
            )
            for img in image_arrays
        ]

        return image_arrays

    def preprocess(
        self,
        images: list[Image.Image],
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
    ):
        # Convert to numpy for later processing steps
        # images = [torch.tensor(np.array(img)) for img in images]

        # Convert to torch
        images = self.process_inner(images)

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    @classmethod
    def pad_image(
        cls,
        image: torch.Tensor,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        output_height, output_width = size["height"], size["width"]

        # image is channels first
        input_height, input_width = image.shape[1], image.shape[2]

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        assert delta_width >= 0 and delta_height >= 0

        pad_top = delta_height // 2
        pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        # padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        padding = [pad_left, pad_top, pad_right, pad_bottom]
        # return pad(
        #     image,
        #     padding,
        #     data_format=data_format,
        #     input_data_format=input_data_format,
        #     constant_values=pad_value,
        # )
        return torchvision.transforms.functional.pad(image, padding, pad_value)

    # @classmethod
    # def align_long_axis(
    #     cls,
    #     image: np.ndarray,
    #     size: Dict[str, int],
    #     data_format: Optional[Union[str, ChannelDimension]] = None,
    #     input_data_format: Optional[Union[str, ChannelDimension]] = None,
    # ) -> np.ndarray:
    #     input_height, input_width = image.shape[:2]
    #     output_height, output_width = size["height"], size["width"]

    #     if (output_width < output_height and input_width > input_height) or (
    #         output_width > output_height and input_width < input_height
    #     ):
    #         image = np.rot90(image, 3)
    #     return image
    @classmethod
    def align_long_axis(
        cls,
        image: Image.Image,
        size: Dict[str, int],
    ) -> Image.Image:
        input_width, input_height = image.size
        output_height, output_width = size["height"], size["width"]

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = image.rotate(270, expand=True)
        return image


class SuryaProcessor(DonutProcessor):
    tokenizer: Byt5LangTokenizer
    image_processor: SuryaImageProcessor

    def __init__(self, image_processor=None, tokenizer=None, train=False, **kwargs):
        image_processor = SuryaImageProcessor.from_pretrained(
            settings.RECOGNITION_MODEL_CHECKPOINT
        )
        tokenizer = Byt5LangTokenizer()
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self._in_target_context_manager = False

    def __call__(self, *args, **kwargs):
        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        langs = kwargs.pop("langs", None)

        if len(args) > 0:
            images = args[0]
            args = args[1:]

        if images is None and text is None:
            raise ValueError(
                "You need to specify either an `images` or `text` input to process."
            )

        if images is not None:
            inputs = self.image_processor(images, *args, **kwargs)

        if text is not None:
            encodings = self.tokenizer(text, langs, **kwargs)

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            inputs["langs"] = encodings["langs"]
            return inputs
