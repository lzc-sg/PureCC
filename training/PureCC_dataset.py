from torch.utils.data import Dataset
from PIL import Image
import torch
import random
import itertools
from pathlib import Path
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from torchvision.transforms.functional import crop
import pandas as pd
import os
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


class PureCC(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        data_path,
        csv_name,
        size=1024,
        repeats=1,
        center_crop=False,
        custom_instance_prompts=True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.custom_instance_prompts = custom_instance_prompts
        self.data_path = data_path
        self.csv_name = csv_name

        self.data = pd.read_csv(os.path.join(data_path, csv_name))
        # 取前10行
        self.data = self.data.head(6)

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        
        # image processing to prepare for using SD-XL micro-conditioning
        self.original_sizes = []
        self.crop_top_lefts = []
        self.pixel_values = []

        self.random_flip = True
        self.center_crop = False
        self.resolution = size

        interpolation = getattr(transforms.InterpolationMode, 'lanczos'.upper(), None)
        if interpolation is None:
            raise ValueError(f"Unsupported interpolation mode {interpolation=}.")
        self.train_resize = transforms.Resize(size, interpolation=interpolation)
        self.train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.num_instance_images = len(self.data)
        self._length = self.num_instance_images


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        item = self.data.iloc[index]
        image_path = item['image_path']
        image = Image.open(image_path)
        # if '[v]' not in item['prompt']:
        #     prompt = item['prompt']
        #     # 如果可以实现replace('a ', 'a [v] ')替换，则替换，否则使用 a [v] 替换
        #     if 'a' in prompt:
        #         prompt = prompt.replace('a', 'a [v]', 1)
        #     else:
        #         prompt = 'a [v] '
        # else:
        #     prompt = item['prompt']
        
        if not self.custom_instance_prompts:
            prompt = item['target_prompt']
        else:
            prompt = item['prompt']

        target_prompt = item['target_prompt']
        base_prompt = item['base_prompt']
        instance_word = item['instance_word']

        example['image'] = image

        image = exif_transpose(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        original_sizes = (image.height, image.width)
        image = self.train_resize(image)
        if self.random_flip and random.random() < 0.5:
            # flip
            image = self.train_flip(image)
        if self.center_crop:
            y1 = max(0, int(round((image.height - self.resolution) / 2.0)))
            x1 = max(0, int(round((image.width - self.resolution) / 2.0)))
            image = self.train_crop(image)
        else:
            y1, x1, h, w = self.train_crop.get_params(image, (self.resolution, self.resolution))
            image = crop(image, y1, x1, h, w)
        crop_top_left = (y1, x1)
        image = self.train_transforms(image)

        example['pixel_values'] = image
        example['target_prompts'] = target_prompt
        example['base_prompts'] = base_prompt
        example['prompts'] = prompt
        example['instance_word'] = instance_word
        example['crop_top_lefts'] = crop_top_left
        example['original_sizes'] = original_sizes

        return example
    
    def collate_fn(self, examples):
        pixel_values = [example['pixel_values'] for example in examples]
        target_prompts = [example['target_prompts'] for example in examples]
        base_prompts = [example['base_prompts'] for example in examples]
        prompt = [example['prompts'] for example in examples]
        instance_word = [example['instance_word'] for example in examples]
        crop_top_lefts = [example['crop_top_lefts'] for example in examples]
        original_sizes = [example['original_sizes'] for example in examples]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        return {
            'pixel_values': pixel_values,
            'target_prompts': target_prompts,
            'base_prompts': base_prompts,
            'prompts': prompt,
            'instance_word': instance_word,
            'crop_top_lefts': crop_top_lefts,
            'original_sizes': original_sizes
        }