import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple

import torchvision.transforms as transforms
import albumentations as A

class OCHumanDataset(torch.utils.data.Dataset):
    """
    OC Human Dataset Class
    """
    def __init__(self, root_dir, img_ids, transforms, train=True) -> None:
        """
        Constructor
        """
        self.root_dir = root_dir     
        self.img_ids = img_ids
        self.transforms = transforms
        self.train = train

    def __len__(self) -> int:
        """
        Function to get size of dataset
        """
        return len(self.img_ids)

    @staticmethod
    def _get_area(box: list) -> int:
        """
        Function to get area
        """
        x1, y1, x2, y2 = box
        return((x2 - x1) * (y2 - y1))

    @staticmethod
    def _augment(img, bboxes=[], masks=[], annotations_flag=False):
        """
        Function for image augmentation during training
        """
        if annotations_flag:
            # Human
            if len(bboxes) > 0:
                transform = A.Compose([
                    A.geometric.rotate.SafeRotate(limit=50, p=0.5, border_mode=cv2.BORDER_REPLICATE),
                    A.HorizontalFlip(p=0.5),
                    A.augmentations.geometric.transforms.Affine([0.8, 1], keep_ratio=True, p=0.7),
                    A.geometric.resize.LongestMaxSize(max_size=600)
                ], bbox_params=A.BboxParams(format='pascal_voc'))
                transformed = transform(
                              image=img,
                              masks=masks,
                              bboxes=bboxes
                            )
            # Background
            else:
                transform = A.Compose([
                    A.geometric.rotate.SafeRotate(limit=50, p=0.5, border_mode=cv2.BORDER_REPLICATE),
                    A.HorizontalFlip(p=0.5),
                    A.augmentations.geometric.transforms.Affine([0.8,1], keep_ratio=True, p=0.7),
                    A.geometric.resize.LongestMaxSize(max_size=600)
                ])
                transformed = transform(image=img)
        # Inference
        else:
            transform = A.Compose([
                    A.geometric.resize.LongestMaxSize(max_size=600)
                ])
            transformed = transform(image=img)

        return transformed


    def __getitem__(self, index: int) -> Tuple[np.array, dict]:
        """
        Get item method
        """
        tensor_transform = transforms.Compose([
                    transforms.ToTensor()
                ])

        image_id = self.img_ids[index]

        if self.train:
            img_path = os.path.join(self.root_dir, "images", "train", image_id + ".jpg")
        else:
            img_path = os.path.join(self.root_dir, "images", "test", image_id + ".jpg")

        img = np.array(Image.open(img_path))
        annotation_path = ""
        if self.train:
            annotation_path = os.path.join(self.root_dir, "annotations", "train", self.img_ids[index] + ".json")
        else:
            annotation_path = os.path.join(self.root_dir, "annotations", "test", self.img_ids[index] + ".json")
        
        if os.path.exists(annotation_path):
            
            with open(annotation_path) as f:
                image_id, annotations = json.load(f)
            bboxes = np.array([annotations["boxes"][i] + [annotations["labels"][i]] for i in range(len(annotations["labels"]))])
            masks = list(np.array(annotations["masks"]))

            transformed = self._augment(img, bboxes, masks, annotations_flag=True)
            transformed_image = transformed["image"]
            transformed_masks = transformed["masks"]
            transformed_bboxes = transformed["bboxes"]

            bboxes__ = np.array([bbox[:-1] for bbox in transformed_bboxes])
            area_tensor = torch.as_tensor(list(map(self._get_area, bboxes__)), dtype=torch.int64)
            bboxes_tensor = torch.as_tensor(bboxes__, dtype=torch.float32)

            height, width, channels = np.shape(transformed_image)
            # Create a black image
            x = height if height > width else width
            y = height if height > width else width
            resized_img = np.zeros((x, y, channels), np.uint8)
            resized_img[0 : int(y-(y-height)), 0 : int(x-(x-width))] = transformed_image

            resized_masks = []
            for i in range(len(transformed_masks)):
                temp = np.zeros((x, y, 1), np.uint8)
                temp[0 : int(y-(y-height)), 0 : int(x-(x-width))] = np.expand_dims(transformed_masks[i], -1)
                resized_masks.append(np.squeeze(temp))
            masks_tensor = torch.from_numpy(np.array(resized_masks))

            labels_tensor = torch.as_tensor(annotations["labels"], dtype=torch.int64)
            is_crowd_tensor = torch.as_tensor([False] * len(annotations["labels"]), dtype=torch.bool)

        else:
            transformed = self._augment(img)
            transformed_image = transformed["image"]
            height, width, channels = np.shape(transformed_image)
            # Create a black image
            x = height if height > width else width
            y = height if height > width else width
            resized_img = np.zeros((x, y, channels), np.uint8)
            resized_img[0 : int(y-(y-height)), 0 : int(x-(x-width))] = transformed_image

            bboxes_tensor = torch.as_tensor(np.array([]).reshape(-1, 4)) 
            area_tensor = torch.as_tensor([])
            masks_tensor = torch.as_tensor(np.array([[[0]]]))
            labels_tensor = torch.as_tensor([], dtype=torch.int64)
            is_crowd_tensor = torch.as_tensor([])

            #print(resized_img.shape)
        img_tensor = tensor_transform(Image.fromarray(resized_img))
        img_id_tensor = torch.as_tensor(int(image_id), dtype=torch.int64)

        target = {}
        target["boxes"] = bboxes_tensor
        target["labels"] = labels_tensor
        target["masks"] = masks_tensor
        target["image_id"] = img_id_tensor
        target["area"] = area_tensor
        target["iscrowd"] = is_crowd_tensor

        return img_tensor, target
