import os
from argparse import ArgumentParser

import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ochuman_dataset import OCHumanDataset
from utils.misc import collate_fn
from utils.engine import evaluate, train_one_epoch

class MaskRCNNTrain():
    """
    Class containing functions for saving all frames of a video streamed over a network
    """
    def __init__(self) -> None:
        """
        Constructor
        """
        self._parser = None
        self._dataset_path = None

    def _define_arguments(self):
        """
        Function to define arguments
        """
        self._parser.add_argument(
            "--img_size",
            dest="img_size",
            action="store",
            type=int, 
            help="Dimensions of the image for training", 
            default=600
        )
        self._parser.add_argument(
            "--batch_size",
            dest="batch_size",
            action="store",
            type=int,
            help="Batch Size for training",
            default=8
        )
        self._parser.add_argument(
            "--train_path",
            dest="train_path",
            action="store",
            type=str,
            help="Path for training data",
            default="./dataset"
        )
        self._parser.add_argument(
            "--out_path",
            dest="out_path",
            action="store",
            type=str,
            help="Path to store trained weights",
            default="./out"
        )
        self._parser.add_argument(
            "--resume",
            dest="resume",
            action="store_true",
            help="Path to store trained weights"
        )
        self._parser.add_argument(
            "--model_path",
            dest="model_path",
            action="store",
            type=str,
            help="Path where training weights are stored, to resume training"
        )

    @staticmethod
    def _get_model(num_classes=1):
        """
        
        """
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)
        return model


    def train(self):
        """
        Train function
        """
        file_ids = []
        for _, _, file_names in os.walk(os.path.join(self._dataset_path, "images", "train")):
            for file_name in file_names:
                file_ids.append(file_name.split(".")[0])

        # use our dataset and defined transformations
        dataset = OCHumanDataset(
                        root_dir=self._dataset_path,
                        img_ids=file_ids,
                        transforms=None
                    )
        dataset_test = OCHumanDataset(
                        root_dir=self._dataset_path,
                        img_ids=file_ids,
                        transforms=None
                    )

        torch.manual_seed(1)

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=True, num_workers=4,
            collate_fn = collate_fn
            )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=3, shuffle=False, num_workers=4,
            collate_fn = collate_fn
            )

        device = torch.device('cuda')

        # our dataset has two classes only - background and person
        num_classes = 2

        # get the model using our helper function
        model = self._get_model(num_classes)
        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(params, lr=0.01)

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

        test_file_ids = []
        for _, _, file_names in os.walk(os.path.join("./dataset", "annotations", "test")):
            for file_name in file_names:
                test_file_ids.append(file_name.split(".")[0])

        dataset_test = OCHumanDataset(
                            root_dir="./dataset/",
                            img_ids=test_file_ids,
                            transforms=None,
                            train=False
                        )
        data_loader_test = torch.utils.data.DataLoader(
                            dataset_test, 
                            batch_size=8, 
                            shuffle=False, 
                            num_workers=4,
                            collate_fn = collate_fn
                        )

        num_epochs = 100

        for epoch in range(1, num_epochs + 1):
            # train for one epoch, printing every 100 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
            # update the learning rate
            lr_scheduler.step()
            # Save weights
            weights_path = os.path.join("out", "weights")
            if not os.path.exists(weights_path):
                os.makedirs(weights_path)
            if (epoch + 1) % 2 == 0 or epoch == 100:
                torch.save(model, os.path.join(weights_path, str(epoch) + ".pth"))
            with torch.no_grad():
            # evaluate on the test dataset
                evaluate(model, data_loader_test, device=device)

    def run(self):
        """
        Run function
        """
        self._parser = ArgumentParser()
        self._define_arguments()
        args = self._parser.parse_args()
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
        self._dataset_path = "./dataset/"
        self.train()

if __name__ == "__main__":
    MaskRCNNTrain().run()

