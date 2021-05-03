import os
import random
import argparse
import torch
from pprint import pprint
from torchvision.transforms import *
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter
from models.pretraining_backbone import ResNet18Backbone
from data.pretraining import DataReaderPlainImg, custom_collate

import numpy as np
import matplotlib.pyplot as plt
import torchvision
import datetime

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data.pretraining import DataReaderPlainImg, custom_collate
from data.transforms import get_transforms_pretraining
from utils import (
    check_dir,
    accuracy,
    get_logger,
    mIoU,
    instance_mIoU,
    set_random_seed,
    save_in_log,
)
from utils.weights import load_from_weights
from models.pretraining_backbone import ResNet18Backbone

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-init", type=str, default="")
    parser.add_argument(
        "--size", type=int, default=256, help="size of the images to feed the network"
    )
    parser.add_argument("--output-root", type=str, default="results")
    args = parser.parse_args()

    args.output_folder = check_dir(
        os.path.join(
            args.output_root,
            "nearest_neighbors",
            args.weights_init.replace("/", "_").replace("models", ""),
        )
    )
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.logs_folder, "nearest_neighbors")

    # build model and load weights
    model = ResNet18Backbone(pretrained=False).to(device)
    
    # load model
    saved_config = torch.load(args.weights_init, map_location=device)
    if type(saved_config) == dict:
        model = load_from_weights(model, args.weights_init)
    else:
        model.load_state_dict(saved_config.state_dict())

    # load dataset
    data_root = "./crops/images"
    # dataset
    val_transform = Compose(
        [Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()]
    )

    val_data = DataReaderPlainImg(
        os.path.join(data_root, str(args.size), "val"), transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.
    query_indices = [10]
    nns = []
    # for idx, img in enumerate(val_loader):
    k = 5
    for idx, img in enumerate(val_loader):
        # idx, img = item[0], item[1]

        if idx not in query_indices:
            continue
        logger.info("Computing NNs for sample {}".format(idx))
        closest_idx, closest_dist = find_nn(model, img, val_loader, k)
        nn_img_path = os.path.join(args.output_folder, "nn_img", "image_{}".format(query_indices)) 
        check_dir(nn_img_path)
        query_img = val_loader.dataset[query_indices]
        logger.info(query_img.shape)
        save_image(query_img, os.path.join(nn_img_path, "image_orig.png"))
        for i, nn_img_idx in enumerate(closest_idx):
            nn_img = val_loader[nn_img_idx]
            save_image(nn_img, os.path.join(nn_img_path, "num_{}_image_{}.png".format(i, nn_img_idx)))
        # raise NotImplementedError(
        #     "TODO: retrieve the original NN images, save them and log the results."
        # 
        # )
        # nns = val_loader.dataset[closest_idx.item().type(torch.LongTensor)]
        # nns = val_loader.dataset[closest_idx.item()]
        # logger.info(nns.shape)
        # torchvision.utils.save_image() # TODO: save images in another folder.
        # logger.info(nns)
        # temp_img = torchvision.utils.make_grid(images[:k,:,:,:])
        # temp_img = temp_img / 2 + 0.5     # unnormalize
        # npimg = temp_img.numpy()
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show()


def find_nn(model, query_img, loader, k):
    """
    Find the k nearest neighbors (NNs) of a query image, in the feature space of the specified mode.
    Args:
         model: the model for computing the features
         query_img: the image of which to find the NNs
         loader: the loader for the dataset in which to look for the NNs
         k: the number of NNs to retrieve
    Returns:
        closest_idx: the indices of the NNs in the dataset, for retrieving the images
        closest_dist: the L2 distance of each NN to the features of the query image
    """
    closest_idx = []
    closest_dist = []
    model.eval()
    for idx, inputs in enumerate(loader):
        if torch.equal(inputs, query_img):
            continue
        # idx, inputs = item[0], item[1]
        inputs = inputs.to(device)
        query_features = model.features(query_img)["out"].flatten()
        data_features = model.features(inputs)["out"].flatten()
        l2_distance = torch.dist(data_features, query_features).item()
        closest_dist.append(l2_distance)
        # closest_idx.append(idx)
    closest_idx, closest_dist = torch.sort(torch.cat([t.view(-1) for t in closest_dist]))
    return closest_idx[0:k], closest_dist[0:k]

if __name__ == "__main__":
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
