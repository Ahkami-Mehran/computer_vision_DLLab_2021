import os
import random
import argparse
import torch
from pprint import pprint
from torchvision.transforms import *
from torch.utils.tensorboard import SummaryWriter
from models.pretraining_backbone import ResNet18Backbone
from data.pretraining import DataReaderPlainImg, custom_collate
from utils import (
    check_dir,
    accuracy,
    get_logger,
    mIoU,
    instance_mIoU,
    set_random_seed,
    save_in_log,
)
import os
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
import torchvision
import datetime

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# import torchvision.transforms as transforms


from pprint import pprint
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
    # model'
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, "NN")
    writer = SummaryWriter('results/NN/tensorboard/lr0.005_bs8_size256_')

    # build model and load weights
    model = torch.load(args.weights_init)

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
        # collate_fn=custom_collate,
    )
    print(2)
    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.
    query_indices = [10]
    nns = []
    # for idx, img in enumerate(val_loader):
    k = 5
    for item in enumerate(val_loader):
        idx, img = item[0], item[1]
        if idx not in query_indices:
            continue
        print("Computing NNs for sample {}".format(idx))
        closest_idx, closest_dist = find_nn(model, img, val_loader, k)
        print(closest_idx)
        print(closest_dist)
        print(3)
        # raise NotImplementedError(
        #     "TODO: retrieve the original NN images, save them and log the results."
        # )
        nns = img[closest_idx[0].type(torch.LongTensor)]
        print(nns[0].shape)
        print(nns)
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
    for item in enumerate(loader):
        idx, inputs = item[0], item[1]
        inputs = inputs.to(device)
        query_features = model.features(query_img)["out"].flatten()
        data_features = model.features(inputs)["out"].flatten()
        l2_distance = torch.dist(data_features, query_features)
        closest_dist.append(l2_distance)
        # closest_idx.append(idx)
    closest_idx, closest_dist = torch.sort(torch.cat([t.view(-1) for t in closest_dist]))
    return closest_idx[0:k], closest_dist[0:k]

def knn(ref, query, k):
    ref_c =torch.stack([ref] * query.shape[-1], dim=0).permute(0, 2, 1).reshape(-1, 2).transpose(0, 1)
    query_c = torch.repeat_interleave(query, repeats=ref.shape[-1], dim=1)
    delta = query_c - ref_c
    distances = torch.sqrt(torch.pow(delta, 2).sum(dim=0))
    distances = distances.view(query.shape[-1], ref.shape[-1])
    sorted_dist, indices = torch.sort(distances, dim=-1)
    return sorted_dist[:, :k], indices[:, :k]

if __name__ == "__main__":
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
