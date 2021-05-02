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

global_step = 0

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIL = True

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_folder", type=str, help="folder containing the data (crops)"
    )
    parser.add_argument("--weights-init", type=str, default="random")
    parser.add_argument("--output-root", type=str, default="results")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--bs", type=int, default=8, help="batch_size")
    parser.add_argument(
        "--size", type=int, default=256, help="size of the images to feed the network"
    )
    parser.add_argument(
        "--snapshot-freq", type=int, default=1, help="how often to save models"
    )
    parser.add_argument(
        "--exp-suffix", type=str, default="", help="string to identify the experiment"
    )
    args = parser.parse_args()

    hparam_keys = ["lr", "bs", "size"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(
        os.path.join(args.output_root, "pretrain", args.exp_name)
    )
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.logs_folder, args.exp_name)
    writer = SummaryWriter(os.path.join(args.output_folder, 'tensorboard'))

    # build model and load weights
    model = ResNet18Backbone(pretrained=False).to(DEVICE)

    # load model
    # saved_config = torch.load(args.weights_init, map_location=DEVICE)
    # model.load_state_dict(saved_config["model"])
    model = load_from_weights(model, args.weights_init, logger=logger)

    # load dataset
    data_root = args.data_folder
    train_transform, val_transform = get_transforms_pretraining(args)
    train_data = DataReaderPlainImg(
        os.path.join(data_root, str(args.size), "train"), transform=train_transform
    )
    val_data = DataReaderPlainImg(
        os.path.join(data_root, str(args.size), "val"), transform=val_transform
    )

    # subset of data
    if DEVICE.type == 'cpu':
        train_data = torch.utils.data.Subset(train_data, np.arange(50)) # TODO: REMOVE
        val_data = torch.utils.data.Subset(val_data, np.arange(30)) # TODO: REMOVE
    elif TRAIL == True:
        train_data = torch.utils.data.Subset(train_data, np.arange(2000)) # TODO: REMOVE
        val_data = torch.utils.data.Subset(val_data, np.arange(500)) # TODO: REMOVE

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate,
    )

    # Visualize the sample batch of images with the labels
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # logger.info(images[1].shape)
    # temp_img = torchvision.utils.make_grid(images[:32,:,:,:])
    # temp_img = temp_img / 2 + 0.5     # unnormalize
    # npimg = temp_img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()

    # TODO: loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info("train_data {}".format(train_data.__len__()))
    logger.info("val_data {}".format(val_data.__len__()))

    best_val_loss = np.inf
    # Train-validate for one epoch. You don't have to run it for 100 epochs, preferably until it starts overfitting.
    model_count = 0
    for epoch in range(100):
        logger.info("Epoch {}".format(epoch))
        t_loss = train(train_loader, model, criterion, optimizer, epoch, logger)
        # training_loss.append(t_loss)
        writer.add_scalar(tag="Training/Mean_loss", scalar_value = t_loss, global_step = epoch)
        v_loss, v_acc = validate(val_loader, model, criterion, epoch, logger)
        # val_loss.append(v_loss)
        # val_acc.append(v_acc)
        writer.add_scalar(tag="Validation/Mean_Loss", scalar_value = v_loss, global_step = epoch)
        writer.add_scalar(tag="Validation/Mean_Accuracy", scalar_value = v_acc, global_step = epoch)
        # save model
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            # TODO: save only top model. due to disk space
            torch.save(model, os.path.join(args.model_folder,"model.pth"))
            logger.info("save model with on epoch{} and validation loss {}".format(epoch, best_val_loss))
            # raise NotImplementedError("TODO: save model if a new best validation error was reached")
        global_step = epoch


# train one epoch over the whole training dataset. You can change the method's signature.
def train(loader, model, criterion, optimizer, epoch, logger):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(loader, 0):
        # len(loader) gives the number of the bataches
        # len(loader.dataset) gives the number of datapoints in a batch
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if DEVICE.type == 'cpu':
            if i % 2 == 0:
                logger.info(
                    "Training: [epoch:%d, batch: %5d/%d] loss: %.3f"
                    % (epoch + 1, i + 1, len(loader), running_loss / len(loader.dataset))
                )
        else:
            if i % 50 == 49:
                logger.info(
                    "Training: [epoch:%d, batch: %5d/%d] loss: %.3f"
                    % (epoch + 1, i + 1, len(loader), running_loss / len(loader.dataset))
                )
    # return the running_loss for further evaluations
    return running_loss / len(loader)


# validation function. you can change the method's signature.
def validate(loader, model, criterion, epoch, logger):
    running_loss = 0.0
    acc = []
    model.eval()
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc.append(accuracy(outputs, labels)[0].item())
        running_loss += loss.item()
        if DEVICE.type == 'cpu':
            if i % 2 == 0:
                logger.info(
                    "Validation: [epoch:%d, batch: %5d/%d] loss: %.3f , accuracy: %.3f"
                    % (
                        epoch + 1,
                        i + 1,
                        len(loader),
                        running_loss / len(loader.dataset),
                        np.mean(acc),
                    )
                )
        else:
            if i % 10 == 9:
                logger.info(
                    "Validation: [epoch:%d, batch: %5d/%d] loss: %.3f , accuracy: %.3f"
                    % (
                        epoch + 1,
                        i + 1,
                        len(loader),
                        running_loss / len(loader.dataset),
                        np.mean(acc),
                    )
                )
    # return mean_val_loss, mean_val_accuracy
    return running_loss / (len(loader)), np.mean(acc)


if __name__ == "__main__":
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
