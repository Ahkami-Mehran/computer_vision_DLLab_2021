import os
import random
import sys
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint

from utils.meters import AverageValueMeter
from utils.weights import load_from_weights
from utils import check_dir, set_random_seed, accuracy, instance_mIoU, mIoU, get_logger
from models.second_segmentation import Segmentator
from data.transforms import get_transforms_binary_segmentation
from models.pretraining_backbone import ResNet18Backbone
from data.segmentation import DataReaderSemanticSegmentation
from torch.utils.tensorboard import SummaryWriter


sys.path.insert(0, os.getcwd())
set_random_seed(0)
global_step = 0

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIL = False

# Use the dt_binary_ss as a starting point for your file
# start you lr at 0.001, try more if you have time
# Use DataReaderSemanticSegmentation as dataloader
# Use utils.instance_mIoU to evaluate your model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", type=str, help="folder containing the data")
    parser.add_argument("--pretrained_model_path", type=str, default="")
    parser.add_argument("--output-root", type=str, default="results")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--bs", type=int, default=32, help="batch_size")
    parser.add_argument(
        "--att",
        type=str,
        default="sdotprod",
        help="Type of attention. Choose from {additive, cosine, dotprod, sdotprod}",
    )
    parser.add_argument("--size", type=int, default=256, help="image size")
    parser.add_argument(
        "--snapshot-freq", type=int, default=5, help="how often to save models"
    )
    parser.add_argument("--exp-suffix", type=str, default="")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs", "att"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(
        os.path.join(args.output_root, "dt_multiclass", args.exp_name)
    )
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # logger and tensorboard writer
    logger = get_logger(args.logs_folder, args.exp_name)
    writer = SummaryWriter(os.path.join(args.output_folder, "tensorboard"))

    # Image size
    img_size = (args.size, args.size)

    # Model

    # pretrained_model = load_from_weights(pretrained_model, args.weights_init, logger=logger)
    saved_config = torch.load(args.pretrained_model_path, map_location=DEVICE)

    # Create model
    pretrained_model = ResNet18Backbone(pretrained=False).to(DEVICE)
    model = Segmentator(6, pretrained_model.features, img_size).to(DEVICE)

    # Load saved model
    # saved_config = torch.load(args.weights_init, map_location=DEVICE)
    # model.load_state_dict(saved_config['model'])
    model = load_from_weights(model, args.pretrained_model_path, logger=logger)

    # dataset
    (
        train_trans,
        val_trans,
        train_target_trans,
        val_target_trans,
    ) = get_transforms_binary_segmentation(args)
    data_root = args.data_folder
    train_data = DataReaderSemanticSegmentation(
        os.path.join(data_root, "imgs/train2014"),
        os.path.join(data_root, "aggregated_annotations_train_5classes.json"),
        transform=train_trans,
        target_transform=train_target_trans,
    )
    val_data = DataReaderSemanticSegmentation(
        os.path.join(data_root, "imgs/val2014"),
        os.path.join(data_root, "aggregated_annotations_val_5classes.json"),
        transform=val_trans,
        target_transform=val_target_trans,
    )
    # subset of data
    if DEVICE.type == "cpu":
        train_data = torch.utils.data.Subset(train_data, np.arange(50))  # TODO: REMOVE
        val_data = torch.utils.data.Subset(val_data, np.arange(30))  # TODO: REMOVE
    elif TRAIL == True:
        train_data = torch.utils.data.Subset(
            train_data, np.arange(2000)
        )  # TODO: REMOVE
        val_data = torch.utils.data.Subset(val_data, np.arange(500))  # TODO: REMOVE

    logger.info("Dataset size: {} samples".format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.bs,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        drop_last=False,
    )

    # Criterion
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
    best_val_miou = 0.0
    for epoch in range(100):
        logger.info("Epoch {}".format(epoch))
        # Train
        t_loss = train(train_loader, model, criterion, optimizer, logger, epoch)
        writer.add_scalar(
            tag="Training/Mean_loss", scalar_value=t_loss, global_step=epoch
        )
        # Validate
        v_loss, v_mIoU = validate(val_loader, model, criterion, logger, epoch)
        writer.add_scalar(
            tag="Validation/Mean_Loss", scalar_value=v_loss, global_step=epoch
        )
        writer.add_scalar(
            tag="Validation/Mean_IoU", scalar_value=v_mIoU, global_step=epoch
        )

        # TODO save model
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_miou = v_mIoU
            save_model(
                model,
                optimizer,
                args,
                epoch,
                best_val_loss,
                best_val_miou,
                logger,
                best=True,
            )


def train(loader, model, criterion, optimizer, logger, epoch):
    # raise NotImplementedError("TODO: training routine")
    loss_meter = AverageValueMeter()
    model.train()
    for i, data in enumerate(loader, 0):
        # len(loader) gives the number of the bataches
        # len(loader.dataset) gives the number of datapoints in a batch
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        labels_unique = torch.unique(labels)
        for i, v in enumerate(labels_unique):
            pos = torch.where(labels == v)
            labels[pos] = torch.tensor(i, dtype=torch.float, device=DEVICE)
        labels = labels.type(torch.LongTensor)
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.squeeze(dim=1).type(torch.LongTensor).to(DEVICE)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_meter.add(loss.item())
        if DEVICE.type == "cpu":
            if i % 2 == 0:
                logger.info(
                    "Training: [epoch:%d, batch: %5d/%d] loss: %.3f"
                    % (epoch + 1, i + 1, len(loader), loss_meter.mean)
                )
        else:
            if i % 30 == 29:
                logger.info(
                    "Training: [epoch:%d, batch: %5d/%d] loss: %.3f"
                    % (epoch + 1, i + 1, len(loader), loss_meter.mean)
                )
    return loss_meter.mean


def validate(loader, model, criterion, logger, epoch=0):
    # raise NotImplementedError("TODO: validation routine")
    loss_meter = AverageValueMeter()
    iou_meter = AverageValueMeter()
    model.eval()
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        labels_unique = torch.unique(labels)
        for i, v in enumerate(labels_unique):
            pos = torch.where(labels == v)
            labels[pos] = torch.tensor(i, dtype=torch.float, device=DEVICE)
        labels = labels.type(torch.LongTensor)
        outputs = model(inputs)
        labels = labels.squeeze(dim=1).type(torch.LongTensor).to(DEVICE)
        outputs = F.interpolate(outputs, size=labels.shape[1:3])
        loss = criterion(outputs, labels)
        iou_meter.add(instance_mIoU(outputs, labels))
        loss_meter.add(loss.item())
        if DEVICE.type == "cpu":
            if i % 2 == 0:
                logger.info(
                    "Validation: [epoch:%d, batch: %5d/%d] loss: %.3f , mean_IoU: %.3f"
                    % (
                        epoch + 1,
                        i + 1,
                        len(loader),
                        loss_meter.mean,
                        iou_meter.mean,
                    )
                )
        else:
            if i % 30 == 29:
                logger.info(
                    "Validation: [epoch:%d, batch: %5d/%d] loss: %.3f , mean_IoU: %.3f"
                    % (
                        epoch + 1,
                        i + 1,
                        len(loader),
                        loss_meter.mean,
                        iou_meter.mean,
                    )
                )
    # in case of not matching dimentions, use F.interpolate to convert them
    return loss_meter.mean, iou_meter.mean


def save_model(model, optimizer, args, epoch, val_loss, val_iou, logger, best=False):
    # save model
    add_text_best = "BEST" if best else ""
    logger.info(
        "==> Saving "
        + add_text_best
        + " ... epoch{} loss{:.03f} miou{:.03f} ".format(epoch, val_loss, val_iou)
    )
    state = {
        "opt": args,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": val_loss,
        "miou": val_iou,
    }
    if best:
        torch.save(state, os.path.join(args.model_folder, "ckpt_best.pth"))
    else:
        torch.save(
            state,
            os.path.join(
                args.model_folder,
                "ckpt_epoch{}_loss{:.03f}_miou{:.03f}.pth".format(
                    epoch, val_loss, val_iou
                ),
            ),
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
