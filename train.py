import os
import argparse
import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as opt
from learnerable_seg import PromptSAM, PromptDiNo
from scheduler import PolyLRScheduler
from metrics.metric import Metric
from datasets import ImgDataset
from lovasz import LovaszSoftmax


parser = argparse.ArgumentParser("Learnable prompt")
parser.add_argument("--path", type=str, required=True,
                    help="path to the training data")
parser.add_argument("--name", type=str, required=True,
                    help="name of the output file the weights should be saved in")
parser.add_argument("--epoch", type=int, default=32, 
                    help="training epochs")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the checkpoint of sam")
parser.add_argument("--model_name", default="default", type=str,
                    help="name of the sam model, default is vit_h",
                    choices=["default", "vit_b", "vit_l", "vit_h"])
parser.add_argument("--save_path", type=str, default="./chkpts",
                    help="path where model weights are saved")
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--mix_precision", action=argparse.BooleanOptionalAction, default=False,
                    help="whether mixed precision training is used")
parser.add_argument("--use_lovasz", action=argparse.BooleanOptionalAction, default=False,
                    help="whether the Lovasz loss is used")
parser.add_argument("--resize_imgs", action=argparse.BooleanOptionalAction, default=False,
                    help="whether input images are resized rather than cropped")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--optimizer", default="sgd", type=str,
                    help="optimizer used to train the model")
parser.add_argument("--weight_decay", default=5e-4, type=float, 
                    help="weight decay for the optimizer")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="momentum for the sgd")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_workers", "-j", type=int, default=1, 
                    help="number of workers")
parser.add_argument("--device", default="0", type=str)
parser.add_argument("--model_type", default="sam", choices=["dino", "sam"], type=str,
                    help="backbone type")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device


def main(args):
    """Main testing function

    :param args: Arguments from the command line. Use "python train.py --help" to obtain a full list.
    """

    name = args.name
    path = args.path
    epochs = args.epoch
    checkpoint = args.checkpoint
    model_name = args.model_name
    save_path = pathlib.Path(args.save_path)
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    lr = args.lr
    momentum = args.momentum
    bs = args.batch_size
    num_workers = args.num_workers
    model_type = args.model_type
    save_path.mkdir(exist_ok=True)
    num_classes = args.num_classes
    mix_precision = args.mix_precision
    use_lovasz = args.use_lovasz
    resize_imgs = args.resize_imgs
    img_size = 1024
    model = None
    if model_type == "sam":
        model = PromptSAM(model_name, checkpoint=checkpoint, num_classes=num_classes, reduction=4,
                          upsample_times=2, groups=4)
    elif model_type == "dino":
        model = PromptDiNo(name=model_name, checkpoint=checkpoint, num_classes=num_classes)
        img_size = 518
    dataset = ImgDataset(path=path, is_fovea=True, img_size=img_size, batch_size=bs, use_resize=resize_imgs)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optim = None
    if optimizer == "adamw":
        optim = opt.AdamW([{"params": model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optim = opt.SGD([{"params": model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay,
                        momentum=momentum, nesterov=True)
    if use_lovasz:
        loss_func = LovaszSoftmax()
    else:
        loss_func = torch.nn.CrossEntropyLoss()
    scheduler = PolyLRScheduler(optim, num_images=dataset.length, batch_size=bs, epochs=epochs)
    metric = Metric(num_classes=num_classes)
    best_iou = 0.
    for epoch in range(epochs):
        for i, record in enumerate(dataloader):
            x = record['img'].to(device)
            target = record['seg'].to(device, dtype=torch.long)
            optim.zero_grad()
            if device_type == "cuda" and mix_precision:
                x = x.to(dtype=torch.float16)
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    pred = model(x)
                    loss = loss_func(pred, target)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                x = x.to(dtype=torch.float32)
                pred = model(x)
                loss = loss_func(pred, target)
                loss.backward()
                optim.step()
            metric.update(torch.softmax(pred, dim=1), target)
            print(f"epoch:{epoch}-{i}: loss:{loss.item()}", flush=True)
            scheduler.step()
        iou = np.nanmean(metric.evaluate()["iou"][1:].numpy())
        print(f"epoch-{epoch}: iou:{iou.item()}", flush=True)
        if iou > best_iou and (epoch + 1) % 100 == 0:
            print(f"New best iou, epoch {epoch}, iou {iou}", flush=True)
            best_iou = iou
            torch.save(model.state_dict(), save_path / f"{model_type}_{model_name}_e{epochs}-bs{bs}_{name}.pth")


if __name__ == "__main__":
    main(args)
