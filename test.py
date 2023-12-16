import os
import argparse
import pathlib
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import ImgDataset, VidDataset, unprocess_img
from learnerable_seg import PromptSAM, PromptDiNo
from torchmetrics.functional import dice


parser = argparse.ArgumentParser("Learnable prompt")
parser.add_argument("--path", type=str, required=True,
                    help="path to folder containing images, or path to video file")
parser.add_argument("--is_fovea", action=argparse.BooleanOptionalAction, default=True,
                    help="whether the folder contains FOVEA images with seg")
parser.add_argument("--patched_pred", action=argparse.BooleanOptionalAction, default=False,
                    help="whether predictions are polled from multiple image patches")
parser.add_argument("--patched_threshold", type=float, default=0.,
                    help="threshold for prediction patching, if active")
parser.add_argument("--resize_imgs", action=argparse.BooleanOptionalAction, default=False,
                    help="whether input images are resized rather than cropped")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the checkpoint of sam")
parser.add_argument("--model_name", default="default", type=str,
                    help="name of the sam model, default is vit_h",
                    choices=["default", "vit_b", "vit_l", "vit_h"])
parser.add_argument("--save_path", type=str, required=True,
                    help="where output images or video files are saved")
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--mix_precision", action="store_true", default=False,
                    help="whether use mix precision training")
parser.add_argument("--num_workers", "-j", type=int, default=1,
                    help="number of workers")
parser.add_argument("--device", default="0", type=str)
parser.add_argument("--model_type", default="sam", choices=["dino", "sam"], type=str,
                    help="backbone type")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device


def run_model(
    x: torch.Tensor,
    model: torch.nn.Module,
    device_type: str
) -> torch.Tensor:
    """Run the model

    :param x: Model input as torch tensor, shape 1CHW (assumes batch size of 1!)
    :param model: The model to run the data on
    :param device_type: Device type, needed for torch.autocast
    :return: Model prediction as a torch tensor containing class numbers, shape HW
    """

    if device_type == "cuda" and args.mix_precision:
        x = x.to(dtype=torch.float16)
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            pred = model(x)
    else:
        x = x.to(dtype=torch.float32)
        pred = model(x)
    return torch.argmax(pred[0], dim=0)


def run_model_patched(
    x: torch.Tensor,
    model: torch.nn.Module,
    device_type: str,
    patch_size: int,
    threshold: float = 0.,
) -> torch.Tensor:
    """Runs the model on an image larger than the img_size set by the model by running five patches individually
    and polling a combined prediction from the resulting patches, taking into account the location in the
    original image. Patches are selected from each corner and as a central crop, resulting in five total.

    :param x: Model input as torch tensor, shape 1CHW (assumes batch size of 1!)
    :param model: The model to run the data on
    :param device_type: Device type, needed for torch.autocast
    :param patch_size: img_size set by the model, smaller than the HW input size of the data x
    :param threshold: Polling threshold above which a pixel is set to 1 in the final prediction. If set to 0, the
        polling is really just a bit-wise OR operation on all the patches, or the Union of all patch predictions
        (corrected for patch location). Setting it to e.g. 0.5 will mean that in regions with a single patch this
        sets the prediction, in regions with two patches both need to agree, in regions with three patches two out
        of three need to agree, etc. Defaults to 0
    :return: Model prediction as a torch tensor containing class numbers, shape HW
    """

    h, w = x.shape[-2], x.shape[-1]
    h_offset = (h - patch_size) // 2
    w_offset = (w - patch_size) // 2
    patch_slices = [
        (slice(patch_size), slice(patch_size)),
        (slice(patch_size), slice(-patch_size, None)),
        (slice(-patch_size, None), slice(patch_size)),
        (slice(-patch_size, None), slice(-patch_size, None)),
        (slice(h_offset, h_offset + patch_size), slice(w_offset, w_offset + patch_size))
    ]
    count = torch.zeros((h, w), device=x.device, dtype=torch.uint8)
    pred = torch.zeros((h, w), device=x.device, dtype=torch.float)
    for i, slices in enumerate(patch_slices):
        count[slices[0], slices[1]] += 1
        pred[slices[0], slices[1]] += run_model(x[..., slices[0], slices[1]], model, device_type)
    pred = (pred / count > threshold).to(torch.uint8)
    return pred


def main(args):
    """Main testing function

    :param args: Arguments from the command line. Use "python test.py --help" to obtain a full list.
    """

    path = args.path
    is_fovea = args.is_fovea
    patched_pred = args.patched_pred
    patched_threshold = args.patched_threshold
    checkpoint = args.checkpoint
    model_name = args.model_name
    save_path = args.save_path
    num_workers = args.num_workers
    model_type = args.model_type
    resize_imgs = args.resize_imgs
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_classes = args.num_classes
    img_size = 1024
    model = None
    if model_type == "sam":
        model = PromptSAM(model_name, checkpoint=None, num_classes=num_classes, reduction=4,
                          upsample_times=2, groups=4)
    elif model_type == "dino":
        model = PromptDiNo(name=model_name, checkpoint=None, num_classes=num_classes)
        img_size = 518
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    check_path = pathlib.Path(path)
    is_video = False
    vid_out = None
    if check_path.suffix == '':  # Directory, assume there are png images in there to check
        dataset = ImgDataset(path, is_fovea=is_fovea, img_size=img_size if not patched_pred else 0,
                             random_aug=False, use_resize=resize_imgs)
    elif check_path.suffix.lower()[1:] in ['mkv', 'avi', 'mp4', 'mov']:  # Video
        dataset = VidDataset(path, img_size=img_size if not patched_pred else 0, use_resize=resize_imgs)
        is_video = True
        vid_path_out = [
            pathlib.Path(save_path) / (check_path.stem + '_seg_pred.mkv'),
            pathlib.Path(save_path) / (check_path.stem + '_seg_pred_overlay.mkv'),
        ]
        vid_out = [
            cv2.VideoWriter(str(vid_path_out[0]), cv2.VideoWriter_fourcc(*'MJPG'),
                            dataset.fps, (img_size, img_size), isColor=False),
            cv2.VideoWriter(str(vid_path_out[1]), cv2.VideoWriter_fourcc(*'MJPG'),
                            dataset.fps, (img_size, img_size), isColor=True),
        ]

    else:
        raise ValueError(f"Path {path} is neither a .mkv video nor a directory")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    dice_vals = []
    conf_matrix_vals = []
    with torch.no_grad():
        for i, record in enumerate(dataloader):
            x = record['img'].to(device)
            if patched_pred:
                pred = run_model_patched(x, model, device_type, img_size, threshold=patched_threshold)
            else:
                pred = run_model(x, model, device_type)
            pred = pred.cpu()
            metric_string, gt_overlay = '', None
            if 'seg' in record:  # ground truth segs are available
                dice_vals.append(dice(pred, record['seg'], ignore_index=0).item())
                gt_overlay = record['seg'][0]
                gt_overlay = torch.stack([gt_overlay]*3, dim=0)  # Ground truth is now white (1, 1, 1)
                gt_overlay[0, pred > 0] = 0
                gt_overlay[2, pred > 0] = 0  # Ground truth where pred is now green (0, 1, 0)
                gt_overlay[2, torch.bitwise_and(gt_overlay[1] == 0, pred > 0)] = 1
                # Where no ground truth but pred, is now red (1, 0, 0)
                # --> green is TP, red is FP, white is FN, black is TN
                total_px = gt_overlay.shape[-2] * gt_overlay.shape[-1]
                tn = torch.sum(torch.bitwise_and(record['seg'][0] == 0, pred == 0)) / total_px
                fn = torch.sum(torch.bitwise_and(record['seg'][0] > 0, pred == 0)) / total_px
                tp = torch.sum(torch.bitwise_and(record['seg'][0] > 0, pred > 0)) / total_px
                fp = torch.sum(torch.bitwise_and(record['seg'][0] == 0, pred > 0)) / total_px
                conf_matrix_vals.append([tp, fp, fn, tn])
                gt_overlay = np.moveaxis(gt_overlay.numpy(), 0, -1) * 255
                metric_string = f', DICE {dice_vals[-1]:.4f}, TP {tp:.4f}, FP {fp:.4f}, FN {fn:.4f}, TN {tn:.4f}'
            pred = pred.numpy().astype('uint8') * 255
            orig_img = unprocess_img(x[0]).cpu().numpy()
            overlay = np.round(.3 * pred[..., np.newaxis] + .7 * orig_img).astype('uint8')
            if is_video:
                vid_out[0].write(pred)
                vid_out[1].write(overlay)
            else:
                id_string = f'FOVEA{record["idx"].item():03d}_' if is_fovea else f'{i:03d}'
                img_path = pathlib.Path(save_path) / f'{id_string}seg.png'
                cv2.imwrite(str(img_path), pred)
                img_path = pathlib.Path(save_path) / f'{id_string}seg_img_overlay.png'
                cv2.imwrite(str(img_path), overlay[..., ::-1])
                if gt_overlay is not None:
                    img_path = pathlib.Path(save_path) / f'{id_string}seg_gt_overlay.png'
                    cv2.imwrite(str(img_path), gt_overlay)
            id_string = f'{i} / FOVEA{record["idx"].item():03d}' if 'idx' in record else f'{i:03d}'
            print(f'Item {id_string} processed{metric_string}')
        if is_video:
            [v.release() for v in vid_out]
            dataset.video.release()
        if 'seg' in record:
            v = np.mean(conf_matrix_vals, axis=0)
            print('Mean DICE: {:.4f}; TP {:.4f}, FP {:.4f}, FN {:.4f}, TN {:.4f}'.format(np.mean(dice_vals), *v))


if __name__ == "__main__":
    main(args)
