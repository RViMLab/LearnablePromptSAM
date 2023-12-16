from PIL import Image
import numpy as np
from albumentations import Compose, Resize, CenterCrop, RandomCrop, Normalize, ColorJitter, HorizontalFlip, VerticalFlip
import pathlib
import torch
import cv2
import random
from torchvision.transforms.functional import normalize


def unprocess_img(img: torch.Tensor, simple: bool = True, switch_dims: bool = True) -> torch.Tensor:
    """Undoes the original processing step, optionally moves CHW to HWC. Used to recover the original image from
    model input, e.g. to show it under the segmentation prediction

    :param img: Torch tensor input of shape CHW
    :param simple: Boolean, if True then using 0.5 for all pixel means and standard deviations. If False, uses
        ImageNet standards. Defaults to True
    :param switch_dims: Boolean, if True then the output will be HWC e.g. for use in NumPy. Defaults to True
    :return: CHW or HWC image as torch uint8 tensor with the original processing (normalisation) undone
    """

    if simple:
        mean = [.5, .5, .5]
        std = [.5, .5, .5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    un_mean = [-m / s for m, s in zip(mean, std)]
    un_std = [1 / s for s in std]
    img = normalize(img, mean=un_mean, std=un_std)
    img *= 255
    if switch_dims:
        img = torch.movedim(img, 0, -1)
    return torch.round(img).byte()


class VidDataset:
    def __init__(
        self,
        path: str,
        pixel_mean: tuple = (.5, .5, .5),
        pixel_std: tuple = (.5, .5, .5),
        img_size: int = 1024,
        use_resize: bool = False,
    ):
        """Dataset that loads a video from a path by iteratively serving the next frame, using cv2 (OpenCV)

        :param path: Full video path
        :param pixel_mean: Mean pixel values for image normalisation. Default 0.5 on all channels
        :param pixel_std: Pixel standard deviation values for image normalisation. Default 0.5 on all channels
        :param img_size: Required image size. If zero, it means no change. Default 1024 (PromptSAM model)
        :param use_resize: Boolean determining whether the input should be resized or cropped to achieve the
            required image size. Defaults to False, i.e. cropping is default behaviour
        """

        self.path = path
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.img_size = img_size
        self.use_resize = use_resize
        self.video = cv2.VideoCapture(self.path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        transforms_list = []
        if self.img_size > 0:
            if self.use_resize:
                transforms_list.append(Resize(self.img_size, self.img_size))
            else:
                transforms_list.append(CenterCrop(self.img_size, self.img_size))
        transforms_list.append(Normalize(mean=self.pixel_mean, std=self.pixel_std))
        self.transforms = Compose(transforms_list)

    def __len__(self) -> int:
        """Returns the video length in frames

        :return: Dataset length, i.e. the frame count as determined through cv2 (OpenCV)
        """

        return self.length

    def __getitem__(self, _) -> dict:
        """Gets the next video frame, throws an error if the frame could not be read

        :param _: Dummy parameter as a stand-in for the usual "item" - this dataloader can only ever return the
            next frame of the video no matter which item number is called
        :return: Dictionary containing the next video frame under the key "img"
        """
        success, img = self.video.read()
        if not success:
            raise ValueError("No image could be read from the video file.")
        aug_data = self.transforms(image=img)
        x = aug_data["image"]
        if img.ndim == 3:
            x = np.transpose(x, axes=[2, 0, 1])
        elif img.ndim == 2:
            x = np.expand_dims(x, axis=0)
        return {'img': torch.from_numpy(x)}


class ImgDataset:
    def __init__(
        self,
        path: str,
        is_fovea: bool,
        pixel_mean: tuple = (.5, .5, .5),
        pixel_std: tuple = (.5, .5, .5),
        img_size: int = 1024,
        batch_size: int = 1,
        random_aug: bool = True,
        use_resize: bool = False,
    ):
        """Dataset that loads images from a folder path

        :param path: Full path to a folder containing images. png, jpg, jpeg, tif, tiff, bmp and gif will be
            recognised.
        :param is_fovea: Boolean determining whether the images are from the FOVEA dataset. If so, images and masks
            will be distinguished by the established FOVEA naming protocol.
        :param pixel_mean: Mean pixel values for image normalisation. Default 0.5 on all channels
        :param pixel_std: Pixel standard deviation values for image normalisation. Default 0.5 on all channels
        :param img_size: Required image size. If zero, it means no cropping. Default 1024 (PromptSAM model)
        :param batch_size: Needed for the case of the requested batch size being larger than the number of images
            in the dataset and oversampling is desired. E.g. single image in training set, but for efficiency a
            batch size of 5 is best: passing this batch size to the dataset will make it appear of the same size as
            the batch size by secretly oversampling. If oversampling shouldn't happen, leave at 1. Defaults to 1
        :param random_aug: Boolean determining whether random augmentations are applied: the albumentations functions
            ColorJitter, VerticalFlip, and HorizontalFlip, all with default parameters. Turn off for validation
            or testing. Defaults to True
        :param use_resize: Boolean determining whether the input should be resized or cropped to achieve the
            required image size. Defaults to False, i.e. cropping is default behaviour
        """

        self.path = pathlib.Path(path)
        self.is_fovea = is_fovea
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.img_size = img_size
        self.batch_size = batch_size
        self.random_aug = random_aug
        self.use_resize = use_resize
        data = {}
        i = 0
        for file in self.path.iterdir():
            if file.suffix.lower()[1:] in ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'gif']:
                if is_fovea:
                    # Assumes images are from FOVEA, with corresponding naming scheme, and all files present
                    idx = int(file.name[5:8])
                    if idx not in data:
                        data[idx] = {}
                    if file.stem[-3:] == 'img':
                        data[idx]['img'] = file
                    else:  # assume it must be a segmentation mask
                        if 'seg' in data[idx]:
                            # if a second non-image file is found for same idx, assume 2nd mask
                            data[idx]['seg'] = [data[idx]['seg'], file]
                        else:
                            data[idx]['seg'] = file
                else:
                    data[i] = {'img': file}
                i += 1
        if is_fovea:  # need to rearrange data by new key
            self.data = {}
            for i, (k, v) in enumerate(data.items()):
                self.data[i] = {'idx': k, 'img': v['img'], 'seg': v['seg']}
            if isinstance(self.data[0]['seg'], list):
                print("More than one mask per record detected")
        else:
            self.data = data
        self.length = len(self.data)
        transforms_list = []
        if self.random_aug:
            transforms_list.extend([
                ColorJitter(),
                VerticalFlip(),
                HorizontalFlip(),
            ])
            if self.img_size > 0:
                if self.use_resize:
                    transforms_list.append(Resize(self.img_size, self.img_size))
                else:
                    transforms_list.append(RandomCrop(self.img_size, self.img_size))
        elif self.img_size > 0:
            if self.use_resize:
                transforms_list.append(Resize(self.img_size, self.img_size))
            else:
                transforms_list.append(CenterCrop(self.img_size, self.img_size))
        transforms_list.append(Normalize(mean=self.pixel_mean, std=self.pixel_std))
        self.transforms = Compose(transforms_list)

    def __len__(self) -> int:
        """Returns the size of the dataset. If batch_size was set to more than 1 during initialisation and the
        actual number of images in the dataset is smaller than the batch size, this will pretend the dataset is as
        large as the batch size to ensure the data loader loads a full batch

        :return: Dataset size
        """

        return max(self.length, self.batch_size)

    def __getitem__(self, index) -> dict:
        """Loads the requested dataset record by item idx as a dict. CAREFUL: if more than one mask was detected in
        the folder containing the data (i.e. FOVEA data with annotator 1 and 2 present), then a random one will be
        returned by __getitem__.

        :param index: Requested record number
        :return: Dictionary containing the image under the key "img", and if the data loaded is from the FOVEA
            dataset, then also the FOVEA idx under "idx" and the annotation mask under "seg" (scaled for max val = 1)
        """

        index %= self.length
        img = Image.open(self.data[index]['img']).convert("RGB")
        img = np.asarray(img)
        if self.is_fovea:
            if isinstance(self.data[index]['seg'], list):  # more than one mask available, select random one
                mask_path = random.choice(self.data[index]['seg'])
            else:
                mask_path = self.data[index]['seg']
            mask = Image.open(mask_path).convert("L")
            mask = np.asarray(mask) // 255
            aug_data = self.transforms(image=img, mask=mask)
        else:
            aug_data = self.transforms(image=img)
        x = aug_data["image"]
        if img.ndim == 3:
            x = np.transpose(x, axes=[2, 0, 1])
        elif img.ndim == 2:
            x = np.expand_dims(x, axis=0)
        record = {'img': torch.from_numpy(x)}
        if self.is_fovea:
            record.update({
                'seg': torch.from_numpy(aug_data["mask"]),
                'idx': self.data[index]['idx']
            })
        return record
