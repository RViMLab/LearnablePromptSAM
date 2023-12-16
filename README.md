# LearnablePromptSAM - adapted for FOVEA

This repository is a fork from the [LearnablePromptSAM GitHub repository](https://github.com/Qsingle/LearnablePromptSAM) which accompanies the technical report ["Learnable Ophthalmology SAM"](https://arxiv.org/abs/2304.13425) on arXiv. No original files have been changed, but four additional files partially adapted from existing code have been added. For FOVEA dataset utilities, see the [FOVEA repository](https://github.com/rvimlab/FOVEA).


## Training

Training is started using the ``train.py`` file, with a number of options that can be seen by running ``python train.py --help``. The options relevant for the FOVEA dataset publication are as follows:

```
python train.py
--name <name of the output file the weights should be saved in>
--path <path/to/training/folder>
--epoch <number of epochs>
--batch_size <batch size>
--model_name <model name>
--checkpoint <path/to/SAM/pretrained/checkpoint>
--save_path <path/to/save/model/weights/in>
--lr <learning rate>
--use_lovasz <whether the Lovász-Softmax loss is used instead of crossentropy>
--resize_imgs <whether input images are resized instead of cropped>
```

The folder given at ``--path`` is assumed to contain FOVEA data with the corresponding naming scheme: the dataset record id is taken from the file name starting with ``FOVEA###``, images end in ``_img.png``, all other files in the folder with matching id numbers are assumed to be the corresponding binary ground truth masks. Careful: if the masks of both annotators are present, both will be loaded by the dataset and a random one will be returned during training.

If the batch size is chosen larger than the number of images available in the dataset, it will be oversampled to provide a full batch. This is done to preserve efficiency and better convergence e.g. in the case of training on a single image.

The models used require inputs to be of a certain size - in the case of ViT-b, 1024 by 1024 pixels. By default or when setting ``--no-resize-imgs``, images will be randomly cropped to the required dimensions. Alternatively, by setting ``--resize-imgs``, images will be resized to the required dimensions without keeping the aspect ratio.

The parameters used to train on the intraoperative FOVEA images with the recommended data split were as follows, with the ``sam_vit_b_01ec64.pth`` checkpoint already downloaded from the [original SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints):

```
python train.py
--name "i-lovasz-crop"
--path <path/to/FOVEA/training/images>
--epoch 2000
--batch_size 5
--model_name vit_b
--checkpoint <path/to/sam_vit_b_01ec64.pth>
--save_path <path/to/chkpts>
--lr 0.1
--use_lovasz
--no-resize_imgs
```


## Testing
Testing is started using the ``test.py`` file, with a number of options that can be seen by running ``python test.py --help``. The options relevant for the FOVEA dataset publication are as follows:

```
python test.py
--path <path/to/test/folder/or/video>
--model_name <model name>
--checkpoint <path/to/saved/model/weights>
--save_path <path/to/prediction/segmentation/output>
--resize_imgs <whether input images are resized instead of cropped>
--is_fovea <whether the folder contains FOVEA images with corresponding masks>
--patched_pred <whether predictions are combined from multiple patches>
--patched_threshold <patched prediction combination threshold>
```

The given input path can either be a folder containing images or a direct path to a video:
* If it is a folder, ``--is_fovea`` can be set to indicate it is a test folder belonging to the FOVEA dataset, and therefore ground truth masks are automatically loaded and the DICE metric is evaluated and printed in console.
* If ``--no-is_fovea`` is set, the dataset will instead look for any images with an extension in the following list: png, jpg, jpeg, tif, tiff, bmp, gif.
* If it is a direct path to a video, it will be recognised as such if its extension is in the following list: mkv, avi, mp4, mov. However, the code could be amended to accept any format that can be read by ``cv2.VideoCapture()``.

By default, images or video frames will either be cropped or resized to the dimensions required by the model, the latter being forced by setting ``--resize_imgs`` which, as before, disregards the aspect ratio. Alternatively, ``--patched_pred`` can be set, which means the image will be cut into five overlapping patches - one at each image corner and one centred on the image centre - fed into the model independently. The five patch predictions are then combined to a single output prediction that matches the original image dimensions.

This combined prediction is based on a "vote" of the patches at each pixel, with a fractional threshold - set via ``--patched_threshold`` - above which a prediction is accepted. By default, the threshold is at 0, which is equivalent to a pixel-wise OR operation or union of all the patch predictions. The other threshold used for the FOVEA dataset tests is 0.5: only if more than half the patches available at a given pixel agree is a prediction accepted.

The parameters used to test on the intraoperative FOVEA images with the recommended data split were as follows:

```
python test.py
--path <path/to/FOVEA/test/images>
--model_name vit_b
--checkpoint <path/to/sam_vit_b_e2000-bs5_i-lovasz-crop.pth>
--save_path <path/to/prediction/segmentation/output>
--no-resize_imgs
--is_fovea
--no-patched_pred
```

The console output for the example above reads as follows:

```
Item 0 / FOVEA007 processed, DICE 0.6430, TP 0.0217, FP 0.0152, FN 0.0088, TN 0.9543
Item 1 / FOVEA012 processed, DICE 0.5776, TP 0.0128, FP 0.0141, FN 0.0047, TN 0.9685
Item 2 / FOVEA026 processed, DICE 0.4985, TP 0.0101, FP 0.0152, FN 0.0052, TN 0.9695
Item 3 / FOVEA030 processed, DICE 0.4996, TP 0.0084, FP 0.0124, FN 0.0044, TN 0.9747
Item 4 / FOVEA032 processed, DICE 0.4612, TP 0.0062, FP 0.0101, FN 0.0043, TN 0.9794
Item 5 / FOVEA033 processed, DICE 0.4871, TP 0.0063, FP 0.0074, FN 0.0058, TN 0.9805
Item 6 / FOVEA037 processed, DICE 0.4926, TP 0.0079, FP 0.0116, FN 0.0046, TN 0.9758
Item 7 / FOVEA038 processed, DICE 0.5264, TP 0.0132, FP 0.0173, FN 0.0065, TN 0.9629
Item 8 / FOVEA039 processed, DICE 0.5722, TP 0.0167, FP 0.0175, FN 0.0075, TN 0.9584
Item 9 / FOVEA040 processed, DICE 0.5627, TP 0.0144, FP 0.0142, FN 0.0083, TN 0.9631
Mean DICE: 0.5321; TP 0.0118, FP 0.0135, FN 0.0060, TN 0.9687
```

The last line reports the mean of all metrics across all images tested. TP / FP / FN / TN are the confusion matrix entries and stand for true positive, false positive, false negative, and true negative as a percentage of all image pixels, where positive is e.g. a vessel prediction and negative is the background.


## Reuse

If you use or adapt code from the new files - [``datasets.py``](https://github.com/RViMLab/LearnablePromptSAM/blob/main/datasets.py), [``train.py``](https://github.com/RViMLab/LearnablePromptSAM/blob/main/train.py), [``test.py``](https://github.com/RViMLab/LearnablePromptSAM/blob/main/test.py) -, please cite the FOVEA dataset publication it was written for:

> Ravasio, C and Flores-Sanchez, B and Bloch, E and Bergeles, C and da Cruz, L.
FOVEA: Preoperative and Intraoperative Retinal Fundus Images with Optic Disc and Retinal Vessel Annotations

Or, in the case of the [``lovasz.py``](https://github.com/RViMLab/LearnablePromptSAM/blob/main/lovasz.py) file adapted from the [LovaszSoftmax repository](https://github.com/bermanmaxim/LovaszSoftmax), please cite their original paper:

> Berman, M and Triki, AR and Blaschko, MB (2018).
The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks.
Proceedings of the IEEE conference on CVPR, 4413-4421.
