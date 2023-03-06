import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import imutils
import albumentations as A
import torch
import numpy as np
import segmentation_models_pytorch as smp
import cv2
import matplotlib.pyplot as plt
import json
import random

from segmentation_models_pytorch import utils as smp_utils
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm.notebook import tqdm

# %%
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# %%
torch.cuda.empty_cache()
# %%
DATA_DIR = 'src/data/'
images_dir = os.path.join(DATA_DIR, 'images')
masks_dir = os.path.join(DATA_DIR, 'masks')
# %%
images = sorted(os.listdir(os.path.join(DATA_DIR, 'images')))
len(images)
# %%
masks = sorted(os.listdir(os.path.join(DATA_DIR, 'masks')))
len(masks)
# %%
images[7], masks[7]
# %%
img = cv2.imread(os.path.join(DATA_DIR, 'masks', '20230223_203409.png'))
np.unique(img)


# %%
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title(), fontsize=18)
        plt.imshow(image)
    plt.show()


# %%
checkpoint_dir = 'checkpoints/fpn_mobilenetv2_512im_bce_v2'
IM_SIZE = (512, 512)
BATCH_SIZE = 8
ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['receipt']
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


# %%
class Dataset(BaseDataset):

    def __init__(
            self,
            images_list,
            masks_list,
            images_dir,
            masks_dir,
            im_size=(1024, 1024),
            augmentation=None,
            preprocessing=None,
    ):
        self.im_size = im_size
        self.images_ids = images_list
        self.masks_ids = masks_list
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.images_ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.masks_ids]

        self.class_values = [86]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i], 0)

        # print(self.images_fps[i], self.masks_fps[i])
        if image.shape[0] != mask.shape[0]:
            # image = imutils.rotate(image, angle=90)
            image = cv2.flip(image.transpose(1, 0, 2), 0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.im_size, interpolation=cv2.INTER_AREA)

        mask = cv2.resize(mask, self.im_size, interpolation=cv2.INTER_AREA)
        mask = np.expand_dims(mask, axis=2).astype('float')
        mask[mask > 0] = 1

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_ids)


# %%
X_train, X_valid, y_train, y_valid = train_test_split(images, masks, test_size=0.2, random_state=SEED)
# X_valid, X_test, y_valid, y_test = train_test_split(X_valid, X_valid, test_size=0.5, random_state = SEED)
print(f"Train: {len(X_train), len(y_train)}")
print(f"Valid: {len(X_valid), len(y_valid)}")

print(f"7th element: {X_train[7], y_train[7]}")
# %%
dataset = Dataset(X_train, y_train, images_dir, masks_dir, im_size=IM_SIZE)

image, mask = dataset[25]  # get some sample


# visualize(
#     image=image,
#     mask=mask.squeeze(),
# )


# %%
def get_training_augmentation():
    train_transform = [
        A.Flip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=1, border_mode=0),
        #         A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        #         A.RandomCrop(height=896, width=896, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """
    add paddings to make image shape divisible by 32
    """
    test_transform = [
        A.PadIfNeeded(384, 480)
    ]
    return A.Compose(test_transform)


def to_tensor(x, **kwargs):
    if len(x.shape) == 3:
        return x.transpose(2, 0, 1).astype('float32')
    else:
        return x.astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


# %%
# Dataset(images, images_dir, masks_dir, )
augmented_dataset = Dataset(
    images,
    masks,
    images_dir,
    masks_dir,
    im_size=IM_SIZE,
    augmentation=get_training_augmentation(),
)

# for i in range(3):
#     image, mask = augmented_dataset[1]
#     visualize(image=image, mask=mask)
# %%
train_dataset = Dataset(
    X_train,
    y_train,
    images_dir,
    masks_dir,
    im_size=IM_SIZE,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    #     classes=CLASSES,
)

valid_dataset = Dataset(
    X_valid,
    y_valid,
    images_dir,
    masks_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    #     classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
# %%
loss = smp_utils.losses.BCELoss()
metrics = [
    smp_utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters()),  # , lr=0.0001
])
# %%
lambda1 = lambda epoch: 0.93 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
# %%
train_epoch = smp_utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp_utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)
# %%
max_score = 0
lrs = []
train_loss = []
train_iou = []
val_loss = []
val_iou = []

for epoch in range(0, 50):

    print(f"\nEpoch: {epoch + 1}")
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    lrs.append(optimizer.param_groups[0]['lr'])
    train_loss.append(train_logs['bce_loss'])
    train_iou.append(train_logs['iou_score'])
    val_loss.append(valid_logs['bce_loss'])
    val_iou.append(valid_logs['iou_score'])
    scheduler.step()

    if max_score < valid_logs['iou_score']:
        print(
            f"=> [On epoch {epoch + 1:03d}] best IOU was improved from {max_score:.3f} to {valid_logs['iou_score']:.3f}")
        max_score = valid_logs['iou_score']
        torch.save(
            {
                'epoch': epoch + 1,
                'arch': 'FPN + mobilenet_v2',
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },
            str(os.path.join(checkpoint_dir,
                             f"epoch{epoch:03d}_IOU({valid_logs['iou_score']:.3f})_bce_loss({valid_logs['bce_loss']:.4f}).pth"))
        )
        print('Model saved!')
    else:
        print(
            f"=> [On epoch {epoch + 1:03d}] best IOU was not improved from {max_score:.3f} ({valid_logs['iou_score']:.3f})")


plt.plot(val_loss, label='val', marker='o')
plt.plot(train_loss, label='train', marker='o')
plt.title('Loss per epoch'); plt.ylabel('loss');
plt.xlabel('epoch')
plt.legend(), plt.grid()
plt.show()
#%%
plt.plot(train_iou, label='train_IoU', marker='*')
plt.plot(val_iou, label='val_IoU',  marker='*')
plt.title('Score per epoch'); plt.ylabel('IoU')
plt.xlabel('epoch')
plt.legend(), plt.grid()
plt.show()