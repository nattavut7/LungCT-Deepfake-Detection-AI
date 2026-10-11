import numpy as np
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def minmax01(img):
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)

def apply_window_hu(img_hu, center=-600.0, width=1500.0):
    lo = center - width/2.0
    hi = center + width/2.0
    img = np.clip(img_hu, lo, hi)
    return (img - lo) / (hi - lo)

def to_rgb_uint8(img01):
    img8 = np.clip(img01*255.0, 0, 255).astype(np.uint8)
    return np.stack([img8, img8, img8], axis=-1)

def make_transform(image_size=224, train=False):
    ops = [transforms.ToPILImage(), transforms.Resize((image_size, image_size))]
    if train:
        ops += [
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)
