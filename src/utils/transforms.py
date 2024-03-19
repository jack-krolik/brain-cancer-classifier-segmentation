import torchvision.transforms.functional as TF
from torchvision import transforms


class DualInputCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class SingleInputCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


# Wrapper to apply the same transform to both the image and the mask
# NOTE: if applying spatial transforms, may need custom implementation to ensure the same transform is applied to both
class DualInputTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, mask):
        return self.transform(img), self.transform(mask)


class SingleInputTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return self.transform(img)


class DualInputResize:
    def __init__(self, size, interpolation=TF.InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, mask):
        # NOTE: mask interpolation is NEAREST to avoid interpolation artifacts shifting the mask values (e.g. 1 -> 0.5)
        return TF.resize(img, self.size, self.interpolation), TF.resize(
            mask, self.size, TF.InterpolationMode.NEAREST
        )


class SingleInputResize:
    def __init__(self, size, interpolation=TF.InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # NOTE: mask interpolation is NEAREST to avoid interpolation artifacts shifting the mask values (e.g. 1 -> 0.5)
        return TF.resize(img, self.size, self.interpolation)
