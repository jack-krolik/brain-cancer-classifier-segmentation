import torchvision.transforms.functional as TF

class DualInputCompose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def add_transform(self, transform):
        self.transforms.append(transform)

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label
    
# Wrapper to apply the same transform to both the image and the mask
# NOTE: if applying spatial transforms, may need custom implementation to ensure the same transform is applied to both
class DualInputTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, label):
        return self.transform(img), self.transform(label)

class DualInputResize:
    def __init__(self, size, interpolation=TF.InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, label):
        # NOTE: mask interpolation is NEAREST to avoid interpolation artifacts shifting the label values (e.g. 1 -> 0.5)
        return TF.resize(img, self.size, self.interpolation), TF.resize(label, self.size, TF.InterpolationMode.NEAREST)

class ImgOnlyTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, label):
        return self.transform(img), label


