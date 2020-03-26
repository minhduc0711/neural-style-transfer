import torch
import torch.nn.functional as F

from torchvision import transforms

train_image_transform = transform = transforms.Compose([
    transforms.Lambda(lambda img: force_rgb(img)),
    transforms.Resize(256),  # only resize the shorter edge to keep aspect ratio
    transforms.CenterCrop(256),  # then crop to get a square image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Test image can be of arbitrary size
test_image_transform = transforms.Compose([ 
    transforms.Lambda(lambda img: force_rgb(img)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def gram_matrix(x):
    batch_size, C, H, W = x.shape
    x = x.reshape(batch_size, C, -1)
    x_t = x.permute([0, 2, 1])
    return torch.bmm(x, x_t) / (C * H * W)


def normalize_images(x, mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).to(x.device)
    std = torch.tensor(std).to(x.device)

    return (x - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)


def force_rgb(img):
    if img.mode != "RGB":
        return img.convert("RGB")
    else:
        return img
