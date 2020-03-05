import torch
import torch.nn.functional as F


def gram_matrix(x):
    batch_size, C, H, W = x.shape
    x = x.reshape(batch_size, C, -1)
    x_t = x.permute([0, 2, 1])
    return torch.bmm(x, x_t) / (C * H * W)


def normalize_images(x, mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    
    return (x - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)