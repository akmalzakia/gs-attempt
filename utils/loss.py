from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size))
    return window


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: Variable,
    window_size: int,
    channel: int,
    size_average=True,
):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    ) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1: torch.Tensor, img2: torch.Tensor, windows_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(windows_size, channel)

    if img1.is_cuda:
        window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, windows_size, channel, size_average)

def get_image_gradients(image):
    if image.dim() == 3:
        image = image.unsqueeze(0) # (1, C, H, W)
        
    C = image.shape[1]
    if C == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=image.device).view(1, 3, 1, 1)
        image = (image * weights).sum(dim=1, keepdim=True)
    
    # Now image has 1 channel
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32, device=image.device)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32, device=image.device)
    
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)
    
    return grad_x, grad_y

def edge_loss(network_output, gt, steepness=15, tau=0.3):
    network_grad_x, network_grad_y = get_image_gradients(network_output)
    gt_grad_x, gt_grad_y = get_image_gradients(gt)

    gt_magnitude = torch.sqrt(gt_grad_x**2 + gt_grad_y**2 + 1e-6)

    edge_mask = torch.sigmoid(steepness * (gt_magnitude - tau))
    
    loss_x = l1_loss(network_grad_x, gt_grad_x, w=edge_mask)
    loss_y = l1_loss(network_grad_y, gt_grad_y, w=edge_mask)
    
    return (loss_x + loss_y) / 2.0
