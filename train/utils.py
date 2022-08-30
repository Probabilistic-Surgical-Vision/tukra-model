from typing import Callable, List, Optional, OrderedDict, Union

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from torchvision.utils import make_grid

# Type hint definitions
Device = Union[torch.device, str]
ImagePyramid = List[Tensor]
LRAdjuster = Callable[[Optimizer, int, float], None]
ScaleAdjuster = Callable[[int, bool], float]
Loss = List[float]

def l1_loss(x: Tensor, y: Tensor) -> Tensor:
    """Calculate the L1 loss between two images."""
    return (x - y).abs().mean()


def scale_pyramid(x: Tensor, scales: int) -> ImagePyramid:
    """Create an pyramid of differently sized images using interpolation.

    Args:
        x (Tensor): The image to create a pyramid from.
        scales (int): The number of scales in the pyramid.

    Returns:
        ImagePyramid: The image pyramid.
    """
    _, _, height, width = x.size()

    pyramid = []

    for i in range(scales):
        ratio = 2 ** i

        size = (height // ratio, width // ratio)
        x_resized = F.interpolate(x, size=size, mode='bilinear',
                                  align_corners=True)

        pyramid.append(x_resized)

    return pyramid


def detach_pyramid(pyramid: ImagePyramid) -> ImagePyramid:
    """Detach an image pyramid from the computational graph.

    Args:
        pyramid (ImagePyramid): The image pyramid to detach.

    Returns:
        ImagePyramid: A copy of the image pyramid detached from the graph.
    """
    return [layer.detach().clone() for layer in pyramid]


def reconstruct(disparity: Tensor, opposite_image: Tensor) -> Tensor:
    """Reconstruct an image given its opposite view, and the disparity
    between the two views.

    Args:
        disparity (Tensor): The single-channel disparity tensor.
        opposite_image (Tensor): The three-channel image from the opposite
            view.

    Returns:
        Tensor: The reconstructed image.
    """
    batch_size, _, height, width = opposite_image.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width) \
        .repeat(batch_size, height, 1) \
        .type_as(opposite_image)

    y_base = torch.linspace(0, 1, height) \
        .repeat(batch_size, width, 1) \
        .transpose(1, 2) \
        .type_as(opposite_image)

    # Apply shift in X direction
    x_shifts = disparity.squeeze(dim=1)

    # In grid_sample coordinates are assumed to be between -1 and 1
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    flow_field = (2 * flow_field) - 1

    return F.grid_sample(opposite_image, flow_field, mode='bilinear',
                         padding_mode='zeros')


def reconstruct_left_image(left_disparity: Tensor,
                           right_image: Tensor) -> Tensor:
    """Reconstruct the left image from the left disparity and right image."""
    return reconstruct(-left_disparity, right_image)


def reconstruct_right_image(right_disparity: Tensor,
                            left_image: Tensor) -> Tensor:
    """Reconstruct the right image from the right disparity and left image."""
    return reconstruct(right_disparity, left_image)


def reconstruct_pyramid(disparities: ImagePyramid,
                        pyramid: ImagePyramid) -> ImagePyramid:
    """Apply `reconstruct()` to each scale of the disparity pyramid.

    Args:
        disparities (ImagePyramid): The disparity at each scale.
        pyramid (ImagePyramid): The original image pyramid,

    Returns:
        ImagePyramid: The reconstructed version of the image pyramid.
    """
    recon_pyramid = []

    for disparity, images in zip(disparities, pyramid):
        left_disp, right_disp = torch.split(disparity, [1, 1], 1)
        left_image, right_image = torch.split(images, [3, 3], 1)

        left_recon = reconstruct_left_image(left_disp, right_image)
        right_recon = reconstruct_right_image(right_disp, left_image)

        recon_image = torch.cat([left_recon, right_recon], dim=1)
        recon_pyramid.append(recon_image)

    return recon_pyramid


def concatenate_pyramids(a: ImagePyramid, b: ImagePyramid) -> ImagePyramid:
    """Concatenate two image pyramids along their channels."""
    return [torch.cat((x, y), 0) for x, y in zip(a, b)]


def adjust_disparity(epoch: int, finetune: bool = False, m: float = 0.02,
                     c: float = 0.0, step: float = 0.2, offset: float = 0.1,
                     min_scale: float = 0.3, max_scale: float = 1.0) -> float:
    """Calculate the disparity scaling of the model given the epoch number.

    The scale is calculated based on a linear equation y = mx + c, where:
    - x is the epoch number.
    - y is the disparity scale.

    This is then quantised to specific step and offset. By default, the step
    and offset are 0.2 and 0.1 respectively, meaning, when the linear
    equation will round to 0.3, 0.5, 0.9, etc.

    Args:
        epoch (int): The epoch number
        m (float, optional): The gradient of the scaling. Defaults to 0.02.
        c (float, optional): The intercept of the scaling. Defaults to 0.0.
        step (float, optional): The step between scales. Defaults to 0.2.
        offset (float, optional): The step offset from 0. Defaults to 0.1.
        min_scale (float, optional): The minimum scale. Defaults to 0.3.
        max_scale (float, optional): The maximum scale. Defaults to 1.0.

    Returns:
        float: The disparity scale.
    """
    if finetune:
        return min_scale

    # Transform epoch to continuous scale using m and c
    scale = ((epoch + 1) * m) + c
    # Quantise to fit the grid defined by step and offset
    scale = (round((scale + offset) / step) * step) - offset
    # Clip to between min and max bounds
    return np.clip(scale, min_scale, max_scale)


def to_heatmap(x: Tensor, device: Device = 'cpu', inverse: bool = False,
               colour_map: str = 'inferno') -> Tensor:
    """Convert a single-channel image to an RGB heatmap.

    Args:
        x (Tensor): The single-channel image to convert.
        device (Device, optional): The torch device to map the output to.
            Defaults to 'cpu'.
        inverse (bool, optional): Reverse the heatmap colours. Defaults to
            False.
        colour_map (str, optional): The matpltlib colour map to convert to.
            Defaults to 'inferno'.

    Returns:
        Tensor: The single-channel image as an RGB image.
    """
    image = x.squeeze(0).cpu().numpy()
    image = 1 - image if inverse else image

    transform = plt.get_cmap(colour_map)
    heatmap = transform(image)[:, :, :3]  # remove alpha channel

    return torch.from_numpy(heatmap).to(device).permute(2, 0, 1)


def combine_disparity(left: Tensor, right: Tensor, device: Device = 'cpu',
                      alpha: float = 20, beta: float = 0.05) -> Tensor:
    """Combine the disparity from both views to remove blind spots.

    This is based off the codebase from Monodepth2, named
    `batch_post_process_disparity()`:
        https://github.com/nianticlabs/monodepth2

    The mask gradient and offset control how quickly the mask for each view
    approaches zero from the left (for left disparity) or the right (for
    right disparity). This should be as smooth as possible without revealing
    the blind spots in the disparity maps.

    Args:
        left (Tensor): The left disparity.
        right (Tensor): The right disparity.
        device (Device, optional): The torch device to map the output to.
            Defaults to 'cpu'.
        alpha (float, optional): The gradient of the mask. Defaults to 20.
        beta (float, optional): The offset of the mask. Defaults to 0.05.

    Returns:
        Tensor: The combined disparity map.
    """
    left_disp = left.cpu().numpy()
    right_disp = right.cpu().numpy()
    mean_disp = (left_disp + right_disp) / 2

    _, height, width = mean_disp.shape

    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, _ = np.meshgrid(x, y)

    left_mask = 1 - np.clip(alpha * (xv - beta), 0, 1)
    right_mask = np.fliplr(left_mask)

    mean_mask = 1 - (left_mask + right_mask)

    combined_disparity = (right_mask * left_disp) \
        + (left_mask * right_disp) \
        + (mean_mask * mean_disp)

    return torch.from_numpy(combined_disparity).to(device)


def run_discriminator(image_pyramid: ImagePyramid,
                      recon_pyramid: ImagePyramid,
                      discriminator: Module,
                      disc_loss_function: Module,
                      batch_size: int) -> Tensor:
    """Run the discriminator predictions and calculate its loss.

    Args:
        image_pyramid (ImagePyramid): The original image pyramid.
        recon_pyramid (ImagePyramid): The reconstructed image pyramid,
        discriminator (Module): The discriminator model.
        disc_loss_function (Module): The discriminator loss function.
        batch_size (int): The batch size of the iteration.

    Returns:
        Tensor: The discriminator loss.
    """
    recon_pyramid = detach_pyramid(recon_pyramid)
    pyramid = concatenate_pyramids(image_pyramid, recon_pyramid)

    predictions = discriminator(pyramid)

    labels = torch.zeros_like(predictions)
    labels[:batch_size] = 1

    return disc_loss_function(predictions, labels) / 2


def get_comparison(image: Tensor, prediction: Tensor, extra: Optional[Tensor],
                   add_scaled: bool = False, device: Device = 'cpu') -> Tensor:
    """Create a comparison image of the image, disparity and reconstruction.

    Args:
        image (Tensor): The original stereo images.
        prediction (Tensor): The stereo disparity image.
        extra (Optional[Tensor]): An extra image to append to the comparison,
            such as the reconstruction. Must be either 2 or 6 channels (i.e.
            stereo single-channel or stereo three-channel).
        add_scaled (bool, optional): Include a scaled version of the
            disparity. Defaults to False.
        device (Device, optional): The torch device to map the output to.
            Defaults to 'cpu'.

    Returns:
        Tensor: The comparison image.
    """
    left_image, right_image = torch.split(image, [3, 3], dim=0)
    left_pred, right_pred = torch.split(prediction, [1, 1], dim=0)

    min_pred, max_pred = prediction.min(), prediction.max()
    scaled_left_pred = (left_pred - min_pred) / (max_pred - min_pred)
    scaled_right_pred = (right_pred - min_pred) / (max_pred - min_pred)

    left_pred = to_heatmap(left_pred, device)
    right_pred = to_heatmap(right_pred, device)

    if extra is not None:
        extra_split = [3, 3] if extra.size(0) == 6 else [1, 1]
        left_extra, right_extra = torch.split(extra, extra_split, dim=0)

        if extra.size(0) == 2:
            left_extra = to_heatmap(left_extra, device)
            right_extra = to_heatmap(right_extra, device)

    images = torch.stack((left_image, right_image, left_pred, right_pred))

    if add_scaled:
        scaled_left_pred = to_heatmap(scaled_left_pred, device)
        scaled_right_pred = to_heatmap(scaled_right_pred, device)

        images = torch.cat((images, scaled_left_pred.unsqueeze(0),
                           scaled_right_pred.unsqueeze(0)))

    if extra is not None:
        images = torch.cat((images, left_extra.unsqueeze(0),
                           right_extra.unsqueeze(0)))

    return make_grid(images, nrow=2)


def prepare_state_dict(state_dict: OrderedDict) -> dict:
    """Cleanup state dict because of `modules` key in DDP."""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def adjust_learning_rate(optimiser: Optimizer, epoch: int, lr: float) -> None:
    """Halve and quarter the learning rate at 30 and 40 epochs respectively.

    Code adapted from:
        https://tinyurl.com/23jb9tnz

    Args:
        optimiser (Optimizer): The optimiser to alter the learning rate of.
        epoch (int): The current epoch number.,
        lr (float): The initial learning rate.
    """
    if epoch > 40:
        target_learning_rate = lr / 4
    elif epoch > 30:
        target_learning_rate = lr / 2
    else:
        target_learning_rate = lr

    for group in optimiser.param_groups:
        group['lr'] = target_learning_rate