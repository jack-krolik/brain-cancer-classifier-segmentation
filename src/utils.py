import matplotlib.pyplot as plt
import numpy as np
import torch


def show_images_with_masks(images: list, masks: list, nmax: int = 4) -> None:
    """
    Display a set of images and their corresponding masks and their overlay side by side.

    Args:
    - images: list of image tensors to display
    - masks: list of mask tensors to display
    - nmax: maximum number of images to display

    Returns:
    - None
    """
    n_rows = (min(len(images), nmax))  # Adjust for maximum display limit and ensure rounding up
    fig, axs = plt.subplots(n_rows, 3, figsize=(10, n_rows * 3))  # Adjust figsize as needed
    axs = axs.flatten()

    for idx, (image, mask) in enumerate(zip(images, masks)):
        if idx >= nmax:  # Stop if we reach the maximum number of images to display
            break
        image_ax = axs[idx*3] # first index of each triplet for images
        mask_ax = axs[idx*3 + 1] # second index of each triplet for masks
        overlay_ax = axs[idx*3 + 2]

        # Assuming images and masks are Tensors of shape [C, H, W]
        image_ax.imshow(image.permute(1, 2, 0).numpy())
        image_ax.set_title("Image")
        image_ax.axis('off')

        # Assuming masks are single-channel Tensors of shape [1, H, W], convert them for display
        mask_ax.imshow(mask.squeeze().numpy(), cmap='gray')
        mask_ax.set_title("Mask")
        mask_ax.axis('off')

        # Overlay the mask on the image
        overlay_ax.imshow(image.permute(1, 2, 0).numpy())
        overlay_ax.imshow(mask.squeeze().numpy(), cmap='jet', alpha=0.5)
        overlay_ax.set_title("Overlay")
        overlay_ax.axis('off')

    # Hide any remaining unused subplots
    for ax in axs[len(images):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()