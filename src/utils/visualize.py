import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


def plot_images_and_masks(images: list, masks: list, include_overlay: bool = True) -> None:
    """
    Display a set of images and their corresponding masks side by side.

    Args:
    - images: list of image tensors to display
    - masks: list of mask tensors to display
    - nmax: maximum number of images to display

    Returns:
    - None
    """
    assert len(images) == len(masks), "Number of images and masks should be the same"

    n_rows = 2 + 1 if include_overlay else 2
    n_samples = len(images)

    fig, axs = plt.subplots(n_rows, n_samples, figsize=(n_samples * 5, 5 * n_rows))

    for idx, (image, mask) in enumerate(zip(images, masks)):
        image_ax = axs[0, idx]
        mask_ax = axs[1, idx]

        # Assuming images and masks are Tensors of shape [C, H, W]
        image_ax.imshow(image.permute(1, 2, 0).numpy(), cmap='gray')
        image_ax.set_title("Image", fontsize=12, fontweight='bold')
        image_ax.axis('off')

        # Assuming masks are single-channel Tensors of shape [1, H, W], convert them for display
        mask_ax.imshow(mask.squeeze().numpy(), cmap='gray')
        mask_ax.set_title("Mask", fontsize=12, fontweight='bold')
        mask_ax.axis('off')

        if include_overlay:
            overlay_ax = axs[2, idx]
            overlay_ax.imshow(image.permute(1, 2, 0).numpy())
            overlay_ax.imshow(mask.squeeze().numpy(), cmap='jet', alpha=0.5)
            overlay_ax.set_title("Overlay", fontsize=12, fontweight='bold')
            overlay_ax.axis('off')

    fig.suptitle("Images and Masks", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()    


def plot_semantic_predictions(images: list, masks: list, predictions: list, include_overlay: bool = True) -> None:
    """
    Display a set of images, masks, and their corresponding predictions side by side. Optionally, overlay the mask and prediction on the image.

    Args:
    - images: list of image tensors to display
    - masks: list of mask tensors to display
    - nmax: maximum number of images to display

    Returns:
    - None
    """
    assert len(images) == len(masks), "Number of images and masks should be the same"

    n_rows = 5 if include_overlay else 3
    n_samples = len(images)

    fig, axs = plt.subplots(n_rows, n_samples, figsize=(n_samples * 5, 5 * n_rows))

    for idx, (image, mask, pred) in enumerate(zip(images, masks, predictions)):
        image_ax = axs[0, idx]
        mask_ax = axs[1, idx]
        pred_ax = axs[2, idx]

        # Assuming images and masks are Tensors of shape [C, H, W]
        image_ax.imshow(image.permute(1, 2, 0).numpy(), cmap='gray')
        image_ax.set_title("Image", fontsize=12, fontweight='bold')
        image_ax.axis('off')

        mask = mask.squeeze().numpy()
        pred = pred.squeeze().numpy()

        mask_ax.imshow(mask, cmap='gray')
        mask_ax.set_title("Mask", fontsize=12, fontweight='bold')
        mask_ax.axis('off')

        pred_ax.imshow(pred, cmap='gray')
        pred_ax.set_title("Prediction", fontsize=12, fontweight='bold')
        pred_ax.axis('off')

        if include_overlay:
            mask_pred_ax = axs[3, idx]
            overlay_ax = axs[4, idx]
            overlay_ax.imshow(image.permute(1, 2, 0).numpy())
            overlay_ax.imshow(mask, cmap='jet', alpha=0.5)
            overlay_ax.imshow(pred, cmap='jet', alpha=0.5)
            overlay_ax.set_title("Overlay", fontsize=12, fontweight='bold')
            overlay_ax.axis('off')

            mask_pred_ax.imshow(mask, cmap='Blues', alpha=0.6)
            mask_pred_ax.imshow(pred, cmap='Reds', alpha=0.4)

            legend_patches = [mpatches.Patch(color='blue', label='Mask'),
                            mpatches.Patch(color='red', label='Prediction'),
                            mpatches.Patch(color='purple', label='Overlay')
                            ]
            
            mask_pred_ax.legend(handles=legend_patches, loc='upper right')
            mask_pred_ax.set_title("Mask and Prediction", fontsize=12, fontweight='bold')
            mask_pred_ax.axis('off')

    fig.suptitle("Images and Masks", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()  