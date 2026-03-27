import os
import numpy as np
import matplotlib.pyplot as plt



# Colorize mask 
def colorize_mask(mask):
    colors = np.array([
        [0, 0, 0],        # background
        [255, 255, 255],  # foreground
    ])
    return colors[mask]



# Denormalize tensor image

def denormalize_image(img):
    """Denormalize from [-1, 1] to [0, 1]"""
    img = img.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5)  # Reverse normalization with mean=0.5, std=0.5
    return np.clip(img, 0, 1)



# Convert tensor image safely

def tensor_to_image(img):
    img = img.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)



# Save grid visualization (Original, GT, Pred)

def save_visual_results(images, gts, preds, save_dir, name="grid"):
    os.makedirs(save_dir, exist_ok=True)

    images = images.cpu()
    gts = gts.cpu()
    preds = preds.cpu()

    B = images.size(0)

    fig, axes = plt.subplots(B, 3, figsize=(9, 3 * B))

    if B == 1:
        axes = [axes]

    for i in range(B):
        img_original = denormalize_image(images[i])
        gt = colorize_mask(gts[i].numpy())
        pred = colorize_mask(preds[i].numpy())

        axes[i][0].imshow(img_original)
        axes[i][0].set_title("Original Image")
        axes[i][0].axis("off")

        axes[i][1].imshow(gt)
        axes[i][1].set_title("GT")
        axes[i][1].axis("off")

        axes[i][2].imshow(pred)
        axes[i][2].set_title("Pred")
        axes[i][2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"))
    plt.close()



# Save enhanced image separately

def save_enhanced_image(images, enhanced, save_dir, name="enhanced"):
    os.makedirs(save_dir, exist_ok=True)

    images = images.cpu()
    enhanced = enhanced.cpu()

    B = images.size(0)

    fig, axes = plt.subplots(B, 2, figsize=(6, 3 * B))

    if B == 1:
        axes = [axes]

    for i in range(B):
        img_original = denormalize_image(images[i])
        img_enhanced = tensor_to_image(enhanced[i])

        axes[i][0].imshow(img_original)
        axes[i][0].set_title("Original Image")
        axes[i][0].axis("off")

        axes[i][1].imshow(img_enhanced)
        axes[i][1].set_title("Enhanced (Physics)")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"))
    plt.close()


# Save physics outputs

def save_physics_maps(images, t, b, j, save_dir, name="physics"):
    os.makedirs(save_dir, exist_ok=True)

    images = images.cpu()
    t = t.cpu()
    b = b.cpu()
    j = j.cpu()

    B = images.size(0)

    fig, axes = plt.subplots(B, 4, figsize=(12, 3 * B))

    if B == 1:
        axes = [axes]

    for i in range(B):
        axes[i][0].imshow(tensor_to_image(images[i]))
        axes[i][0].set_title("Input")

        axes[i][1].imshow(t[i, 0], cmap="gray")
        axes[i][1].set_title("Turbidity")

        axes[i][2].imshow(tensor_to_image(b[i]))
        axes[i][2].set_title("Backscatter")

        axes[i][3].imshow(tensor_to_image(j[i]))
        axes[i][3].set_title("Restored J")

        for j_ in range(4):
            axes[i][j_].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"))
    plt.close()


# Save individual comparison

def save_individual_result(image, gt, pred, save_path):
    img = denormalize_image(image)  # Use denormalize_image for proper normalization
    gt = colorize_mask(gt.cpu().numpy())
    pred = colorize_mask(pred.cpu().numpy())

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt)
    plt.title("GT")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred)
    plt.title("Pred")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()