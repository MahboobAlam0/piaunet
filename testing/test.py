import os
import csv
import torch
from tqdm import tqdm

from metrics.metricsEvaluations import compute_segmentation_metrics
from visualization.visualization import save_visual_results, save_individual_result



# Calculate per-image IoU

def calculate_iou(pred, target):
    """Calculate IoU between prediction and target"""
    intersection = (pred & target).sum().item()
    union = (pred | target).sum().item()
    if union == 0:
        return 0.0
    return intersection / union



# TTA 

def tta_inference(model, img):
    preds = []

    # original
    preds.append(model(img)[0])

    # horizontal flip
    flipped = torch.flip(img, dims=[3])
    pred = model(flipped)[0]
    pred = torch.flip(pred, dims=[3])
    preds.append(pred)

    return torch.mean(torch.stack(preds), dim=0)



# TEST FUNCTION

@torch.no_grad()
def test_model(
    model,
    test_loader,
    device,
    save_dir="results_test",
    save_mode="both",
    num_classes=2,
    top_k=10,
):
    os.makedirs(save_dir, exist_ok=True)
    top_results_dir = os.path.join(save_dir, "top_results")
    os.makedirs(top_results_dir, exist_ok=True)

    model.eval()

    print("Running evaluation...")

    # First pass: Calculate overall metrics
    metrics = compute_segmentation_metrics(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=num_classes
    )

    print("\n" + "="*50)
    print("           🎯 TEST RESULTS 🎯")
    print("="*50)
    print(f"Pixel Accuracy: {metrics.get('Pixel_Accuracy', 0):.4f}")
    print(f"mIoU:           {metrics.get('mIoU', 0):.4f}")
    print(f"Dice:           {metrics.get('Dice', 0):.4f}")
    print(f"Precision:      {metrics.get('Precision', 0):.4f}")
    print(f"Recall:         {metrics.get('Recall', 0):.4f}")
    print("="*50 + "\n")

    # Save metrics to CSV
    csv_path = os.path.join(save_dir, "test_metrics.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.4f}"])
    print(f"Metrics saved to {csv_path}\n")

    # visualization (grid) - [Original Image, GT, Pred]
    print("Generating visualizations...")
    batch = next(iter(test_loader))
    imgs = batch["image"].to(device)
    masks = batch["mask"].to(device)

    preds = tta_inference(model, imgs)
    preds = torch.argmax(preds, dim=1)

    save_visual_results(imgs, masks, preds, save_dir, "test_grid")
    print(f"Test grid [Original Image | GT | Pred] saved to {save_dir}/test_grid.png")

    # Second pass: Collect individual results and identify top K
    print(f"\nCollecting results for top {top_k} best predictions...")
    all_results = []
    idx = 0
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)

        preds = tta_inference(model, imgs)
        preds = torch.argmax(preds, dim=1)

        for i in range(imgs.size(0)):
            # Calculate IoU for this image
            iou = calculate_iou(preds[i], masks[i])
            all_results.append({
                'idx': idx,
                'iou': iou,
                'img': imgs[i].cpu(),
                'mask': masks[i].cpu(),
                'pred': preds[i].cpu()
            })
            idx += 1

    # Sort by IoU and get top K
    all_results.sort(key=lambda x: x['iou'], reverse=True)
    top_results = all_results[:top_k]

    # Save top results
    print(f"\nSaving top {top_k} results to {top_results_dir}:")
    for rank, result in enumerate(top_results, 1):
        save_path = os.path.join(top_results_dir, f"top_{rank:02d}_idx_{result['idx']}_iou_{result['iou']:.4f}.png")
        save_individual_result(
            result['img'],
            result['mask'],
            result['pred'],
            save_path,
        )
        print(f"   {rank}. Image #{result['idx']} - IoU: {result['iou']:.4f}")

    # Save all individual results
    if save_mode in ["all_individual", "both"]:
        print(f"\nSaving all {idx} individual predictions to {save_dir}...")
        for i, result in enumerate(tqdm(all_results, desc="Saving")):
            save_path = os.path.join(save_dir, f"img_{result['idx']}.png")
            save_individual_result(
                result['img'],
                result['mask'],
                result['pred'],
                save_path,
            )
        print(f"Saved {idx} individual predictions\n")

    print("="*50)
    print("TEST COMPLETE!")
    print(f"Overall metrics: {save_dir}/test_metrics.csv")
    print(f"Top {top_k} results: {top_results_dir}/")
    print("="*50)

    return metrics