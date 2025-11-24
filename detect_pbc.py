import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.pbc.pbc import correct_boxes_pbc  # your PBC function


def save_predictions_yolo(save_dir, path, boxes, confs, cls):
    """Save predictions in YOLO txt format."""
    txt_path = save_dir / (path.stem + ".txt")
    with open(txt_path, "w") as f:
        for box, conf, c in zip(boxes, confs, cls):
            # Convert x1y1x2y2 -> x_center y_center width height normalized
            x1, y1, x2, y2 = box
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            # Normalize to 0-1 using image size
            img_h, img_w = 1.0, 1.0  # placeholder; we'll pass real size
            # Write normalized coordinates
            f.write(f"{int(c)} {xc / img_w:.6f} {yc / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f} {conf:.6f}\n")


def run(
    weights="runs/train/sip_yolov5/weights/best.pt",
    data="data/data.yaml",
    batch_size=16,
    imgsz=640,
    conf_thres=0.25,
    iou_thres=0.45,
    device="cuda",
    save_dir="runs/val_pbc",
    verbose=True,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names, _pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    dataloader, _dataset = create_dataloader(
        path=None, imgsz=imgsz, batch_size=batch_size, stride=stride, pad=0.0, rect=True, mode="val", data=data
    )

    # Metrics
    seen, confusion_matrix = 0, ConfusionMatrix(nc=len(names))
    stats = []

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        targets = targets.to(device)
        img.shape[0]

        # Inference
        with torch.no_grad():
            pred = model(img)

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for si, det in enumerate(pred):
            im0 = shapes[si][0]  # original image
            _img_h, _img_w = im0.shape[:2]

            if len(det):
                # Rescale boxes to original image
                det[:, :4] = scale_boxes(img[si].shape[1:], det[:, :4], im0.shape).round()

                # Convert to numpy for PBC
                boxes_np = det[:, :4].cpu().numpy()
                confs = det[:, 4].cpu().numpy()
                cls = det[:, 5].cpu().numpy()

                # Apply PBC
                boxes_corrected = correct_boxes_pbc(im0, boxes_np)

                # Update det tensor
                det[:, :4] = torch.tensor(boxes_corrected, device=det.device)

                # Save YOLO format predictions
                save_predictions_yolo(save_dir, Path(paths[si]), boxes_corrected, confs, cls)

            # Update metrics
            labels = targets[targets[:, 0] == si, 1:]
            stats.append((det, labels))

            # Update confusion matrix
            if len(det):
                confusion_matrix.process_batch(det[:, :4], labels[:, 1:], det[:, 4])

            seen += 1

    # Compute metrics
    stats_tensor = [s[0] for s in stats if s[0] is not None]
    if len(stats_tensor):
        stats_tensor = torch.cat(stats_tensor, 0)
    else:
        stats_tensor = torch.zeros((0, 6))

    # mAP computation
    if stats_tensor.shape[0]:
        precision, recall, AP, f1, ap_class = ap_per_class(
            stats_tensor[:, :4], stats_tensor[:, 4].long(), stats_tensor[:, 5]
        )
    else:
        precision, recall, AP, f1, _ap_class = 0, 0, 0, 0, 0

    # Print results
    print("\nPBC Evaluation Results:")
    print(f"mAP@0.5: {AP:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Corrected predictions saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/train/sip_yolov5/weights/best.pt")
    parser.add_argument("--data", type=str, default="data/data.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="runs/val_pbc")
    opt = parser.parse_args()

    run(
        weights=opt.weights,
        data=opt.data,
        batch_size=opt.batch_size,
        imgsz=opt.imgsz,
        conf_thres=opt.conf,
        iou_thres=opt.iou,
        device=opt.device,
        save_dir=opt.save_dir,
    )
