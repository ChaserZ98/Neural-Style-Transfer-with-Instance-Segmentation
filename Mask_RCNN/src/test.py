import os
import time
import json
import torch
from ochuman_dataset import OCHumanDataset
from utils.eval import *
from utils.misc import collate_fn

if __name__ == "__main__":
    file_ids = []
    for _, _, file_names in os.walk(os.path.join("./dataset", "annotations", "test")):
        for file_name in file_names:
            file_ids.append(file_name.split(".")[0])

    dataset_test = OCHumanDataset(
                        root_dir="./dataset/",
                        img_ids=file_ids,
                        transforms=None,
                        train=False
                    )
    data_loader_test = torch.utils.data.DataLoader(
                        dataset_test, 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=1,
                        collate_fn = collate_fn
                    )

    device = torch.device("cuda")
    cpu_device = torch.device("cpu") 

    eval_dict = {}

    base_weight_dir = "./out/weights/"
    base_metrics_dir = "./out/metrics/"
    if not os.path.exists(base_metrics_dir):
        os.makedirs(base_metrics_dir)

    with torch.no_grad():
        for weight_file in os.listdir(base_weight_dir):
            # Load model from weights
            model = torch.load(os.path.join(base_weight_dir, weight_file))
            model.to(device)
            model.eval()

            pred_dict = {}
            gt_dict = {}
            epoch = weight_file.split(".")[0]
            bboxes = []
            scores = []

            for i, (images, targets) in enumerate(data_loader_test):
                print(f"Processing - Epoch: {epoch}, Index: {i}")
                images = list(img.to(device) for img in images)

                outputs = model(images)
                pred_boxes = outputs[0]["boxes"].to(cpu_device).numpy()
                pred_scores = outputs[0]["scores"].to(cpu_device).numpy()
                image_id = targets[0]["image_id"].item()
                gt_boxes = targets[0]["boxes"].to(cpu_device).numpy()
                pred_dict[image_id] = {"boxes": pred_boxes, "scores": pred_scores}
                gt_dict[image_id] = gt_boxes

            start_time = time.time()
            ax = None
            avg_precs = []
            iou_thrs = []
            for idx, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):
                data = get_avg_precision_at_iou(gt_dict, pred_dict, iou_thr=iou_thr)
                avg_precs.append(data['avg_prec'])
                iou_thrs.append(iou_thr)
                precisions = data['precisions']
                recalls = data['recalls']
                #ax = plot_pr_curve(
                #    precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx*2], ax=ax)
            # prettify for printing:
            avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
            iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
            print('mAP: {:.2f}'.format(100 * np.mean(avg_precs)))
            print('Average Precisions: ', avg_precs)
            print('IoU Thresholds:  ', iou_thrs)
    
            eval_dict[epoch] = dictionary = dict(zip(iou_thrs, avg_precs))
            with open(os.path.join(base_metrics_dir, "metrics.json"), "w") as f:
                json.dump(eval_dict, f, indent=4)


