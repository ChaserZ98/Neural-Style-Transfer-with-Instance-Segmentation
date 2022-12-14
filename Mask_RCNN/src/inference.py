import os
import torch
from ochuman_dataset import OCHumanDataset
from utils.misc import collate_fn
import cv2
import numpy as np

def save_img(index, images, outputs, cpu_device=torch.device("cpu")):
    overall_mask = np.zeros((600, 600, 1))
    for i in range(len(outputs)):
        for j in range(len(outputs[i]["masks"])):
            mask = outputs[i]["masks"][j].to(cpu_device).permute(1, 2, 0).detach().numpy()
            overall_mask[mask >= 0.25] = 1
        print(f"Image ID: {index}, Number of masks: {j + 1}")
        overall_mask[overall_mask > 0] = 1
        overall_mask = np.squeeze(overall_mask)

        img = images[i].to(cpu_device).permute(1, 2, 0).numpy()
        img *= 255

        # add cyan to img and save as new image
        blend = 0.65
        cyan = np.full_like(img,(255,255,0))
        img_cyan = cv2.addWeighted(img, blend, cyan, 1-blend, 0)
        
        idx = (overall_mask == 1)
        img[idx] = img_cyan[idx]

        cv2.imwrite(os.path.join("./out", f"{index}_{i}.jpg"), img[:, :, ::-1])
        
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

    with torch.no_grad():
        model = torch.load("./out/weights/100.pth")
        model.to(device)
        model.eval()

        for i, (images, targets) in enumerate(data_loader_test):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            save_img(i, images, outputs)     
