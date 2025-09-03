from ultralytics import YOLO
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import math

def reliability_plot(bin_dict, prec, bin_iou, loss, modl):

    bin_ends = sorted(bin_dict.keys())
    # Compute bin starts and widths
    bin_starts = [0.0] + bin_ends[:-1]
    widths = 0.1
    heights = [bin_dict[end] for end in bin_ends]
    print(f"heights: {heights}")
    print(f"precision: ",prec)
    print("IOU: ", bin_iou)
    fig, ax = plt.subplots()
    bars = ax.bar(bin_starts, heights, width=widths, align='edge', edgecolor='black', alpha=0.7, label='Bin-wise Avg Confidence Values')
    difference_heights = [prec[i]*bin_iou[i] - heights[i] for i in range(len(heights))]
    for i in range(len(difference_heights)):
        diff = difference_heights[i]
        print(f"difference for {i}th bin:", diff )
        x_pos = bin_starts[i] + widths / 2
        # If the difference is positive, stack it above the original bin
        if diff > 0:
            ax.bar(bin_starts[i], diff, width=widths, bottom=heights[i], align='edge', edgecolor='black', color = "maroon", alpha=0.7, label='Gap' if i == 0 else "")
            ax.text(x_pos, heights[i]+diff, f"{diff:.2f}", ha='center', va='bottom', fontsize=9)
        # If the difference is negative, stack it below the original bin
        elif diff < 0:
            ax.bar(bin_starts[i], abs(diff), width=widths, bottom=heights[i] + diff, align='edge', edgecolor='black', color = "maroon", alpha=0.7, label='Gap' if i == 0 else "")
            ax.text(x_pos, heights[i] + diff, f"{diff:.2f}", ha='center', va='top', fontsize=9)
    ax.set_xlabel('Confidence Bin')
    # ax.set_ylabel('Precision')
    ax.set_title('Reliability Plot')
    ax.set_xticks([round(k, 1) for k in bin_dict.keys()])
    ax.set_yticks([round(k, 1) for k in bin_dict.keys()])
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Ideal Calibration')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    # Display the plot
    file_path = f"/home/Anjali/Desktop/Anjali_dev/calib_Plots/LaECE Plots/cf_0.25/RP/{loss}_{modl}.png"
    plt.savefig(file_path)
    print(f"RP for {loss} {modl} saved")
    plt.show()
    clean_lst = [x for x in difference_heights if not math.isnan(x)]
    return sum(clean_lst)
    


def confidence_histogram(conf_dict, prec, bin_iou, loss, modl):

    bin_ends = sorted(conf_dict.keys())
    # Compute bin starts and widths
    bin_starts = [0.0] + bin_ends[:-1]
    heights = [len(conf_dict[k]) for k in bin_ends]
    total = sum(heights)
    print(heights)
    width = 0.1
    plt.figure(figsize=(5, 5))
    plt.bar(bin_starts, heights, width=width, align='edge', edgecolor='black', color='steelblue')
    for i, (count, center) in enumerate(zip(heights, bin_ends)):
        if count == 0:
            continue  # Skip empty bins
        percent = (count / total) * 100
        x_pos = center
        y_pos = count + 0.2
        label = f"{percent:.1f}%"
        plt.text(x_pos, y_pos, label, ha='center', va='bottom', fontsize=9)
    tot = 0
    cnt = 0
    for lst in conf_dict.values():
        tot += sum(lst)
        cnt += len(lst)
    avg_conf = tot / cnt 
    avg_precision = sum(prec)/len(prec)
    avg_iou = sum(bin_iou)/len(bin_iou)
    plt.plot([avg_conf, avg_conf], [0, max(heights) + 2000], linestyle = "--", color = "black", label = f"Avg conf = {avg_conf:.2f}")
    plt.plot([avg_precision*avg_iou, avg_precision*avg_iou], [0, max(heights) + 2000], linestyle = "--", color = "red", label = f"Avg prec * Avg IOU = {avg_precision*avg_iou:.2f}")
    # Add axis labels and title
    plt.legend()
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Histogram')
    # Optional axis limits
    plt.xlim(0, 1)
    plt.ylim(0, max(heights) + 2000)
    plt.xticks([round(k, 1) for k in bin_starts])
    # plt.yticks([round(k, 1) for k in bin_starts])
    file_path = f"/home/Anjali/Desktop/Anjali_dev/calib_Plots/LaECE Plots/cf_0.25/CH/{loss}_{modl}.png"
    plt.savefig(file_path)
    print(f"CH for {loss} {modl} saved")
    plt.show()
    return (avg_conf-(avg_precision*avg_iou))
    
    

def round_down_to_interval(x, step):
    return round(math.ceil(x * 10) / 10, 1)

def precision(lst):
    count_0 = lst.count(0)
    count_1 = lst.count(1)
    return count_1/(count_1 + count_0)


with open ("La DECE CF 0.25_noabs.txt", 'a') as file:
    file.write(f"Loss\tModel\tDECE\tPr-Cf")
    losses = ['Org', 'MBLS', 'ACLS', 'MDCA']
    # models = ['yolo11n_scratch', 'yolov8n_scratch', 'yolov10n_scratch']
    # losses = ['Org']
    # models = ['yolo11n_FT']
    models = ['yolo11n_scratch', 'yolov8n_scratch', 'yolov10n_scratch', 'yolo11n_FT', 'yolo11n_LPFT', 'yolov8n_FT', 'yolov8n_LPFT', 'yolov10n_FT', 'yolov10n_LPFT']

    for loss in losses:
        for modl in models:
            if loss == 'Org':
                model_path = f"/home/Anjali/Documents/VisDrone_runs/runs/detect/origi_new_{modl}/weights/best.pt"
            else:
                model_path = f'/home/Anjali/Desktop/Anjali_dev/runs/detect/{loss}/{modl}/weights/best.pt'  # or path to fine-tuned binary-class model

            if os.path.exists(model_path):
                print(f"{model_path} found")
                image_dir = '/home/Anjali/Documents/VisDrone_runs/VisDrone2019-DET-test-dev-copy/images'   # Folder with images
                label_dir = '/home/Anjali/Documents/VisDrone_runs/VisDrone2019-DET-test-dev-copy/labels/'   # YOLO format: one .txt per image

                model = YOLO(model_path)

                keys = np.linspace(0.1, 1, 10)
                conf_dict = {round(k,1):[] for k in keys}
                tp_fp = {round(k,1):[] for k in keys}
                bbox_iou = {round(k,1):[] for k in keys}


                for img_file in os.listdir(image_dir):
                    img_path = os.path.join(image_dir, img_file)
                    label_file = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))

                    image = cv.imread(img_path)
                    h, w = image.shape[:2]
                    labels = []
                    if os.path.exists(label_file):
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                cls, xc, yc, bw, bh = map(float, parts)
                                if int(cls) != 0:  
                                    continue
                                xmin = int((xc - bw / 2) * w)
                                ymin = int((yc - bh / 2) * h)
                                xmax = int((xc + bw / 2) * w)
                                ymax = int((yc + bh / 2) * h)
                                labels.append([xmin, ymin, xmax, ymax])

                    results = model.predict(source=image, save=False, conf=0.25, verbose = False)[0]  
                    boxes = results.boxes

                    for box in boxes:
                        pred_cls = int(box.cls)
                        if pred_cls != 0:  # Only for person class
                            continue

                        conf = float(box.conf)
                        rounded_conf = round_down_to_interval(conf, 0.1)
                        conf_dict[rounded_conf].append(conf)


                        pred_box = box.xyxy.cpu().numpy().flatten()


                        # Match with any GT box (IoU > 0.5)
                        # 1 for TP
                        # 0 for FP
                        matched = False
                        for gt in labels:
                            xi1 = max(pred_box[0], gt[0])
                            yi1 = max(pred_box[1], gt[1])
                            xi2 = min(pred_box[2], gt[2])
                            yi2 = min(pred_box[3], gt[3])
                            inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
                            box1_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                            box2_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
                            iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)
                            if iou > 0.5:
                                matched = True
                                break
                        
                        if matched: 
                            tp_fp[rounded_conf].append(1)
                            bbox_iou[rounded_conf].append(iou)
                        else:
                            tp_fp[rounded_conf].append(0) 

                prec = []
                bin_iou = []
                # for k in conf_dict:
                    # print(f"k: {k}, Avg confidence: {np.mean(conf_dict.get(k))}")
                for key, value in tp_fp.items():
                    if len(value) == 0:
                        pr = 0
                        avg_iou = 0
                    else:
                        pr = precision(value)
                        if len(bbox_iou[key]) != 0:
                            avg_iou = sum(bbox_iou[key])/len(bbox_iou[key])
                        else:
                            avg_iou = 0
                    # print(f"key : {key} Precision in this bin: {pr}")
                    prec.append(pr)
                    bin_iou.append(avg_iou)

                avg_conf_dict = {k: np.mean(v) for k, v in conf_dict.items()}
                total_gap = reliability_plot(avg_conf_dict, prec, bin_iou, loss, modl)
                diff = confidence_histogram(conf_dict, prec, bin_iou, loss, modl)

                file.write(f"\n{loss}\t{modl}\t{total_gap:.3f}\t{diff:.3f}")
            else:
                print(f"{model_path} not found not found hence movin on to the next model.")