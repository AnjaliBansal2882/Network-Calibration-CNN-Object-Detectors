import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import typing


def data_read_prep(model1: list, model2: list, model3: list, type: str) -> None:

    with open("/home/Anjali/Desktop/Anjali_dev/calib_Plots/DECE_Plots/cf_0.25/DECE CF 0.25_noabs.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split("\t")
            if type in data[1]:
                print(data[1])
                if data[0] == 'Org':
                    indx = 0
                elif data[0] == "MBLS":
                    indx = 1
                elif data[0] == 'ACLS':
                    indx = 2
                else:
                    indx = 3                    
                if '8' in data[1]:
                    model1.insert(indx, float(data[2]))
                elif '10' in data[1]:
                    model2.insert(indx,float(data[2]))
                else:
                    model3.insert(indx, float(data[2]))



ft_type = ['scratch', '_FT', 'LPFT']   
for type in ft_type:   
    model1 = []
    model2 = []
    model3 = []
    data_read_prep(model1, model2, model3, type)
    data = [model1, model2, model3]
    print("data:\n", data)
    labels = ['Orig', 'MbLS', 'ACLS', 'MDCA']
    model_names = ['v8', 'v10', 'v11']

    # Set positions for the bars
    n_models = len(data)
    n_types = len(data[0])
    bar_width = 0.08
    group_spacing = 0.1

    # Compute positions
    positions = []
    start = 0
    for i in range(n_models):
        positions.append([start + j * bar_width for j in range(n_types)])
        start += n_types * bar_width + group_spacing

    # Flatten positions and data for plotting
    flat_positions = [p for group in positions for p in group]
    flat_heights = [val for group in data for val in group]
    # print(type(flat_heights[0]))

    # Set colors and labels
    colors = ['blue', 'hotpink', 'green', 'orange'] * n_models
    bar_labels = model_names[0:1]*4 + model_names[1:2]*4 + model_names[2:3]*4

    # Plot
    plt.figure(figsize=(6, 6))
    # bars = plt.bar(flat_positions, flat_heights, width=bar_width, color=colors, alpha = 0.3)
    positive_heights = [abs(h) for h in flat_heights]
    bars = plt.bar(flat_positions, positive_heights, width=bar_width, color=colors, alpha = 0.3)
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # center horizontally
            height + 0.005,                     # slightly above the bar
            f'{height:.2f}',                    # format value to 2 decimal places
            ha='center', va='bottom', fontsize=8
        )
    for i, h in enumerate(flat_heights):
        if h < 0:
            plt.text(flat_positions[i], abs(h) + 0.05, "Neg", ha='center', va='bottom', fontsize = 9)

        


    # Set x-axis ticks to be in the middle of each group
    group_centers = [np.mean(pos) for pos in positions]
    plt.xticks(group_centers, model_names)

    plt.xlabel(f'{type} Models')
    plt.ylabel('DECE')
    plt.title(f'Comparison of DECE on Different LOSS functions on YOLO {type}')

    legend_elements = [
        Patch(facecolor='blue', alpha = 0.3,label='Orig'),
        Patch(facecolor='hotpink', alpha = 0.3, label='MbLS'),
        Patch(facecolor='green', alpha = 0.3, label='ACLS'),
        Patch(facecolor='orange', alpha = 0.3, label='MDCA')
        # Patch(facecolor = 'Text', label = "Neg = OverCf")
    ]
    plt.legend(handles=legend_elements, loc=4)

    # plt.legend(labels, title='Types')
    plt.tight_layout()
    file_path = f"/home/Anjali/Desktop/Anjali_dev/calib_Plots/DECE_Plots/cf_0.25/comparison_plots/DECE_{type}.png"
    plt.savefig(file_path, dpi = 500, bbox_inches = 'tight')
    plt.show()
