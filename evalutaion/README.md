# Calibration Evaluation
----

Evaluating a model's ability to predict the confidence scored close to its true likelihood of correctness is a crucial a part of the process as it helps to maintain a certain level of acuracy and compare the various methods deployed.
Confidence calibration, initially defined for image classification used **ECE**(Expected Calibration Error) for network calibration evaluation. It aims to ensure that for each interval of confidence the average confidence of the interval is equal to its average accuracy in the interval. 

1. ECE is given by the formula:

<img width="469" height="94" alt="Screenshot from 2025-09-03 10-39-44" src="https://github.com/user-attachments/assets/c76e1ff1-8cc1-4c11-abbb-75e699244c56" />

But since, Accuracy; given by the formula **((TP+TN)/(TP+TN+FP+FN))**, which also has True Negatives(TN) in denominator, is not known to us in object detection tasks, we devied to **DECE**(Detection Expected Calibration Error)

2. DECE as defined in the paper [Beyond Classification: Definition and Density-based Estimation of Calibration in
Object Detection](https://openaccess.thecvf.com/content/WACV2024/papers/Popordanoska_Beyond_Classification_Definition_and_Density-Based_Estimation_of_Calibration_in_Object_WACV_2024_paper.pdf) uses precision **(TP/(TP+TN))** instead of accuracy:

<img width="560" height="100" alt="image" src="https://github.com/user-attachments/assets/bbf48218-cc38-4f73-a441-ff3797f8adc3" />


