# Network Calibration for CNN based object detectors
---

Most of the image detection model also provide a confidence score with each object prediction they make. The confidences are further used to get highly accuracte predictions by thresholding. But these confidences are often gauged overconfidently by the model and don't reflect the actual accuracy of the model. This gives us a misleading picture and hamper the overall metrics of the models as well. 

Confidence calibration refers to aligning the predicted confidence scores of a model with the true likelihood that the predictions are correct. A model is said to be well-calibrated if, for all predictions that have a confidence score of p, the accuracy of those predictions is also approximately p. 

For eg.: Among all predictions made with 70% confidence, roughly 70% should be correct.

## Loss Functions
---

Various types of auxilliary loss functions can be applied along with the conventional regression and classification losses of the model, used for computing the gradients and hence updating the activations.
I have personally studied and implemented 3 of such losses for getting a true picture of my model. The losses I implemented are as follows:

1. **MbLS** (Margin based Label Smoothing): which is also an implementation of the paper [The Devil is in the Margin:
Margin-based Label Smoothing for Network Calibration](https://arxiv.org/pdf/2111.15430)

2. **ACLS** (Adaptive and Conditional Label Smoothing): a carefully modified implementation for specifically object detection of the paper [ACLS: Adaptive and Conditional Label Smoothing for Network Calibration](https://openaccess.thecvf.com/content/ICCV2023/papers/Park_ACLS_Adaptive_and_Conditional_Label_Smoothing_for_Network_Calibration_ICCV_2023_paper.pdf)

3. **MDCA** (Multi-class Difference in Confidence and Accuracy): object detection implementation of the paper [A Stitch in Time Saves Nine:
A Train-Time Regularizing Loss for Improved Neural Network Calibration](https://arxiv.org/pdf/2203.13834)

#### Steps for Implementation

1. Find the class in your network reponsible for computing the loss.
2. append to the file the loss of your choice from /repo/losses/{custom_loss}.py


## Calibration Evaluation and Visualization

Calibration Plot and Confidence Histogram are both tools used to visualize the calibration of a model's confidence (i.e., how well the predicted probabilities match the actual accuracy).

### Calibration Plot

A calibration plot also called a reliability diagram visually shows how well the predicted confidence scores match the true outcomes.

Key Idea being:
- X-axis: The predicted confidence score (probability, e.g., 0.0 to 1.0).

- Y-axis: The observed frequency of correctness (actual accuracy for that confidence level).

The plot divides predictions into bins based on their predicted probability (confidence). For each bin, it shows the actual accuracy (the proportion of correct predictions) for all predictions that fall into that confidence range. A perfectly calibrated model will have a diagonal line.

<img width="419" height="392" alt="image" src="https://github.com/user-attachments/assets/f3c4c9bf-fbf3-4a31-8fa2-db8e3da369e9" />


### Confidence Histogram

A confidence histogram shows the distribution of the predicted confidence scores for all the model’s predictions.

Key Idea:
- X-axis: The predicted confidence (e.g., 0 to 1).
- Y-axis: The frequency of predictions at each confidence level.

The histogram shows how often the model predicts certain confidence levels (e.g., how many predictions were made with 90% confidence, 80% confidence, etc.). It gives us an idea of whether your model tends to be overconfident (many predictions near 1) or underconfident (many predictions near 0).

<img width="419" height="392" alt="image" src="https://github.com/user-attachments/assets/f06cf165-a48f-47d2-9e10-981f20141166" />
