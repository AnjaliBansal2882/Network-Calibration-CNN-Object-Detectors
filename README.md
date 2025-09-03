# Network Calibration for CNN based object detectors
---

Most of the image detection model also provide a confidence score with each object prediction they make. The confidences are further used to get highly accuracte predictions by thresholding. But these confidences are often gauged overconfidently by the model and don't reflect the actual accuracy of the model. This gives us a misleading picture and hamper the overall metrics of the models as well. 

Confidence calibration refers to aligning the predicted confidence scores of a model with the true likelihood that the predictions are correct. A model is said to be well-calibrated if, for all predictions that have a confidence score of p, the accuracy of those predictions is also approximately p. 

For eg.: Among all predictions made with 70% confidence, roughly 70% should be correct.

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
