# Loss Functions
----

Various types of loss functions can be applied along with the conventional regression and classification losses of the model, used for computing the gradients and hence updating the activations.
I have personally studied and implemented 3 of such losses for getting a true picture of my model. The losses I implemented are as follows:

1. **MbLS** (Margin based Label Smoothing): which is also an implementation of the paper [The Devil is in the Margin:
Margin-based Label Smoothing for Network Calibration](https://arxiv.org/pdf/2111.15430)

2. **ACLS** (Adaptive and Conditional Label Smoothing): a carefully modified implentation for specifically object detection: [ACLS: Adaptive and Conditional Label Smoothing for Network Calibration](https://openaccess.thecvf.com/content/ICCV2023/papers/Park_ACLS_Adaptive_and_Conditional_Label_Smoothing_for_Network_Calibration_ICCV_2023_paper.pdf)

3. **MDCA** (Multi-class Difference in Confidence and Accuracy): object detection implementation of [A Stitch in Time Saves Nine:
A Train-Time Regularizing Loss for Improved Neural Network Calibration](https://arxiv.org/pdf/2203.13834)
