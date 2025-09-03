# Network Calibration for CNN based object detectors
---

Most of the image detection model also provide a confidence score with each object prediction they make. The confidences are further used to get highly accuracte predictions by thresholding. But these confidences are often gauged overconfidently by the model and don't reflect the actual accuracy of the model. This gives us a misleading picture and hamper the overall metrics of the models as well. 
