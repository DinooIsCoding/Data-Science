### AI, Machine Learning, and Deep Learning: AI for Advanced Image Analysis

![image](https://github.com/dino-3007/Data-Science/assets/109076114/451faba2-5038-4b73-babb-0d993ea8e86a)


#### **Conventional Machine Learning Vs. Deep Learning for image analysis**
###### - Conventional Machine Learning can learn from a small amount of data, but the engineer need to handpick features to feed into a classification algorithm (such as Random Forest or SVM). It may not work well on datasets distinct from the training data because the handful of parameters used by Machine Learning cannot be tuned to anticipate the variability in future data.
##### *Features can be obtained from training images through the use of digital image filters such as Sobel, Entropy, and Gabor.
###### - Deep Learning networks trained on extensive datasets can be utilized as a method for feature extraction instead of manual feature crafting. Deep Learning does not require hand-tuning of features. It automatically optimizes millions of parameters during training without humans explicitly engineering the features. Deep Learning learns from the current training data so if the training data does not contain sufficient examples of the variations, the model may not perform well on those variations.

### How to train custom AI models for image segmentation

#### **What is image segmentation?**
###### Image segmentation is the process of dividing an image into various sections corresponding to different regions of similarity, referred to as **regions of interest (ROI)** in scientific terminology. These regions represent original image in a way that is easier to analyze.

#### **Algorithms for image segmentation**

###### - **Otsu's segmentation method**:
######    + provides a way perform automatic segmentation using the histogram threshold approach.
######    + returns a single intensity threshold value that separates pixels into either foreground or background classes.
######    + is a global threshold method and assumes the image is homogenous and follows a bimodal distribution
###### => May not be ideal for noisy images or showing multiple regions with similar mean grey levels but varying textures. However, It is the preferred choice for simple segmentation tasks.



# To be continued
