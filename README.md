# Indian Sign Language Recognition System

**Objective**

The objective of this project is to develop a real-time system capable of recognizing and interpreting Indian Sign Language (ISL) gestures using computer vision and machine learning techniques. The system aims to bridge the communication gap between deaf and hearing individuals by translating hand gestures into meaningful outputs. Key goals include:

- Capturing hand gestures via a webcam.
- Preprocessing and extracting robust features from the images.
- Classifying gestures using machine learning models.
- Providing real-time feedback to users.

**Methodology**
  
  The project follows a structured pipeline from data collection to real-time gesture recognition:
  1. Data Collection
     
     The sign language followed in the project is refrered from this image.
     
     ![Reference](https://github.com/Harshini-9/Indian-Sign-Language-Detection/blob/main/ISL_ML_PROJECT/ISL_gestures.jpg)
     
     - Tool: OpenCV for video capture.
     - Process:- Users perform gestures within a defined Region of Interest (ROI) on the screen.
               - 1,200 frames per gesture are captured and saved as JPEG images (resized to uniform dimensions).
               - Data is organized into folders, with each folder representing a unique gesture label.
  3. Data Augmentation
     
To enhance dataset diversity and prevent overfitting, the following transformations are applied:

- Brightness Modification: Random adjustment (factor: 0.7–1.3).
- Color Variation: Saturation adjusted randomly (range: 0.7–1.3).
- Random Flipping: Horizontal flipping with 50% probability.
- Random Rotation: Images rotated between -15° to 15°.

3.  Feature Extraction

- SURF Descriptors: Speeded-Up Robust Features (SURF) are extracted to capture keypoints and descriptors from images.
- K-Means Clustering: Mini-Batch K-Means reduces feature space dimensionality by clustering SURF descriptors.
- Visual Words & Histograms: Histograms of visual words (cluster centers) are generated for each image to quantify feature occurrences.

4. Model Training & Evaluation

Models Tested:

- K-Nearest Neighbors (KNN)
- Random Forest
- Logistic Regression
- Decision Tree
- Gradient Boosting
- Naive Bayes
- Support Vector Machine (SVM)

**Model Comparision results**
``` Model: KNN
Accuracy: 0.9531, Precision: 0.9531, Recall: 0.9531, F1 Score: 0.9531

Model: Decision Tree
Accuracy: 0.5086, Precision: 0.5086, Recall: 0.5086, F1 Score: 0.5086

Model: Naive Bayes
Accuracy: 0.9380, Precision: 0.9380, Recall: 0.9380, F1 Score: 0.9380

Model: Gradient Boosting
Accuracy: 0.9283, Precision: 0.9283, Recall: 0.9283, F1 Score: 0.9283

Model: Logistic Regression
Accuracy: 0.9626, Precision: 0.9626, Recall: 0.9626, F1 Score: 0.9626

Model: Random Forest
Accuracy: 0.9313, Precision: 0.9313, Recall: 0.9313, F1 Score: 0.9313
```

  *Best Model: SVM (linear kernel, regularization=0.1) achieved the highest accuracy.
  
  *Dataset Split: 80% training (28,000 images) and 20% testing (7,000 images) across 35 gesture classes.
  
  *Evaluation Metrics: Accuracy, precision, recall, F1-score, and confusion matrices.

5. Real-Time Gesture Recognition
- Pipeline:
   1. Capture video frames via webcam.
  
  2. Isolate ROI and apply Canny edge detection.
  
  3. Extract SURF features and create visual words.
  
  4. Classify gestures using SVM.
  
  5. Display results in real-time.

# Output Samples
[Letter i](https://github.com/Harshini-9/Indian-Sign-Language-Detection/blob/main/i.jpg)
[Number 2](https://github.com/Harshini-9/Indian-Sign-Language-Detection/blob/main/2.jpg)
  # Key Concepts & Technologies

Computer Vision
- OpenCV: For video capture, ROI isolation, and image preprocessing.
- Canny Edge Detection: Detects gesture boundaries through grayscale conversion, Gaussian blur, non-max suppression, and thresholding.
- SURF Algorithm: Efficiently extracts and describes interest points using Hessian matrix-based detection and orientation assignment.

Machine Learning
- Classical ML Models: Focus on non-neural-network approaches (e.g., SVM, Random Forest).
- Feature Engineering: SURF descriptors + visual word histograms for robust gesture representation.
- Performance Optimization: Regularization (e.g., SVM’s C=0.1) and ensemble methods (Random Forest with 100 trees).

  Data Handling
- Augmentation: Brightness, color, flip, and rotation adjustments to improve model generalization.
- Clustering: Mini-Batch K-Means for efficient feature space reduction.

# Conclusion
The Indian Sign Language Recognition System successfully demonstrates the potential of computer vision and machine learning in bridging communication gaps for the hearing-impaired community. By leveraging OpenCV for real-time gesture capture, SURF for robust feature extraction, and SVM for accurate classification, the system achieves high precision and recall in recognizing 35 distinct ISL gestures.

Key Achievements
- Real-Time Processing: The system processes gestures with minimal latency, making it suitable for interactive applications.
- Non-Neural Approach: Classical ML techniques (SVM, Random Forest) deliver strong performance without requiring deep learning resources.
- Scalability: The modular pipeline allows for easy expansion to new gestures or languages.
-  User Accessibility: Simple hardware (webcam) ensures affordability and ease of deployment.
