# HEMORRHAGE-DETECTION


Introduction:

This project focused on developing an automated system for hemorrhage detection and classification in head CT scans. The dataset comprised 100 normal head CT slices and 100 with hemorrhages. Multiple machine learning algorithms were applied to analyze the dataset, emphasizing their performance in detecting and classifying hemorrhages.

Requirements:

    Python 3.6+
    NumPy
    Pandas
    OpenCV
    Scikit-learn
    SciPy
    glob
    Matplotlib
    Tensorflow
    Keras

About the Project:

The project evaluated the performance of the following classifiers:

    KNN: Euclidean distance and earth mover's distance
    SVM: Linear, polynomial, RBF, sigmoid kernels
    ADABOOST
    Decision Tree
    Random Forest
    CNN (Convolutional Neural Network)

Feature Extraction:

Two approaches were utilized for feature extraction:

    Simple Approach: The images were resized using OpenCV, then flattened into a one-dimensional array.
    Histogram Approach: A color histogram was used to represent the distribution of pixel intensities across images.

After feature extraction, the data was split into training and test sets before being sent to the classifiers.

Classification Process:

Each classifier was trained on the data, and accuracy metrics were calculated based on their ability to classify the CT images.

    For CNN, the images were resized to 320x320 pixels and fed into a five-layer convolutional model for classification.

Performance:

The Convolutional Neural Network (CNN) showed the highest accuracy at 100%, while simpler methods like KNN and Random Forest achieved accuracies of 90% and 85%, respectively. In contrast, the histogram-based approach yielded lower performance compared to the simple resizing technique.

Conclusion:

The project demonstrated that convolutional neural networks are highly effective for image classification tasks in medical imaging, achieving perfect accuracy. The Simple approach also produced better results than the histogram-based method, confirming that resizing and simplifying images is sufficient for many classifiers.

The integrated approach mentioned in the research focused on combining active contours and decision tree classifiers to improve hemorrhage detection accuracy. Active contours were particularly useful in defining hemorrhage regions, while decision trees classified the hemorrhage types based on extracted features like area and perimeter. The system achieved an overall accuracy of 92.5%.
These are the requirements and results documentation.
