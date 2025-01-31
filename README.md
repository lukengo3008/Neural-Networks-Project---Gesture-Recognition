# Neural-Networks-Project---Gesture-Recognition

## I.	Introduction:
The goal of this gesture recognition project is to develop a Conv3D-based neural network capable of accurately recognizing five different gestures performed by the user to control a smart TV without a remote. The following write-up outlines the experiments conducted to fine-tune the model, the results obtained, and the rationale behind the decisions made during the model selection process.
## II.	Base Model Choice:
The base model chosen for this project is a Conv3D architecture, which is well-suited for processing video data and capturing both spatial and temporal features. The base model consists of three Conv3D layers, each followed by Batch Normalization, ReLU activation, and MaxPooling3D layers for spatial downsampling. Dropout layers with a rate of 0.5 are used for regularization. The final layer consists of a Dense layer with a Softmax activation function for multi-class classification.
### III.	Reasoning and Metrics for Modifications and Experiments:
1. Experiment 1 - establish a baseline performance of the initial Conv3D model with default hyperparameters:
   - Reason: Experiment 1 serves as a starting point to understand the model's initial capabilities without any modifications.
   - Metrics: The training and validation accuracies (0.36% and 0.42%, respectively) indicate that the model is not performing well in its initial state.
2. Experiment 2 - Scaling to Smaller Images (50x50):
   - Reason: Experiment 2 scales down the image size to 50x50 to see if smaller images lead to better performance and faster processing.
   - Metrics: The training and validation accuracies (0.39% and 0.42%, respectively) indicate that the model performance is slightly improved with smaller images, but the differences are not significant.
3. Experiment 3 - Further Scaling to Smaller Images (25x25):
   - Reason: Experiment 3 reduces the image size to 25x25 to test the effect of extremely small images on the model's accuracy.
   - Metrics: The training and validation accuracies (0.42% and 0.63%, respectively). Both training and validation accuracy has improved compared to Experiment 2, but the model is still not performing well. Smaller image sizes might be causing information loss.
4. Experiment 4 - Reducing Batch Size:
   - Reason: Experiment 4 aims to investigate the impact of reducing the batch size on model performance and overfitting.
   - Metrics: The training and validation accuracies (0.53% and 0.75%, respectively) increased compared to Experiment 3. A smaller batch size led to improvements, and the model may require smaller batches for better training.
5. Experiment 5 - Using Adam Optimizer:
   - Reason: Experiment 5 tests the performance of the Adam optimizer with the previously used SGD optimizer to see if it can lead to faster convergence and improved accuracy.
   - Metrics: The training and validation accuracies (0.22% and 0.2%, respectively) significantly dropped compared to previous experiments. Adam optimizer's adaptive learning rate helps the model converge faster but improve accuracy.
6. Experiment 6 - Reducing Number of Frames (10 frames):
   - Reason: Experiment 6 investigates the effect of reducing the number of frames in the video sequence on the model's performance.
   - Metrics: The training and validation accuracies (0.21% and 0.2%, respectively) are lower than Experiment 5, suggesting that using fewer frames may negatively impact the model's ability to capture temporal information.
7. Experiment 7 - Introducing Dropout Regularization:
   - Reason: Experiment 7 explores the impact of dropout regularization to prevent overfitting and improve model generalization.
   - Metrics: The training and validation accuracies (0.65% and 0.68%, respectively) significantly improved, indicating that dropout effectively reduced overfitting and improved the model's generalization.
8. Experiment 8 - Increasing Number of Epochs:
   - Reason: Experiment 8 tests the model with more epochs to see if the performance improves further.
   - Metrics: The training and validation accuracies (0.45% and 0.55%, respectively) have slightly decreased compared to Experiment 7, suggesting that the model might start overfitting with more epochs.
## IV.	Final Model Selection:
After conducting a series of experiments and evaluating the model's performance, we selected Experiment 7 as the final model due to the following reasons:
•	Experiment 7 achieved the highest training and validation accuracies (65% and 68%, respectively) among all experiments.
•	The model exhibited reduced overfitting with a dropout rate of 0.25, leading to improved generalization to unseen data.
•	The choice of 10 frames effectively captured essential temporal features within the gesture videos.
•	The image size of 50x50 provided enough spatial information for accurate gesture recognition.
•	The Adam optimizer with a learning rate of 0.0002 helped the model converge better, leading to improved accuracy.
## V.	Conclusion:
The final Conv3D model from Experiment 7 demonstrated excellent gesture recognition performance on the validation dataset. The chosen architecture and hyperparameters strike a good balance between complexity, regularization, and convergence. Further evaluation on a separate test dataset would be necessary to validate the model's robustness and practicality for real-world gesture recognition applications.
