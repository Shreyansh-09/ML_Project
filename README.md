# Brain-Tumor-MRI-Classification
## Project Title: Project Title: Intelligent Brain Tumor MRI Classification using CNN 

## Objective: Medical imaging, especially MRI for the brain, is crucial in diagnosing and treating diseases. This project utilizes Convolutional Neural Networks and artificial intelligence for classifying brain tumors in MRI scans.

## What is a brain tumor?

A brain tumor is a collection, or mass, of abnormal cells in your brain. Your skull, which encloses your brain, is very rigid. Any growth inside such a restricted space can cause problems. Brain tumors can be cancerous (malignant) or noncancerous (benign). When benign or malignant tumors grow, they can cause the pressure inside your skull to increase. This can cause brain damage, and it can be life-threatening.

## The importance of the subject

Early detection and classification of brain tumors is an important research domain in the field of medical imaging and accordingly helps in selecting the most convenient treatment method to save patients' life therefore.

## Basic Requirements

| Library                | Version                                    |
|------------------------|--------------------------------------------|
| Python                 | 3.7.10 | packaged by conda-forge | (default, Feb 19 2021, 16:07:37) [GCC 9.3.0] |
| TensorFlow            | 2.4.1                                      |
| Keras                  | 2.4.3                                      |
| Keras Preprocessing   | 1.1.2                                      |
| Matplotlib            | 3.4.1                                      |
| OpenCV                | 4.5.1                                      |
| scikit-learn          | 0.24.1                                     |


## Dataset

We have referred to the following Research Paper and the corresponding dataset used in the paper
[Paper: Classifying Brain Tumors on Magnetic Resonance Imaging by Using Convolutional Neural Networks](https://www.mdpi.com/2076-3417/10/6/1999)

The dataset utilized in this paper are Brain Tumor MRI dataset Msoud and we also used the same. [here](https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset).

### Dataset Details

This dataset contains **7022** images of human brain MRI images which are classified into 4 classes:

- glioma
- meningioma
- no tumor
- pituitary

However, we have used a subset of the above dataset, which consists of 2870 images in the training data and 394 images in the testing data. We did this because of limited computational resources (GPUs and RAM) and ran our model in the Kaggle Notebook.

### Data Pre-processing

We performed the following pre-processing steps :

- Normalization 
- Data Augmentation
- Convert Vector Classes to Binary Class Matrices 

## CNN Model 

In the first phase of our project, we proposed two baseline CNN architectures along with the LeNet5 and AlexNet, and a pipeline was made.

## Pre-trained Model

A pre-trained model is a model that was trained on a large benchmark dataset to solve a problem similar to the one that we want to solve. Accordingly, due to the computational cost of training such models, it is common practice to import and use models from published literature. For this project, I decided to use the **ResNet50** model to perform image classification for brain tumor MRI images.[Resnet50 Article](https://arxiv.org/abs/1512.03385)

For this project, I opted to leverage the advantages of pre-trained models, which are models that have been previously trained on extensive benchmark datasets to address problems akin to the one at hand. Given the substantial computational resources required for training such models, it is customary to adopt models from established literature to benefit from their learned representations. In this context, I selected the ResNet50 model for the task of image classification, specifically applied to the domain of brain tumor MRI images. The ResNet50 architecture, introduced by He et al., is renowned for its deep structure and residual learning, enabling more efficient training and improved performance on challenging tasks. The original ResNet50 paper can be found here.[Resnet50 Article](https://arxiv.org/abs/1512.03385)

Furthermore, I considered alternative pre-trained models to explore their efficacy in this application:

- DenseNet: DenseNet is a densely connected convolutional network that promotes feature reuse and facilitates gradient flow. To understand its architecture and advantages, refer to the original DenseNet paper here.[DenseNet Article](https://arxiv.org/abs/1608.06993)

- EfficientNet: EfficientNet is known for its efficient scaling of model depth, width, and resolution, achieving state-of-the-art performance with fewer parameters. The original EfficientNet paper can be accessed here. [EfficentNet Article] (https://arxiv.org/abs/1905.11946)

- InceptionV3: InceptionV3 utilizes a combination of inception modules to capture features at multiple scales. To delve into its architecture and design principles, consult the InceptionV3 paper here.[InceptionV3 Article](https://arxiv.org/abs/1512.00567)

Ensemble Learning (using Histogram of Gradient, Morphological features, etc.): Ensemble learning combines predictions from multiple models to enhance overall performance. For this approach, incorporating features such as Histogram of Gradient and Morphological features can provide diverse perspectives. While there may not be a specific paper for this ensemble approach, relevant literature on ensemble learning principles and feature extraction techniques can be explored.

## Training 

We have trained our model on CNN and fine-tuned it on pre-trained architecture.

## Training Time

- ResNet50:
Training Time: 15 minutes for 20 epochs
- EfficientNet B0:
Training Time: 20 minutes for 20 epochs
- DenseNet:
Training Time: 10 minutes for 20 epochs
- InceptionV3:
Training Time: 18 minutes for 20 epochs

## Evaluation

Our results are not benchmark against findings in relevant research papers, because we have not completely reproduced the research paper. We took an idea from that because there were so many additional things. While we observed the differences, when we compared the results with our models in this project, because in the referred Paper(1): Classifying Brain Tumors on Magnetic Resonance Imaging by Using Convolutional Neural Networks (https://www.mdpi.com/2076-3417/10/6/1999) using a deep neural network (DNN), they obtained an accuracy of 96.97\%, but our model which achieved highest accuracy on test data was 65.9\% (InceptionV3).\\
The areas for improvement were identified, particularly in the architecture of CNN, hyperparameter tuning, and some regularization and activation functions to play with. Future work could focus on refining to improve testing and training accuracy with custom CNN and fault analysis to pre-trained models for better fine-tuning,to achieve even more competitive results

## Metrics Used

- Accuracy is used here for our models and subsequent classification report is generated in the Jupyter Notebook. 
​
## Results
- ResNet50:
• Accuracy: Training - 82.40%, Validation - 71.25%, Testing - 55.00%

- EfficientNet B0:
• Accuracy: Training - 28.22%, Validation - 29.15%, Testing - 25.38%

- DenseNet:
• Accuracy: Training - 99.65%, Validation - 87.17%, Testing - 63.9%

- InceptionV3:
• Accuracy: Training - 94.60%, Validation - 83.39%, Testing - 65.23%

- Ensemble Model:
Testing Accuracy - 49%

## Contribution to the project 

All team members collaborated on research and gathered relevant literature on brain tumor classification using convolutional neural networks (CNNs) and intelligence techniques throughout the project lifecycle. Our collective efforts ensured the successful completion of the Brain Tumor Classification task, reflecting our shared commitment to project success, not up to the mark we say, but complete efforts were there.

## Contribution by Each Member :


- Abu Talha (12310020):
  As I mentioned in an earlier stage I would explore and integrate intelligence techniques, including feature selection and ensemble learning, to improve model performance. I used histogram of equalization and morphological feature extraction techniques to improve image quality, normalization, noise reduction object separation, etc. The testing accuracy I got in this case was 49/%. Though I did not get satisfactory results as per my expectations and plans, I learned from that. I will try to improve my model with this learning and will use some additional techniques mentioned in the conclusion section. The classification report would be there in the notebook. I collaborated with my team members to work on the process of training other models in the project, to enhance my knowledge and skill set.

- Shreyansh Faye (12310310):
  I have explored EffiecientNet B0 in place of VGG16: and ResNet: Adapt and fine-tuned. I have tried to build one custom CNN architecture as well for which I got an accuracy of almost 90% on training data but the test accuracy was coming out to be 40/% which clearly shows that the model was overfitting. The results obtained from ResNet were far better than EfficientNet B0 and custom CNN. I will further explore how can I make it better from here.

- Md Arif Khan (31900010):
I have explored InceptionV3: Fine-tune for brain tumor classification & DenseNet: a pre-trained model. The results obtained from this model are good. It may be the case of overfitting somewhere.
 
## Contributing

Thank you for considering contributing to the Brain Tumor MRI Classification project! Your involvement is essential for the success of this project and the improvement of its capabilities.

### How to Contribute

1. Fork the Repository: Start by forking the project on GitHub.
2. Clone the Repository: Clone the forked repository to your local machine using the following command:
   ```bash
   git clone https://github.com/AbuTalhaGT/Brain-Tumor-MRI-Classification.git

## Note
You can see more details about training steps and testing results inside [Group_8_Brain_Tumor_MRI.ipynb](https://github.com/AbuTalhaGT/Brain-Tumor-MRI-Classification/blob/main/Group_8_Brain_Tumor_MRI.ipynb)

