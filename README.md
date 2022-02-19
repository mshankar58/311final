# CSC 311 Introduction to Machine Learning -- Final Project
"Online education services, such as Khan Academy and Coursera, provide a broader audience with
access to high-quality education. On these platforms, students can learn new materials by watching
a lecture, reading course material, and talking to instructors in a forum. However, one disadvantage
of the online platform is that it is challenging to measure students' understanding of the course
material. To deal with this issue, many online education platforms include an assessment component
to ensure that students understand the core topics. The assessment component is often composed
of diagnostic questions, each a multiple choice question with one correct answer. The diagnostic
question is designed so that each of the incorrect answers highlights a common misconception."
Final Project Introduction File

<!-- ABOUT THE PROJECT -->
## About The Project


Our group are given raw data from a online education services and our goal is to predict students' correctness on questions that they haven't seen yet.
We started by applying existing machine learning algorithms such as Neural Network and K-nearest neighbour. Then we compare the performance between different models and implemented bagging emsemble on all ofthe models.
Lastly, we tried to improve the performace of Nerual Network model using various techniques.

About the mdoels:
* k-Nearest Neighbor: 
  * We tried two different ways of interpretations. One is impute by questions (find the questions that have simialr difficulties) and the other is impute by users (find users that have similar abilities)
* Item Response Theory: 
  * We calculated the probability by assigning each student with an ability value and each qustion with a difficulty value.
  * Then performed alternating gradient descent to maximize the log-likelihood
* Neural Networks:
  * We started with a two layer NN and gradually increased to 5 layers using Pytorch.

### Ensemble and Improvements

We combined above models using bagging ensemble to comapre the performance with individual models.

We tried using different activation functions for the layers, adding more layers to extend our
neural network model, and splitting our dataset by different features.
* Trial Method A: Using different activation functions for input layer and output layer. 
  * In this part, we implemented combinations of different activation functions like logSigmoid() and ReLU(). We included both the input layer and output layer when modifying the activation
functions.
* Trial Method B: Adding more layers to extend our neural network model. 
  * In this part, we added more layers to capture more details from the given dataset. We tried out models with two and three hidden layers, as opposed to the base model’s one hidden layer.
* Trial Method C: Splitting our dataset by different features. 
  * In this part, we split our dataset by different features like gender and premium pupil. These were provided as additional metadata.
