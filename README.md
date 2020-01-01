# DeepLearning2019
This is an assignment solutions for Deep Learning 2019 in Seoul National University, lectured by Sungroh Yoon


### Assignment0 : Set-up Environments

make conda environments for deep-learning assignments

```
tar -zxvf Assignment0.tar.gz
cd Assignment0
cd env
Anaconda envs directory path(ANACONDA_ENV_PATH) check in setup_env.sh
bash setup_env.sh
source activate deep-learning-19 (conda activate deep-learning-19)
pip install –r requirements.txt
source deactivate (conda deactivate)
```


### Assignment1 : Naive Neural Networks

Part 1: Data Curation
Practice loading and preprocessing of data using the notMNIST dataset
Implement a simple machine learning code using sklearn library

Part 2: Implementing Neural Networks from Scratch
Understand the deep learning models
Implement a simple deep learning model


Part 3: Neural Networks with TensorFlow
Understand the roles of hyperparameter
Practice TensorFlow code implementing deep learning models


### Assignment2 : Convolutional Neural Networks

Part 1: Implementing CNN
To understand CNN architecture before using the TensorFlow 
Implement forward / backward passes for (1) convolution layer and (2) max pooling layer

Part 2: Training CNN
Learn how to define, train, and evaluate CNNs with TensorFlow
Explore various hyperparameters to design a better CNN model

Part 3: Visualizing CNN
Learn how to visualize and interpret a trained CNN model
Implement the codes for generating (1) image-specific class saliency maps,(2) class representative images, and (3) adversarial examples


### Assignment3 : Recurrent Neural Networks

Part 1: Implementing RNN
To understand RNN architecture before using TensorFlow
Implement forward/backward of single timestep 
Entire sequence based on single timestep

Part 2: Image Captioning
Design RNN model for image captioning with TensorFlow
Explore various RNN structure and hyperparameters

Part 3: Language Modeling
Learn probability distribution of characters from our language

Part 4: Neural Machine Translation
Implement an attention
Train and evaluate your model

Part 5: Transformer
Explore hyperparameters and pick the best


### Assignment4 : Generative Adversarial Network

Part 1: Implementing VAE with MNIST data

Part 2: Implementing conditional-GAN with MNIST data

Part 3: Implementing conditional-GAN with Face data


### Assignment5 : Reinforcement Learning

Part 1: Implementing and Training a Deep Q-Network (DQN)
Understanding Q-learning with DQN papers
Implement 4 classes for DQN with TensorFlow
Environment: CartPole-v0

Part 2: Playing Atari games using a A3C agent [Optional]
Deal with more complex task
Environment: Pong-v0
