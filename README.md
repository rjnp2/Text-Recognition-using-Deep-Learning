# Text-Recognition-using-Deep-Learning

%Creating a CRNN model to recognize text in an image.

# Requirements
   1. Tensorflow 1.8.0
   2. Flask
   3. Numpy
   4. OpenCv 3
   
# Description

Use Convolutional Recurrent Neural Network to recognize the text image without pre segmentation into words or characters. Use CTC loss Function to train.

## We will use the following steps to create our text recognition model.

1. Collecting Dataset
2. Preprocessing Data
3. Creating Network Architecture
4. Defining Loss function
5. Training model
6. Decoding outputs from prediction

# Dataset
We will use data provided by Visual Geometry Group. This is a huge dataset total of 10 GB images. Here I have used only 40000 images for the training set with 0.08% images for validation dataset. This data contains text image segments which look like images shown below:

![image-data](https://user-images.githubusercontent.com/58425689/91627135-ed316f80-e9d4-11ea-935d-2e3d1f317d75.png)

To download the dataset either you can directly download from this [link](https://www.robots.ox.ac.uk/~vgg/data/text/#sec-synth) or use the following commands to download the data and unzip.
 
    ------------
    wget https://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
    tar -xvzf mjsynth.tar.gz
    
# Preprocessing
Now we are having our dataset, to make it acceptable for our model we need to use some preprocessing. We need to preprocess both the input image and output labels. To preprocess our input image we will use followings:

1. Read the image and convert into a gray-scale image
2. Make each image of size (128,32) by using padding
3. Expand image dimension as (128,32,1) to make it compatible with the input shape of architecture
4. Normalize the image pixel values by dividing it with 255.

To preprocess the output labels use the followings:

1. Read the text from the name of the image as the image name contains text written inside the image.
2. Encode each character of a word into some numerical value by creating a Function ( as ‘a’:0, ‘b’:1 …….. ‘z’:26 etc ). Let say we are having the word ‘abab’ then our encoded label would be [0,1,0,1]
3. Compute the maximum length from words and pad every output label to make it of the same size as the maximum length. This is done to make it compatible with the output shape of our RNN architecture.

In preprocessing step we also need to create two other lists: one is label length and other is input length to our RNN. These two lists are important for our CTC loss( we will see later ). Label length is the length of each output text label and input length is the same for each input to the LSTM layer which is 31 in our architecture

# Model Architecture
we will create our model architecture and train it with the preprocessed data.

# Model = CNN + RNN + CTC loss

Our model consists of three parts:

## 1. Convolution Neural Network Layer
A convolutional neural network (CNN, or ConvNet) is a class of deep learning, feed artificial neural networks that has successfully been applied to analyzing visual imagery. CNN compares any image piece by piece and the pieces that it looks for in an while detection is called as features. The convolutional neural network to extract features from the image

There are five main operations in the CNN:
    a) Convolution.
    b) ReLU.
    c) Pooling or Sub

### a) Convolution layer

The primary purpose of Convolution in case of a CNN is to extract features from the input image. Each convolution layer takes image as a batch input of four dimension N x Color-Channel x width x height. Kernels or filters are also four dimensional (Number of feature maps in, number of feature maps out, filter width and filter height) which are set of learnable parameters (weights and biases). In each convolution layer, four dimensional convolution is calculate between image batch and feature maps by dot p between them. After convolution only parameter that changes are image width and height.

### b) Rectified Linear Unit

An additional operation called ReLU has been used after
every Convolution operation. A Rectified Linear Unit (ReLU)
is a cell of a neural network which uses the following
activation function to calculate its output given x:
R(x) = Max(0,x) 

### c) Pooling Layer
In this layer, the dimensionality of the feature map
reduces to get shrink maps that would reduce the parameters
and computations. Pooling can be Max, Average or Sum.
Number of filters in convolution layer is same as the number
of output maps from pooling. Pooling takes input from
rectified feature maps and then downsized it according to
algorithm. 

## 2. Recurrent neural network
Recurrent neural network to predict sequential output per time-step. 
Recurrent Neural Network(RNN) are a type of Neural Network where the output from previous step are fed as input to the current step. in cases like when it is required to predict the next word of a sentence, the previous words are required and hence there is a need to remember the previous words. The main and most important feature of RNN is Hidden state, which remembers some information about a sequence. RNN have a “memory” which remembers all information about what has been calculated.

Disadvantages of Recurrent Neural Network

1. Gradient vanishing and exploding problems.
2. Training an RNN is a very difficult task.
3. It cannot process very long sequences if using tanh or relu as an activation function.

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems. This is a behavior required in complex problem domains like machine translation, speech recognition, and more. LSTMs are a complex area of deep learning. It can be hard to get your hands around what LSTMs are, and how terms like bidirectional and sequence-to-sequence relate to the field. 

LSTMs help preserve the error that can be backpropagated through time and layers. By maintaining a more constant error, they allow recurrent nets to continue to learn over many time steps (over 1000), thereby opening a channel to link causes and effects remotely. we have three different gates that regulate information flow in an LSTM cell. A forget gate, input gate, and output gate.

## CTC loss function
CTC loss functio which is transcription layer used to predict output for each time step. CTC LOSS Alex Graves is used to train the RNN which eliminate the Alignment problem in Handwritten, since handwritten have different alignment of every writers. We just gave the what is written in the image (Ground Truth Text) and BLSTM output, then it calculates loss simply as aim to minimize negative maximum likelihood path.

CTC solves problems:
1. we only have to tell the CTC loss function the text that occurs in the image. Therefore we ignore both the position and width of the characters in the image.
2. no further processing of the recognized text is needed

### This network architecture.
Let’s see the steps that we used to create the architecture:

1. Input shape for our architecture having an input image of height 32 and width 128.
2. Here we used seven convolution layers of which 6 are having kernel size (3,3) and the last one is of size (2.2). And the number of filters is increased from 64 to 512 layer by layer.
3. Two max-pooling layers are added with size (2,2) and then two max-pooling layers of size (2,1) are added to extract features with a larger width to predict long texts.
4. Also, we used batch normalization layers after convolution layers which accelerates the training process.
5. Then we used a lambda function to squeeze the output from conv layer and make it compatible with LSTM layer.
6. Then used two Bidirectional LSTM layers each of which has 128 units. This RNN layer gives the output of size (batch_size, 31, 63). Where 63 is the total number of output classes including blank character.

![Screenshot from 2020-08-29 09-05-18](https://user-images.githubusercontent.com/58425689/91627417-e3107080-e9d6-11ea-83d0-1945e595bc68.png)
![Screenshot from 2020-08-29 09-05-36](https://user-images.githubusercontent.com/58425689/91627419-e4419d80-e9d6-11ea-8d9a-24b355e707e1.png)

# Loss Function

CTC loss is very helpful in text recognition problems. It helps us to prevent annotating each time step and help us to get rid of the problem where a single character can span multiple time step which needs further processing if we do not use CTC. If you want to know more about CTC( Connectionist Temporal Classification).

A CTC loss function requires four arguments to compute the loss, predicted outputs, ground truth labels, input sequence length to LSTM and ground truth label length. To get this we need to create a custom loss function and then pass it to the model. To make it compatible with our model, we will create a model which takes these four inputs and outputs the loss.

# Test the model

Our model is now trained with 400000 images. Now its time to test the model. We can not use our training model because it also requires labels as input and at test time we can not have labels. So to test the model we will use ” act_model ” that we have created earlier which takes only one input: test images.

As our model predicts the probability for each class at each time step, we need to use some transcription function to convert it into actual texts. Here we will use the CTC decoder to get the output text. 
