[//]: # (Image References)

[image1]: ./images/example_face.png "Sample Face"
[image2]: ./images/sample_human_output.png "Sample Human"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"
[image4]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image5]: ./images/sample_dog_output.png "Sample Dog"
[image6]: ./images/sample_cnn.png "Sample CNN"

## CNN Project: Dog Breed Classifier

This repository contains the final project of Udacity's Machine Learning Nanodegree, which is a Dog breed classifier. A Convolutional Neural Networks (CNN) and transfer learning in PyTorch was used to carry out the model.

## Project Overview

In this project, a machine learning model will be made that can be used both on a website and in a mobile app, where a series of images contributed by the user from the real world will be processed. The algorithm used will perform three tasks:

   - By providing the user with an image of a dog, the algorithm will identify the breed to which it belongs, giving an estimate.
   - If the image is of a human, the algorithm will identify the closest breed of dog.
   - In the event that the image is not of a dog or human, the model will throw an error informing us that the image is invalid.

As it is a multiclass classification (we have several classes that the dog or the human race can belong to), it is best to use a Convolutional Neural Network to solve the problem. To do this, we must follow three steps:

   - We will detect human images using existing algorithms such as OpenCV’s implementation of Haar feature based cascade classifiers.
   - To detect the dog images, we will use a pretrained VGG16 model.
   - Finally, once the image is identified as dog or human, we will pass it to a CNN model, which will process the image and make a prediction of the breed to    which it belongs out of the 133 available.


## Project Instructions

This project belongs to the Nanodegree of Machine Learning of Udacity. Despite the fact that it is a finished project, it is possible to do it from scratch, including the extra images added in this project, following these steps:

### Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/udacity/deep-learning-v2-pytorch.git
		cd deep-learning-v2-pytorch/project-dog-classification
	```
	
__NOTE:__ if you are using the Udacity workspace, you *DO NOT* need to re-download the datasets in steps 2 and 3 - they can be found in the `/data` folder as noted within the workspace Jupyter notebook.

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 
4. Download the test_images that are included in this repository.
5. Make sure you have already installed the necessary Python packages according to the README in the program repository.
6. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.
	
	```
		jupyter notebook dog_app.ipynb
	```

__NOTE:__ Amazon SageMaker has been used to carry out this project. Although it is not mandatory, it would be recommended to use said platform, in order to obtain similar results, as well as to facilitate the work, when using other tools on this platform.


## Detect Humans

Once the datasets are downloaded and analyzed, the first thing we are going to do is implement the logic to detect humans in the images, we use OpenCV's implementation of Haar feature-based cascade classifiers for this purpose.

OpenCV provides many pre-trained face detectors, we can see an example in the following image:

![Sample Face][image1]

To analyze the operation of OpenCV with our images, and see how little failure it is, we are going to write a human face detector, where we will obtain values close to 100% correctness.

## Detect Dogs

As before with humans, we use a pre-trained model to detect dogs in images, in this case a VGG-16 model. This pre-trained VGG-16 model returns a prediction for the object that is contained in the image.

We are going to make predictions with the Pre-trained model, in order to see its effectiveness, and then write a dog detector. In order to check to see if an image is predicted to contain a dog by the pre-trained VGG-16 model, we need only check if the pre-trained model predicts an index between 151 and 268 (inclusive).

In the same way as with the human detector, when analyzing our images, we see with the model it is capable of identifying dogs with a low percentage of incorrect classifications. In the image a VGG-16 model(citation: https://neurohive.io/en/popular-networks/vgg16/):

![VGG16 Model Figure][image3]


## Create a CNN to Classify Dog Breeds (from Scratch)

To solve the multiclass classification problem (races), a CNN model has been built from scratch. This model has 3 convolutional layers with a kernel size of 3 and stride 1. The first conv layer (conv1) takes the 224*224 input image and the final conv layer(conv3) produces an output size of 128.
I used the ReLU activation function and I used the pooling layer of (2,2) in order to reduce the input size by 2. The dimensional out is 133, produced for the two fully connected layers that we have here. A dropout of 0.25 is added to avoid over overfitting. After creating a model, we train and test it to meet the specification of a test accuracy of at least 10%. In the image a CNN example:

![Sample CNN][image6]


## Create a CNN to Classify Dog Breeds (using Transfer Learning)

Once an acceptable precision is obtained from the CNN that I have created from scratch, these results can be improved, for which we will use the transfer learning. Our CNN must attain at least 60% accuracy on the test set. 
I will use Resnet101 architecture which is pre-trained on ImageNet dataset, the architecture is 101 layers deep. The last convolutional output of Resnet101 is fed as input to our model. We only need to add a fully connected layer to produce 133-dimensional output (one for each dog category). The model performed extremely well when compared to CNN from scratch. With just 5 epochs, the model got 81% accuracy.


## Model Evaluation

After training and validating the model, we will test it to see if it meets the 60% specification. In this case, the CNN model created using transfer learning with ResNet101 architecture was trained for 5 epochs, and the final model produced an accuracy of 81% on test data. The model correctly predicted breeds for 680 images out of 836 total images.


## Write and test an Algorithm

Finally, we will write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither. Some sample output for our algorithm is this image:

![Sample Human][image2]                          ![Sample Dog][image5]

Of course, this repository contains just one example of what could be done with more images and improving the fit of the model by playing with the hyperparameters. Feel free to perform the appropriate tests in order to obtain the desired results.
