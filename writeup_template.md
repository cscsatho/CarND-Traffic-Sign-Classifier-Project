#**Traffic Sign Recognition** 



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[image1a]: ./examples/sample_sign1.png "Sample Sign"
[image2a]: ./examples/sign_stats.png "Sign Stats"
[image3a]: ./examples/signs_gr_hist_table.png "Sign Preprocessing"
[image4a]: ./examples/valid_acc.png "Validation accuracy change"

[image1b]: ./examples/1_max30.png "Speed limit (30km/h)"
[image2b]: ./examples/12_prio.png "Priority road "
[image3b]: ./examples/13_yield.png "Yield"
[image4b]: ./examples/35_ahead.png "Ahead only"
[image5b]: ./examples/36_sorr.png "Go straight or right"
[image6b]: ./examples/38_keepr.png "Keep right"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the np.unique to calculate summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

Sample sign:
![alt text][image1a]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many of each sign category are there in the training dataset.

![alt text][image2a]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale using ```cv2.cvtColor()``` because color images were matched poorly.
Then I enhanced the contrast using ```cv2.equalizeHist()``` to gain better matching.

Here is an example of an original image and an augmented images:

![alt text][image3a]

I was also trying to use ```exposure.adjust_sigmoid()``` and ```exposure.equalize_adapthist()```, but the latter one put an extreme load on the CPU thus would be unusable in real-time environments.

After normalizing the contrast I converted the [0-255] range of the pixel colors into float32 array of range [0-1].

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16   				|
| Fully connected		| linear xW+b, input 400, output 120    									|
| RELU					|												|
| Fully connected		| linear xW+b, input 120, output 84    									|
| RELU					|												|
| Fully connected		| linear xW+b, input 84, output 43    									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the ```tf.nn.softmax_cross_entropy_with_logits()``` along with the ```tf.train.AdamOptimizer()```. I was experimenting with introducing ```tf.nn.l2_loss()``` but it did not bring much improvement, so now it is "switched off" in the code.
I came to the conclusion that EPOCHs between 15-20 can be satisfactory, and I saw that when I use smaller batch sizes (e.g. <= 64) then the result tends to be more accurate.

As of the hyperparameters:
```
EPOCHS = 20
BATCH_SIZE = 64
DROPOUT1 = .75 # used after the 1st fully connected layer only during training
DROPOUT2 = .75 # used after the 2nd fully connected layer only during training
LEARNRATE = 0.00075 # the smaller the better
L2_BETA = 0 # I did not use L2 after all
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 96.6% 
* test set accuracy of 93.5%

A well known architecture was chosen
* I chose the LeNet architcture.
* The reason was - aside from being recommended during class - that recognizing traffic signs instead of numbers does not seem to be such a different task, in particular when using grayscale imges with the same size of 32x32. Noise can occur in both cases.
* The final model's accuracy seems to be acceptable, it's above 93% on both sets, furthermore it performs good on random external images. The accuracy percentages are fairly stable as well.

Validation accuracy values while iterating through the epocs (the yellow dot is the test accuracy):
![alt text][image4a]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I printscreened in Google Maps:

![alt text][image1b] ![alt text][image2b] ![alt text][image3b] 
![alt text][image4b] ![alt text][image5b] ![alt text][image6b]

The second image might be difficult to classify because there is a sticker on its front side.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction (w/ the top 5 softmax probs):

Matching stats of sign "Speed limit (30km/h)"   |    perc
-----------------------------------------------:|:------------
Speed limit (30km/h)                            |  99.7391224%
Speed limit (50km/h)                            |   0.2602126%
Speed limit (80km/h)                            |   0.0006517%
Speed limit (60km/h)                            |   0.0000093%
Speed limit (20km/h)                            |   0.0000007%



Matching stats of sign "Priority road"          |    perc.
-----------------------------------------------:|:------------
Priority road                                   | 100.0000000%
Yield                                           |   0.0000000%
Stop                                            |   0.0000000%
No passing                                      |   0.0000000%
Roundabout mandatory                            |   0.0000000%

Matching stats of sign "Yield"                  |    perc.
-----------------------------------------------:|:------------
Yield                                           | 100.0000000%
Ahead only                                      |   0.0000000%
Priority road                                   |   0.0000000%
No vehicles                                     |   0.0000000%
Keep right                                      |   0.0000000%

Matching stats of sign "Ahead only"             |    perc.
-----------------------------------------------:|:------------
Ahead only                                      |  99.2856979%
Turn right ahead                                |   0.7131097%
Dangerous curve to the left                     |   0.0007735%
Road work                                       |   0.0001638%
Keep left                                       |   0.0001607%

Matching stats of sign "Go straight or right"   |    perc.
-----------------------------------------------:|:------------
Go straight or right                            |  99.9999762%
Road work                                       |   0.0000262%
Ahead only                                      |   0.0000028%
End of no passing                               |   0.0000002%
Turn left ahead                                 |   0.0000001%

Matching stats of sign "Keep right"             |    perc.
-----------------------------------------------:|:------------
Keep right                                      |  99.9890447%
Turn left ahead                                 |   0.0054279%
No vehicles                                     |   0.0049481%
Yield                                           |   0.0002144%
Speed limit (60km/h)                            |   0.0001998%



The model was able to correctly guess all 6 traffic signs, which gives an accuracy of 100%. This is better than the results on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


