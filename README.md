
---

**Behavioral Cloning Project**

The goals  of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report 
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_nvidia_6.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Before I decided to settle for the final model I have tried two model architucture to extract feature. The first one is the LeNet which consitst of normalization layer, 2 convolution layers, 2 fully connected layers and the second one is the NVIDIA end to end model which consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers.

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

To protect over fitting of the model I have tried to impliment dropout  regularization after the first conolutional layer and the first fully connected layer. I also have tried the 'L2' regularization technique but both haven't imporved model performance.  When I included the regularization technique the model most of the times doesn't drerive well, The model has difficulties to stay longer on track during simulation. As the dropout method used probability to keep some of the data for trainning the CNN model, it might have left some critical data point sepecifically on the curve, that leads the simulation to steer off track when approaching curves. What helps more to better model performance is the standaerdization of the streering angle measuremet. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I have used the key board to collect the training data which was actually very challenging to get an accurate data, as the car sometimes go off track while collecting and difficult to drive smoothly on the track. For that I have to collect training data seveeral times until I believe that I have got appropraite dataset.  I have actaully collected training over half dozen of times. The final dataset I used for training contains dataset of about 6 laps, 2 lap driving center, one lap dring smoothly along curves, one lap recovery and  two laps reverse driving. This finally gets me to collect more than 15,000 datasets which helps train the model better. 

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As I explained above before I settle for the final model I have used two well know model in deep learning the LeNet and NVIDIA end-to-end models. For the final test I have collected 6 laps of data from diverse perspective, center, recover and reverse drive, which produces about more than 47k datasets in total before agumentation.  The two models employ convolution neural networks model with different architucture. Before training these models and decide which one to go with, I used data agumentation technique to get more data or information for training the model. After augmentation of data, the total dataset become more than 95k. The dataset(i.e images and steering angles) were split into training and validation sets. For both models the the mse for training set is less than that of the mse for validation sets. Howere it is only the NVIDIA end-to-end mode which was able to drive that car authonomous around the track successfully.  

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Attempts was made to modify the NVIDIA model to perform better or to reduce overfitting or underfitting, however every time I introduce regularization technique such as droput or L2 the model prefoms worse,So I chose to not modify the original model architecture, and simple extract features from the model. The final model architucture is depicted in the figure below. The model consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers


The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text](https://github.com/kulu80/CarND_Behavioral_Cloning_P3/blob/master/cnn-architecture-624x890.png)

#### 3. Creation of the Training Set & Training Process
Before the gathering the final dataset for training, I have collectd training data by simply driving along the center track for 4 laps, 3 laps and even 6 laps, but these dataset fail to produce good dataset to appropraitly train the model to autonomously drive along the first track. The final dataset used for training the final model is collected from 2 center driving laps, 1 driving smoothly along the curve, 1 lap recover driving with out stopping recording as through the whole lap, and 2 lap reverse driving which yeild good training dataset. The figure below shows records from all cameras driving center of the track. 

![alt text](https://github.com/kulu80/CarND_Behavioral_Cloning_P3/blob/master/sample_image.png)

The total data set after agumentation was applied is 95496,The agumentation was made by filiping the image (figure below) and multipling the corrosponding streeting angle by -1. The figure below shows the histogram of the original data set and after the data set was standardized.

![alt text](https://github.com/kulu80/CarND_Behavioral_Cloning_P3/blob/master/histogram_ori_scaled.png)

For model training the dataset was shuffled and split into 80% training data and 20 % validation data. Due to memory issue I only trained the model for 3 epochs. I tried to use the GPU in AWS however after training the model and copying it to my local machine the model didn't work to drive the car autonomously. So decided to use my laptop for training the model.  

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:


![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
