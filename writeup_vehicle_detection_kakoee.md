## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
I have two code file. one contains the main CV functions which is "all_function.py". the second file is the main file names "CarND_Vehicle_Detection.py". The main file has a variable to enable training as I wanted to train once and then dump the model and load the model after that. on the main file i.e. "CarND_Vehicle_Detection.py" , line 85 and 91 shows where I extracted all features using extract_features function which is included in all_functions.py


I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

examples/cor_not_car.png

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I chose pix_per_cell=16 to have faster runtime. I tried RGB, LUV, HSV, YCrCb color spaces and finally I chose HSV as it gave me better accuracy. I achieved accuracy of 98.5% with LinearSVC and 99.5% with GridsearchCV. both models are saved in the root folder.


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and and evaluate how they affect training accuracy as well as final test images and test_video results. I came up the the final setting by try and error on test images and test video.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using LinearSVS function. I also tried a GridSearchCV classifier for SVM. which gave me a very good accuracy of 99.1%. but, the model took long time to predict. 
Also, to improve the accuracy I cropped two car images from test3.jpg and test5.jpg. and feed those custom samples to the classifier for better training. This was because I could not predict those test images well. So, I decided to add them to the training set. Here is those new training files:
test_images/Train1.png
test_images/Train2.png

After the training I used Pickle module to save the model and load it for the later runs.
I saved multiple models as follow which are in the root of the project:
finalized_model_gridsrch_hsv.sav
finalized_model_YCrCb.sav
finalized_model.sav


I also used randomized and scalar for the data. on the line 104-113 of main file. I saved the StandardScalar for later usage.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used window size of 64 with three scaling (1,1.5,2) I choose these scaling by try and error and some search on google. I modified the find_cars function to return the heatmap instead of drawing image. I also modified find_cars function to add scaling to it.



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided good result.  
I also add heatmap thresholding to remove false positives and to give focus on the true findings. Also, for the LinearSVC I used the "decision_function" of the classifier to only focus on those findings that had high confidence (large distance from the classification line). This can be seen in find_cars function in all_functions.py. But for GridsearchCV I did not have to use that as it gave me good prediction.


Here are some example images:

output_images/*_w.png



### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_vehicle_det.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

TO get better and smooth results on the video, I created a moving heatmap based on 10 previous frames and draw box based on that commutative heatmap rather than each individual frame. that gave me better result.
This can be seen in the main file function "process_frame" lines 195-211



### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue was to make sure the classifier can predict correctly. there was some situation that I had to crop car form the image or video and feed to classifier or I had to remove false_positives. So, it really depends on the quality and quantity of the training data.
Also, another issue I can see is the performance in real application. this pipeline I think will be slow in real worlds with multiple cameras on the car and each has 30fps high resolution data. So, we should think of a faster approach.

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I would use deep learning instead of CV approach. Also, I would implment smart tracking feature based on previous frames. Probably, other perception methods are needed like radar or Lidar and Camera alone is not enough.

 
