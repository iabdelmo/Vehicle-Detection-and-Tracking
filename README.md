
# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Train_images_sample.png
[image2]: ./output_images/hog_feat.png
[image3]: ./output_images/findcar_func_output.png
[image4]: ./output_images/failed_detection.png
[image5]: ./output_images/multi_scale_reg1.png
[image6]: ./output_images/multi_scale_reg2.png
[image7]: ./output_images/multi_scale_reg3.png
[image8]: ./output_images/multi_scale_reg4.png
[image9]: ./output_images/success_detection.png	
[image10]: ./output_images/heatmap_img.png
[image11]: ./output_images/heatmap_after_threshold.png
[image12]: ./output_images/labeled_img.png
[image13]: ./output_images/label_box.png
[image14]: ./output_images/pipeline_output_testimgs.png							

[video1]: Vehicle-Detection-and-Tracking/project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cells from 1 to 5 of the IPython notebook  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `gray` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters during the lesson and in the project and my choice was based on the best classifier accuracy from the used parameters.

I decided to use YCrCb as I understood from searching that it has a good performance with the HOG Increasing the orientations has a very few effect on the accuracy I noticed also increasing the pixels per cell could give better accuracy and using all the channels of will lead to better accuracy than one channel. I reached with the accuracy to more than 98% which I assumed it will be enough

So I settled on the following parameters `YCrCb`,`orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using extracted HOG features form the training images "car and not car images".

I have split the extracted hog features to training and test sets using `train_test_split()` function from sklearn.

Then I have trained my linear SVM using the training feature vector using `fit()` function of sklearn.svm.

Then I applied testing for the output model to predict `20` samples of the test_set using `predict()` function

Finally I calculated the accuracy using `score()` function

The code for this part is implemented in cells: 6 and 7 in IPython notebook.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

 I decided to search the bottom half of the image with a scale of 1.5 (default search window is 64x64) and 75% overlapping which was suitable for the image I was testing on this case. I used for that the function `find_cars()` which can extract the hog features from the image and run the sliding window technique then where ever the classifier returns a positive detection the position of the window in which the detection was made will be saved

Here is an output of the implementation

![alt text][image3]

It performs well although there are multiple detection on the two cars. and that what shall be fixed afterwards. 

Also with fixed search window area and scale, The pipeline failed to detect the car in the below test image ; As it was marked by one window only and when the pipeline applied the heatmap threshold this detection will be ignored and marked as false positive detection.

![alt text][image4]

So to fix the above issues , I have applied multi-scale window searching that is illustrated in the below section.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features which provided a nice result.
I also applied an optimization for searching. As long as the car gets further from the car the search width decreases. Here are 4 images for the 4 scales I used for searching : 

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

and here is the result on a test image that was failed with the fixed searching window
and scale:

![alt text][image9]

As we can see now the car is detected by more than one searching window and this will solve marking the detection as false positive detection.

#### 3. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To be able to handle the multiple detections and false positives I needed to apply the heat map technique then apply a threshold below it I reject the detection.

The implementation of this part can be found in cells from 12 to 15 in the IPython notebook.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.

Here is the output of the previous test image:

![alt text][image10]

After threshold of 1 :

![alt text][image11]

Then I used the scipy.ndimage.measurements.label() function to collect detected objects:

![alt text][image12]

And here is the output on the test image:

![alt text][image13]

#### Then I implemented the whole pipeline in this function `VehTrackingProcessImage()` that can be used to detect vehicles from images or from videos.

I run the pipeline on all test images and here is the output:

![alt text][image14]



---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

I tried first to run the pipeline on the test video, the result was not bad but It was noisy and unstable, So I used the information of previous 15 video frames to add more confidence for the current detected objects. so I've done the heatmap on all the previous detections of the object not only on the current detection and a higher threshold also is applied to fit the higher heatmap.

Here's a [link to my video result](https://github.com/iabdelmo/Vehicle-Detection-and-Tracking/blob/master/project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

I started by trying one feature (Hog) which returned to perform nicely with YCrCb and the test accuracy was acceptable, then I used the sliding window technique to capture the detected cars but the output contained false positives and multiple detections which was handled afterwards using the heatmap. I've done a search optimization which enabled to search in a smaller area and not get distracted by the trees and cars from the other road. Finally the video performance has been improved by adding the detection information from previous frames.

The pipeline shall work well in most cases but for sure it needs some improvements like :



- Dynamically calculate the search area for the sliding window to be more robust specially for sharp curves.


- Try different types of classifiers as I found that such case can be completely handled using Neural Networks like YOLO which needs massive training data but for sure will give much better results

