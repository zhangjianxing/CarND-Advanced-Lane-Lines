## Writeup Template


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given 
a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of 
lane curvature and vehicle position.

[compare_dist]: ./output_images/compare_dist.jpg
[undistort_img_road]: ./output_images/undistort_img_road.jpg
[bindary_img]: ./output_images/bindary_img.jpg
[transformed_img]: ./output_images/transformed_img.jpg
[search_poly_from_img]: ./output_images/search_poly_from_img.jpg
[example_output]: ./output_images/example_output.jpg

[project_video_out]: ./output_images/project_video_out.mp4
[harder_challenge_video_out]: ./output_images/harder_challenge_video_out.mp4
[challenge_video_out]: ./output_images/challenge_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in 
"./P2.ipynb" (and wrote in `lib/camera_calibration.py`).  

I start by preparing "object points", which will be the (x, y) coordinates of the chessboard 
corners in the world. Here I am assuming the chessboard x is fix at 9 but y varies between 6 and 5. 
such that the object points are the varies for each calibration image.  
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with 
a copy of it every time I successfully detect all chessboard corners in a test image.  
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image
 plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion 
coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction 
to the test image using the `cv2.undistort()` function and obtained this result: 

![alt_text][compare_dist]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][undistort_img_road]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image 
(thresholding steps at lines # through # in `lib/color_and_gradient.py`).  
Here's an example of my output for this step.  

In function `color_and_gradient.img_to_binary()` uses combination of H, S, R channels 
(where H, S comes from HLS image, and R fom RGB image) to generate gray scale img. 
And In function `color_and_gradient.img_to_binary()` use Sobel function (kerner=3, thresholding=(30, 60)) 
to detect edge.

![alt text][bindary_img]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_perspective_transform_meta_data()`,
 which appears in lines 22 through 37 in the file `./lib/utils.py`. The `get_perspective_transform_meta_data()`
 function returns transform_matrix and inverse transform_matrix, Which is used by `cv2.warpPerspective()`
 function to perform perspective transform.
 I chose the hardcode the source and destination points in the following manner:

```python
import numpy as np
img_size_x, img_size_y = None, None # in put
src = np.float32(
    [[(img_size_x / 2) - 63, img_size_y / 2 + 100],
     [((img_size_x / 6) - 12), img_size_y],
     [(img_size_x * 5 / 6) + 90, img_size_y],
     [(img_size_x / 2 + 70), img_size_y / 2 + 100]])
dst = np.float32(
    [[(img_size_x / 4), 0],
     [(img_size_x / 4), img_size_y],
     [(img_size_x * 3 / 4), img_size_y],
     [(img_size_x * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source	| Destination| 
|:---------:|:----------:| 
| 577, 460	| 320, 0	 | 
| 201, 720	| 320, 720	 | 
| 1156, 720	| 960, 720	 | 
| 710, 460	| 960, 0	 | 

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][transformed_img]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?


The code for my lane-line detection and find their fit is located in `./lib/find_line_poly.py`.

1. `search_poly_from_scratch()` function iterate through 9 windows from bottom to top twice(one for each line), trying to find 
edges which representing lines. 

2. `search_poly_around_old()` function will try to detect lines based on line that detected in previous img.
Which might decrease computing time and improve accuracy.


![alt text][search_poly_from_img]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines in my code in `lib/measure_curve.py`. 

For radius of curvature: Did in `poly_to_polycr()` function: 

1. I convert left_fit and right_fit to real world fit, based on `ym_per_pix` and `xm_per_pix`

2. Then, in `measure_real_world_curvature_for_line()` function, I use `line_fit_cr` to find `curvature` for one line.

3. Finally, road has two lines and we need to take average of these two lines. `avg_curvature()` Did it 

For finding midpoint: Did in `measure_mid_position()` function:

1. set a point representing mid point of car

2. find distnace from mid point to both line

3. apply `xm_per_pix` to compute real world distance to mid point 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `./lib/map_lane.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][example_output]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a line to [project_video_out]

Here's a line to [harder_challenge_video_out]

Here's a line to [challenge_video_out]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, 
1. I simple use iamge channels to detect edge and use simple rule to find a line.
That worked because we assume the road is flat and the line is roughly straight.
When light condition is bad or the road is too twisty or line is missing, we might fail to detect a line 
and fail to detect a road.
