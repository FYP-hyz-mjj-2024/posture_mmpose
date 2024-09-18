# A Posture Analysis Model to Detect Cell-Phone Usages

# Methodology
&emsp; Before talking more about the project, we would like to share our methodology. Notice that this is just a 
preliminary note for the project, not the final report.
![methodology.png](project_documents%2FREADME_images%2Fmethodology.png)

# Personnel
- Group Members: Mai Jiajun, Huang Yanzhen
- Supervisor: Prof. Bob Yibo Zhang

# Schedules
## Step 0. Data Gathering & Feature Selection
### 0.0 Data Gathering
&emsp; From arbitrary sources, gather plenty of usable image data where 
the posture of a person is clearly given in the picture.

### 0.0.1 Videos to Images
&emsp; Due to the lack of resources and resource varieties on the internet, we
decided to use ourselves as the dataset source. We capture videos that surrounds
us for a cycle, and crop each frame as a data source.

### 0.0.2 Variables
&emsp; There are a few variables that needs to be covered to ensure variety:
- Left/Right arm
- Angle of elbow
- Height angle of phone
- Height angle of face

### 0.1 Feature Selection
&emsp; Decide which features is used to decide whether a person is using a phone.
A feature of a datapoint (a person) is one of the many numerical data of the person's
skeleton. E.g., an angle of a key joint.


## Step 1. Skeleton Extraction & Image Annotation
### 1.0 Extract Skeleton
&emsp; Extract the skeleton of the person with a pre-trained model and retrieve
the decided numerical data of the key feature. Mediapipe is used.

&emsp; According to Google, a set of landmarks contains 33 key landmarks, each
represented by a tuple of their x, y, z coordinates and a visibility factor.

### 1.1 Transcribe
&emsp; Transcribe landmark coordinates of a single person into a vector of key features.
Here, we calculate the angles of some key-joints using those coordinates, where a vector
of key angles we considered is regarded as the key feature vector of this person.

&emsp; In one word: Extract & Select info from landmarks into a vector of angles.

### 1.2 Annotate Image
&emsp; For each image of a person, after extracting the key features, store this person
into a vector of key features. Then, multiple people with multiple vectors of key features 
would form a feature matrix. For instance, the feature matrix for class "using" is below.

| feature_1 | feature_2 | feature_3 | ... | feature_m | label |
|-----------|-----------|-----------|-----|-----------|-------|
| 80.096    | 73.637    | 19.399    | ... | 170.191   | 1     |
| 92.949    | 78.624    | 22.692    | ... | 175.262   | 1     |
| ...       | ...       | ...       | ... | ...       | ...   |
| 17.473    | 51.129    | 63.18     | ... | 171.259   | 1     |

&emsp; The shape of this matrix is PERSON_NUM * (FEATURE_NUM + 1). A row in this matrix denotes 
a feature vector of a person. A column in a matrix denotes the distribution of a feature among
all the people. The *label* column is the manual classification result of the person, since this 
is considered a supervised learning problem.

&emsp; For each class (which is either "using" or "not_using"), we get such a matrix. Therefore,
eventually we will get two such matrices. Eventually, both matrices will be stored in a .csv file.


## Step 2. Self-Classification Model Training
### 2.0 Gather Data
&emsp; Arrange annotated data into a single dataframe table to feed into the model.
We have 2 source .csv files. Extract both of them and concatenate them in the first axis (merge
rows) to form a training dataframe.

### 2.1 Train a Model
&emsp; Using Random Forest / SVR / CNN, a model is trained based on the datasets. To ensure
accuracy, random shuffle is applied to the dataset and the 100 separate model is trained at once.
After that, select the model with the accuracy closest to 0.85 as our target model (temporary measure).

### 2.2 Stability & Reliability
&emsp; Currently, there are two kinds of instabilities.

&emsp; Firstly, the training accuracies are unstable. As we train 100 models at the same time (without one interfering
another), the accuracy fluctuates quite evidently and the standard deviation is high. According to our discussion,
there are many reasons for this to happen, e.g., inadequacy of data samples, bad model/training method usages, etc. 
However, we are not sure whether this is really unacceptable or not, since we selected the model with the accuracy 
closest to 0.85 as our target model among the 100 trained ones, and it performed "visibly" better than expected.

#### Preliminary Test of Model Training Performance
| Num of Training Data per Class | Accuracy Std Dev |
|--------------------------------|------------------|
| 11                             | ~0.18            |
| 21                             | ~0.13            |
| 31                             | ~0.10            |
| 40                             | ~0.09            |
| 71                             | ~0.07            |

&emsp; Secondly, the model selected itself is not stable. During the demo, we have encountered a **jump** in the 
classification text. To be specific, it occasionally glitches between "using" & "not using" in a very short time, like
a millisecond. This is something that needs to be improved.


## Step 3. Model Application
### 3.0 Multi-Personal Detection
&emsp; To enable the model to detect multiple people in a single image. This is very important because that the
pedestrians always appear together, and we could not afford to run too many instances of the model from scratch
at the same time. Therefore, measures must be taken to allow a model to detect multiple people in a single image.

&emsp; There are at least two ways of reaching this goal.

#### 1. Mediapipe Only (Not Recommended)
&emsp; Feed an entire frame into mediapipe, and mediapipe detects multiple sets of landmarks. This is not recommended
since mediapipe is weak in this part.

#### 2. YOLO + Mediapipe
&emsp; Feed a frame into YOLO first. YOLO would extract people (pedestrians) in the frame, and return multiple sub-images.
Then, feed those sub-images into mediapipe one-by-one to get the skeletons of multiple people, thus the classification
results (using cellphone or not).

![model_application.png](project_documents%2FREADME_images%2Fmodel_application.png)

### 3.1 Performance Analysis
#### 3.1.1 Addressing the issue
&emsp; For each frame of the video source, there are three different model instances running. (Probably more in the future :) )
YOLOv5s (Temporary), Mediapipe Posture Detection Model, as well as our Self-Trained Classification Model. Therefore,
computing in each frame is quite time & performance consuming. There are multiple measures to improve running 
performance and reduce glitching.
- Hardware
  - Use CUDA (Nvidia) or MPS (Apple M-Series Chip) to accelerate YOLO model.
  - Maybe (?) use a camera with a lower resolution. We don't think we need a high-res camera just for posture detection.
- Software
  - Reduce looping for each frame. 
    - The ideal time complexity of this model is O(detection_target_num_per_person * people_num * frame_num).
  - Use lambda for-loop instead of regular for-loop.
  - Reduce re-instantiation of models. Use closures & dependency injection instead.
  - Reduce useless function calls caused from over-encapsulation.
- Models
  - YOLOv5s: This is just a temporary solution. Further investigation will be conducted in attempt to find the lightest object detection model that could realize our tasks.
  - Self-Trained Classification Model: Train this model with different approaches for better performance.

#### 3.1.2 Stress Testing
&emsp; We used a 5-minute live virtual cam stream pushing of Time Square in New York City to stress test the 
performance of the machine. According to our experiment, the overall performance is dominated by the mediapipe instance
(displayed as the "Classification Time" curve in the figure).

Testing Source: 
- https://www.youtube.com/watch?v=fR22wuArNQY&t=357s (~10:00 - ~15:00), using OBS Virtual Camera.

Hardware Condition: 
- GPU: Nvidia GeForce RTX 4060 Laptop
- CPU: Intel Core i7-13650HX

![performance_analysis_stress.png](project_documents%2FREADME_images%2Fperformance_analysis_stress.png)

#### 3.1.3 Regular Testing
&emsp; We used the built-in webcam of the computer to perform regular testing. Compared to the performance in stress
testing, regular testing gives us a higher framerate and a smaller difference between the time consumption of YOLO
object detection and the mediapipe posture detection.

Testing Source: 
- Built-in Webcam.

Hardware Condition: 
- GPU: Nvidia GeForce RTX 4060 Laptop
- CPU: Intel Core i7-13650HX

![performance_analysis_regular.png](project_documents%2FREADME_images%2Fperformance_analysis_regular.png)

### 3.1 Face Detection & Comparison
**Main Focus on the next semester.**
&emsp; Since we are able to detect who is using the cellphone by restricting the self-trained model in a sub-frame from
an object detection model, we could even do more advanced tasks on the basis of the existence of sub-frames.

Some nice study sources to check out:
- [Mediapipe Face Detector](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python)
- [Face Comparison](https://github.com/YanzhenHuang/face-comparison)

### 3.2 Database Connection & Frontend Development
&emsp; This should be the last part of the project. This may involve server registration, database connection, and
frontend development.

## To be continued...