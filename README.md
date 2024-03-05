# Einstein Vision
This project aims to reimagine and enhance the visualization experience utilizing a series of videos captured from the cameras of a 2023 Tesla Model S.  the objective is to create an advanced, rendered video output that presents the frontal view captured by the car's cameras and also incorporates a visualization of the car itself, along with its surroundings.
For a comprehensive understanding of the methodologies and implementation strategies employed, please refer to the report provided.

### Approaches
1. Lane Detection
Lane segmentation and drivable region segmentation was done using classical approach of using Hough Lines from cv2. 
Reference: https://medium.com/computer-car/udacity-self-driving-car-nanodegree-project-1-finding-lane-lines-9cd6a846c58c

2. Object Detection and Classification	
Road objects in the videos such as vehicles, Pedestrians, Traffic lights, Road signs, etc. were detected using YOLOv5. It generates 2D bounding
boxes around the detected objects. The coordinates of the bounding boxes were tracked using this model and stored in a ’.csv’ file.
Reference: https://colab.research.google.com/github/changsin/DLTrafficCounter/blob/main/notebooks/traffic_counter_yolov5.ipynb#scrollTo=2CY-FOgFW5Gr

3. Depth Estimation
For a 3D scene construction, the depth of each object was tracked using transformer based MiDAS model. It predicts the distance of each pixel in an image from the camera that captured it.
Reference: https://pytorch.org/hub/intelisl_midas_v2/

4. Scene Building
For the final output visualization of the scenes, Bleder software was used. Assets for objects such as Vehichles,  Traffic signal, Stop Sign, Traffic Cone, Traffic Pole, Speed Sign and Pedestrian were spawned at each scene using the results from object detection and depth estimation.
Reference: https://drive.google.com/drive/folders/1Shr6cQDLWiov53f1tXZArhBBfoPczNTp

### Results
The output videos for driving scenes are given in Videos folder. 

The outputs for a few indivisual images are given below:

<p align="center">
  <figure>
    <img src="Output Images/scene5frame351/frame351.png" alt="Front View" width="200"/>
    <figcaption>Fig.1 - Input Image</figcaption>
  </figure>
  <figure>
    <img src="Output Images/scene5frame351/yolo351.png" alt="Object Detection" width="200"/>
    <figcaption>Fig.2 - Object Detection using YOLO</figcaption>
  </figure>
  <figure>
    <img src="Output Images/scene5frame351/depthmap351.png" alt="Depth Map" width="200"/>
    <figcaption>Fig.3 - Depth Map Generation</figcaption>
  </figure>
  <figure>
    <img src="Output Images/scene5frame351/blender351_view2.png" alt="3D Rendered View" width="200"/>
    <figcaption>Fig.4 - 3D Rendered View with Blender</figcaption>
  </figure>
</p>

<p align="center">
  <figure>
    <img src="Output Images/scene5frame1774/frame1774.png" alt="Front View" width="200"/>
    <figcaption>Fig.1 - Input Image</figcaption>
  </figure>
  <figure>
    <img src="Output Images/scene5frame1774/yolo1774.png" alt="Object Detection" width="200"/>
    <figcaption>Fig.2 - Object Detection using YOLO</figcaption>
  </figure>
  <figure>
    <img src="Output Images/scene5frame1774/depthmap1774.png" alt="Depth Map" width="200"/>
    <figcaption>Fig.3 - Depth Map Generation</figcaption>
  </figure>
  <figure>
    <img src="Output Images/scene5frame1774/blender1774_view2.png" alt="3D Rendered View" width="200"/>
    <figcaption>Fig.4 - 3D Rendered View with Blender</figcaption>
  </figure>
</p>


<p align="center">
  <figure>
    <img src="Output Images/scene4frame261/frame261.png" alt="Front View" width="200"/>
    <figcaption>Fig.1 - Input Image</figcaption>
  </figure>
  <figure>
    <img src="Output Images/scene4frame261/yolo261.png" alt="Object Detection" width="200"/>
    <figcaption>Fig.2 - Object Detection using YOLO</figcaption>
  </figure>
  <figure>
    <img src="Output Images/scene4frame261/depthmap261.png" alt="Depth Map" width="200"/>
    <figcaption>Fig.3 - Depth Map Generation</figcaption>
  </figure>
  <figure>
    <img src="Output Images/scene4frame261/blender261_view2.png" alt="3D Rendered View" width="200"/>
    <figcaption>Fig.4 - 3D Rendered View with Blender</figcaption>
  </figure>
</p>

### References
1. https://rbe549.github.io/spring2023/proj/p3/

