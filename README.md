# Active-Polyhedral-Scene-Matching
![](https://github.com/SajjadPSavoji/Active-Polyhedral-Scene-Matching/blob/main/Report/figure/poly1.png?raw=true)

In short My solution consists of four main modules: observation, feature extraction, comparison, active motion policy. Observation module receives a camera position and object ID from the Active Policy module and presents an image of the object. Feature extraction takes in the observation and creates informative features from it. Comparison module takes in those features and decides whether the two objects are the same. Such decision is then propagated to the Active Motion Policy where the next observation position is created. The result of my implementation is submitted for evaluation; it follows an original idea and presents a reasonable structure.

Given two polyhedral scene IDs $\{I_1, I_2\}$ decide whether the underlying 3D objects are the same or not. To solve this challenge one should follow an active vision scheme in which each observations are denoted as $\{ F_i^n, F_j^m \}_{n, m = 1}^{N}$ in which $i$ and $j$ are object indexes and N is the maximum number of view-points taken from an object. that being the case my algorithm consists of four major modules: observation, feature extraction, comparison, and motion policy. The feature extraction part itself consists of 6 parts: pre-processing, edge detection, line detection, intersecting lines, clustering joints, and graph representation.
<br><br>

![](https://github.com/SajjadPSavoji/Active-Polyhedral-Scene-Matching/blob/main/Report/figure/features.png?raw=true)

# Pre-Processing:
As a the generated polyhedrals all have the same color, in the pre-processing a gray-scale version is generated. After that, all the functions will be aplied to this image. It is worth mentioning that most edge detection bydefault only work with gray-scale images; in that sense this step is inevitable.

# Edge Detection:
Edge detection is an image-processing technique, which is used to identify the boundaries (edges) of objects, or regions within an image. Edges are among the most important features associated with images. We come to know of the underlying structure of an image through its edges. Computer vision processing pipelines therefore extensively use edge detection in applications. Edges are characterized by sudden changes in pixel intensity. To detect edges, we need to go looking for such changes in the neighboring pixels. Sobel Edge Detection is one of the most widely used algorithms for edge detection. The Sobel Operator detects edges that are marked by sudden changes in pixel intensity.
$$
G_x = K_x * I \qquad G_y = K_y * I \qquad G_{xy} \sqrt{G_x^2 G_y^2} \qquad \Theta = arcTan(G_y/G_x)
$$

# Line Detection
For the line detection part the Hough Line Transform is used. It is a transform used to detect straight lines only. To apply the Transform, first an edge detection pre-processing is desirable which was covered in the previous section. As you know, a line in the image space can be expressed with two variables $(m, b) or (r, \theta)$ For Hough Transforms, we will express lines in the Polar system.

# Line Intersection:
While the edges of the polydedral are important features, having them alone will not be of use as it will not be representative enough. As such, it is possible to extract the corners of the polyhedral too. To do so I am using a hybrid method wich combines the Harris corner detector with my own line intersection algorithm. As for the latter case, once a line equation is given as $y = mx+b$ one can use it to detect the intersection point of two lines. using the following formula:

$$
y_1 = ax+b,\quad y_2 = cx+d \quad \rightarrow \quad P_0 = \left(\frac{d-c}{a-b}, a\frac{d-c}{a-b}+c\right)
$$

This simple idea is combined with Harris corner detection. Corners are regions in the image with large variation in intensity in all the directions. One early attempt to find these corners was done by Chris Harris & Mike Stephens in their paper A Combined Corner and Edge Detector in 1988, so now it is called the Harris Corner Detector. He took this simple idea to a mathematical form. It basically finds the difference in intensity for a displacement of $(u,v)$ in all directions.

$$
E(u,v) = \sum_{x,y} \underbrace{w(x,y)}_\text{window function} \, \[{\underbrace{I(x+u,y+v)}}_{\text{shifted intensity}}-\underbrace{I(x,y)}_\text{intensity}\]^2
$$

The window function is either a rectangular window or a Gaussian window which gives weights to pixels underneath. We have to maximize this function $E(u,v)$ for corner detection. That means we have to maximize the second term. Applying Taylor Expansion to the above equation and using some mathematical steps (please refer to any standard text books you like for full derivation), we get the final equation as shown below. Then comes the main part. After this, they created a score, basically an equation, which determines if a window can contain a corner or not.


# Clustering Corners:
Since the corners detected in the previous step are far more than the actual corners, a clustering algorithm will be used to cluster them to unified corners. Once they are clustered, the center of each cluster will be used as the new object corner. For this specific task, I chose DBSCAN as it does not require the number of clusters in its hyper parameters.

# Graph Representation:
Last but not least, once the corners and edges of the polyhedral are extracted, these information will be treated as a underacted graph with scaled edges. This means that an adjacency matrix M is computed for each graph in which nodes are the polyhedral corners and the edges are its corresponding edges. As for the cost of each eadge, it is computed via the pixed distance of corners devided by the camera scale parameter $r$.
<br><br>
![](https://github.com/SajjadPSavoji/Active-Polyhedral-Scene-Matching/blob/main/Report/figure/ff.png?raw=true)
