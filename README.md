# Active-Polyhedral-Scene-Matching
![](https://github.com/SajjadPSavoji/Active-Polyhedral-Scene-Matching/blob/main/Report/figure/poly1.png?raw=true)

In short My solution consists of four main modules: observation, feature extraction, comparison, active motion policy. Observation module receives a camera position and object ID from the Active Policy module and presents an image of the object. Feature extraction takes in the observation and creates informative features from it. Comparison module takes in those features and decides whether the two objects are the same. Such decision is then propagated to the Active Motion Policy where the next observation position is created. The result of my implementation is submitted for evaluation; it follows an original idea and presents a reasonable structure.
<br>
<br>

![](https://github.com/SajjadPSavoji/Active-Polyhedral-Scene-Matching/blob/main/Report/figure/module.png?raw=false)

Given two polyhedral scene IDs $\{I_1, I_2\}$ decide whether the underlying 3D objects are the same or not. To solve this challenge one should follow an active vision scheme in which each observations are denoted as $\{ F_i^n, F_j^m \}_{n, m = 1}^{N}$ in which $i$ and $j$ are object indexes and N is the maximum number of view-points taken from an object. that being the case my algorithm consists of four major modules: observation, feature extraction, comparison, and motion policy.
<br><br>

![](https://github.com/SajjadPSavoji/Active-Polyhedral-Scene-Matching/blob/main/Report/figure/features.png?raw=true)
The feature extraction part itself consists of 6 parts: pre-processing, edge detection, line detection, intersecting lines, clustering joints, and graph representation.

# Pre-Processing:
As a the generated polyhedrals all have the same color, in the pre-processing a gray-scale version is generated. After that, all the functions will be aplied to this image. It is worth mentioning that most edge detection bydefault only work with gray-scale images; in that sense this step is inevitable.

# Edge Detection:
Edge detection is an image-processing technique, which is used to identify the boundaries (edges) of objects, or regions within an image. Edges are among the most important features associated with images. We come to know of the underlying structure of an image through its edges. Computer vision processing pipelines therefore extensively use edge detection in applications. Edges are characterized by sudden changes in pixel intensity. To detect edges, we need to go looking for such changes in the neighboring pixels. Sobel Edge Detection is one of the most widely used algorithms for edge detection. The Sobel Operator detects edges that are marked by sudden changes in pixel intensity. This approach states that edges can be detected in areas where the gradient is higher than a particular threshold value. In addition, a sudden change in the derivative will reveal a change in the pixel intensity as well. With this in mind, we can approximate the derivative, using a 3×3 kernel. We use one kernel to detect sudden changes in pixel intensity in the X direction, and another in the Y direction. These are the kernels used for Sobel Edge Detection:

```math
$K_x =$\begin{bmatrix}
-1 & 0 & +1\\
-2 & 0 & +2\\
-1 & 0 & +1
\end{bmatrix}
\qquad 
$K_y =$\begin{bmatrix}
+1 & +2 & +1\\
0 & 0 & 0\\
-1 & -2 & -1
\end{bmatrix}
```

