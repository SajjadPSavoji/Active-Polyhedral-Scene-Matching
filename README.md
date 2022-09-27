# Active-Polyhedral-Scene-Matching
![](https://github.com/SajjadPSavoji/Active-Polyhedral-Scene-Matching/blob/main/Report/figure/poly1.png?raw=true)

In short My solution consists of four main modules: observation, feature extraction, comparison, active motion policy. Observation module receives a camera position and object ID from the Active Policy module and presents an image of the object. Feature extraction takes in the observation and creates informative features from it. Comparison module takes in those features and decides whether the two objects are the same. Such decision is then propagated to the Active Motion Policy where the next observation position is created. The result of my implementation is submitted for evaluation; it follows an original idea and presents a reasonable structure.

![](https://github.com/SajjadPSavoji/Active-Polyhedral-Scene-Matching/blob/main/Report/figure/module.png?raw=true)

Given two polyhedral scene IDs $\{I_1, I_2\}$ decide whether the underlying 3D objects are the same or not. To solve this challenge one should follow an active vision scheme in which each observations are denoted as $\{ F_i^n, F_j^m \}_{n, m = 1}^{N}$ in which $i$ and $j$ are object indexes and N is the maximum number of view-points taken from an object. that being the case my algorithm consists of four major modules: observation, feature extraction, comparison, and motion policy.
