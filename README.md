# Active-Polyhedral-Scene-Matching
![](https://github.com/SajjadPSavoji/Active-Polyhedral-Scene-Matching/blob/main/Report/figure/poly1.png?raw=true)

In short My solution consists of four main modules: observation, feature extraction, comparison, active motion policy. Observation module receives a camera position and object ID from the Active Policy module and presents an image of the object. Feature extraction takes in the observation and creates informative features from it. Comparison module takes in those features and decides whether the two objects are the same. Such decision is then propagated to the Active Motion Policy where the next observation position is created. The result of my implementation is submitted for evaluation; it follows an original idea and presents a reasonable structure.
