import cv2
import numpy as np

corners = np.array([[0, 0], [0, 1], [2, 0]])
# given corner computer the distance matrix

M = get_graph(corners, 1)
print(M)