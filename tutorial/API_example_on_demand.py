from tokenize import Special
from libs import *
from utils import *

cam_r = 1/10
cam_theta = -1 * np.pi/6
cam_phi = np.pi/2
r_deg = 0 * np.pi


ID = '3b4018ba-16e6-48b8-be42-9ecb451d9796'
fr = np.array([cam_r, cam_theta, cam_phi])
cam_qx, cam_qy, cam_qz, cam_qw = qua_params(fr, r_deg = r_deg)
params = cam_dict(ID, fr, cam_qx, cam_qy, cam_qz, cam_qw)

img = get_image(params, verbose=True)