from tokenize import Special
from libs import *
from utils import *

cam_r = 1
cam_theta = 0 * np.pi
cam_phi = np.pi/2
r_deg = 0 * np.pi


ID = 'eval_529cfdef-0e35-483f-8b47-53fb614c70f5'
fr = np.array([cam_r, cam_theta, cam_phi])
cam_qx, cam_qy, cam_qz, cam_qw = qua_params(fr, r_deg = r_deg)
params = cam_dict(ID, fr, cam_qx, cam_qy, cam_qz, cam_qw)

img = get_image(params, verbose=True)


