from libs import *
from utils import *


cam_x = 1/10
cam_y = 1/10
cam_z = 1/10
ID = '3b4018ba-16e6-48b8-be42-9ecb451d9796'
fr = np.array([cam_x, cam_y, cam_z])
r = R.from_rotvec(np.pi * np.array([0, 0, 1.0]))
cam_qx, cam_qy, cam_qz, cam_qw = qua_params(fr, r=r)
params = cam_dict(ID, cam_x, cam_y, cam_z, cam_qx, cam_qy, cam_qz, cam_qw)

img = get_image(params, verbose=True)