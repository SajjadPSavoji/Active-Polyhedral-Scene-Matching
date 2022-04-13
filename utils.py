from libs import *
PP_TH = 0.5
PP_GT = 0.12
MX_CP = 2
IMG_W = 1080
IMG_H = 1920
DEG_STEP = np.pi/6


def scaled_distance(corner1, corner2, r):
    return np.linalg.norm(corner1 - corner2)/r
def get_graph(corners, r):
    M = np.zeros(shape=(len(corners), len(corners)))
    for i, c1 in enumerate(corners):
        for j, c2 in enumerate(corners):
            M[i, j] = scaled_distance(c1, c2, r)
    return M
    
def find_corners(img, r):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,3,3,0.03)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.005*dst.max(),255,0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(10,10),(-1,-1),criteria)
    mean_corner = np.mean(corners, axis = 0)
    corners = [corners[i] for i in range(len(corners))]
    corners.sort(key=lambda x: scaled_distance(x, mean_corner, r))

    return corners
    
def draw_corners(img, corners, obj_data_path, obj_cam_loc):
    for corner in corners:
        center_coordinates = (int(corner[0]), int(corner[1]))
        radius = 10
        color = (0, 0, 255)
        thickness = 2
        img = cv2.circle(img, center_coordinates, radius, color, thickness)
    save_path = os.path.join(obj_data_path,f"{hash(obj_cam_loc)}_corners.jpg")
    cv2.imwrite(save_path,img)

class CamLoc():
    def __init__(self, cam_r=10, cam_theta=np.pi/2, cam_phi=0, r_deg=0) -> None:
        self.cam_r = cam_r
        self.cam_theta = cam_theta
        self.cam_phi = cam_phi
        self.r_deg = r_deg

    def get_params(self):
        return (self.cam_r, self.cam_theta, self.cam_phi,self.r_deg)

    def __add__(self, cam2_loc) -> None:
        cam2_r, cam2_theta, cam2_phi, r2_deg = cam2_loc.get_params()
        self.cam_r += cam2_r
        self.cam_theta += cam2_theta
        self.cam_phi += cam2_phi
        self.r_deg += r2_deg
    def __hash__(self) -> int:
        return hash(self.get_params())

def observe(obj_id, cam_loc):
    '''
    given camera spherical coordinates and its relative rotations
    returns an obsrevation of the object 
    '''
    cam_r, cam_theta, cam_phi, r_deg = cam_loc.get_params()
    fr = np.array([cam_r, cam_theta, cam_phi])
    cam_qx, cam_qy, cam_qz, cam_qw = qua_params(fr, r_deg = r_deg)
    params = cam_dict(obj_id, fr, cam_qx, cam_qy, cam_qz, cam_qw)
    img = get_image(params)
    return img

def look_at(fr, to, r):
    # assume points are in cartesian coordinate system here
    forward = fr - to
    forward /= np.linalg.norm(forward)
    temp = np.array([0.0, 1.0, 0.0])
    temp /= np.linalg.norm(temp)
    temp = r.apply(temp)
    right = np.cross(temp, forward)
    up = np.cross(forward, right)
    return np.vstack((right, up, forward)).T

def qua_params(fr, to = np.array([0.0, 0.0, 0.0]), r_deg = 0 * np.pi):
    # assume fr and to are in spherical coordinate system for now
    # they should be arranged as np.array[r, th, phi]
    fr = SphericalPt(fr).toCartesian().npy()
    to = SphericalPt(to).toCartesian().npy()
    
    r1 = R.from_rotvec(r_deg*np.array([0.0, 0.0, 1.0]))
    M = look_at(fr, to, r1)
    r = R.from_matrix(M)
    cam_qx, cam_qy, cam_qz, cam_qw = r.as_quat()
    return cam_qx, cam_qy, cam_qz, cam_qw

def cam_dict(ID, fr, cam_qx, cam_qy, cam_qz, cam_qw, light_fixed = 'true', random_cam = 'false'):

    cam_cartesian = SphericalPt(fr).toCartesian().npy()
    cam_x = cam_cartesian[0]
    cam_y = cam_cartesian[1]
    cam_z = cam_cartesian[2]

    parameter = {
        'ID': ID,
        'light_fixed':light_fixed,
        'random_cam': random_cam,
        'cam_x': cam_x,
        'cam_y': cam_y,
        'cam_z': cam_z,
        'cam_qw':cam_qw,
        'cam_qx':cam_qx,
        'cam_qy':cam_qy,
        'cam_qz':cam_qz,
    }
    return parameter

    

def get_image(cam_dict, verbose=False):
    json_params = dumps(cam_dict, indent=2)
    print(json_params)
    # Send API request
    ws = create_connection("wss://polyhedral.eecs.yorku.ca/api/")
    ws.send(json_params)

    # Wait patiently while checking status
    last_status = None
    while True:
        result = json.loads(ws.recv())
        if not last_status == result['status']:
            print("Job Status: {0}".format(result['status']))
            last_status = result['status']
        if result['status'] == "SUCCESS":
            break
        elif "FAILURE" in result['status'] or "INVALID" in result['status']:
            sys.exit()

    # Processing result
    image_base64 = result['image']
    image_decoded = base64.b64decode(str(image_base64))

    # Create Open CV 2 Image
    image = Image.open(io.BytesIO(image_decoded))
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    if verbose:
        cv2.imshow('image',cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Close Connection
    ws.close()
    return cv_image

def pp(img, pp_th, ax, side):
        '''
        compute the pading percent of image
        '''
        if side == "left":
            step = +1
        else:
            step = -1

        temp = np.max(img, axis=ax)
        temp  = temp / np.max(temp)
        count = 0
        for x in temp[::step]:
            if x < pp_th:
                count += 1
            else: break
        pp = count/len(temp)
        return pp

def image_pp(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pps = []
    axs = [0, 1]
    sides = ["left", "right"]
    for ax in axs:
        for side in sides:
            pps.append(pp(img, PP_TH, ax, side))
    return np.min(pps)

def adjust_to_fill(img, cam_loc) -> CamLoc:
    pp = image_pp(img)
    rx = PP_GT/pp
    cam_loc.cam_r *= rx
    return cam_loc


class Pt(object):
    def __init__(self, coordinate):
        self.x = coordinate[0]
        self.y = coordinate[1]
        self.z = coordinate[2]
        
    def __str__(self):
        return '(%0.4f, %0.4f, %0.4f)' % (self.x, self.y, self.z)

    def __repr__(self):
        return 'Pt(%f, %f, %f)' % (self.x, self.y, self.z)

    def __add__(self, other):
        return Pt(self.x+other.x, self.y+other.y, self.z+other.z)

    def __sub__(self, other):
        return Pt(self.x-other.x, self.y-other.y, self.z-other.z)
    
    def __mul__(self, f):
        return Pt(self.x*f, self.y*f, self.z*f)

    def dist(self, other):
        p = self-other
        return (p.x**2 + p.y**2 + p.z**2)**0.5

    def toSpherical(self):
        r = self.dist(Pt(0, 0, 0))
        theta = np.atan2(np.sqrt(self.x**2+self.y**2), self.z)
        phi = np.atan2(self.y, self.x)
        return SphericalPt(np.array([r, theta, phi]))

    def npy(self):
        return np.array([self.x, self.y, self.z])

class SphericalPt(object):
    def __init__(self, coordinate):
        # radial coordinate, zenith angle, azimuth angle
        self.r = coordinate[0]
        self.theta = coordinate[1]
        self.phi = coordinate[2]

    def __str__(self):
        return '(%0.4f, %0.4f, %0.4f)' % (self.r, self.theta, self.phi)

    def __repr__(self):
        return 'SphericalPt(%f, %f, %f)' % (self.r, self.theta, self.phi)

    def toCartesian(self):
        x = self.r*np.cos(self.phi)*np.sin(self.theta)
        y = self.r*np.sin(self.phi)*np.sin(self.theta)
        z = self.r*np.cos(self.theta)
        return Pt(np.array([x,y,z]))

    def npy(self):
        return np.array([self.r, self.theta, self.phi])
