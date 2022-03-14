from libs import *

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
    # Send API request
    ws = create_connection("wss://polyhedral.eecs.yorku.ca/api/")
    ws.send(json_params)

    # Wait patiently while checking status
    while True:
        result = json.loads(ws.recv())
        print("Job Status: {0}".format(result['status']))
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
    return image

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
