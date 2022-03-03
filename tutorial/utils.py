from libs import *

def look_at(fr, to, r):
    forward = fr - to
    forward /= np.linalg.norm(forward)
    temp = np.array([0.0, 1.0, 0.0])
    temp /= np.linalg.norm(temp)
    temp = r.apply(temp)
    right = np.cross(temp, forward)
    up = np.cross(forward, right)
    return np.vstack((right, up, forward)).T

def qua_params(fr, to = np.array([0.0, 0.0, 0.0]), r=R.from_rotvec(np.array([0, 0, 0]))):
    # fr = [cam_x, cam_y, cam_z]
    M = look_at(fr, to, r)
    r = R.from_matrix(M)
    cam_qx, cam_qy, cam_qz, cam_qw = r.as_quat()
    return cam_qx, cam_qy, cam_qz, cam_qw

def cam_dict(ID, cam_x, cam_y, cam_z, cam_qx, cam_qy, cam_qz, cam_qw, light_fixed = 'true', random_cam = 'false'):

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

