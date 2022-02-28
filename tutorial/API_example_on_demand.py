from websocket import create_connection
import io, sys, json, base64
from json import dumps
from PIL import Image
import cv2
import numpy as np

# Create Connection
ws = create_connection("wss://polyhedral.eecs.yorku.ca/api/")
# ws = create_connection("ws://230t.eecs.yorku.ca:8044/api/") # only available to students at YorkU (note: ws vs wss for the protocol)

# Set Parameters
parameter = {
    'ID':'3b4018ba-16e6-48b8-be42-9ecb451d9796',
    'light_fixed':'true',
    'random_cam': 'false',
    'cam_x':-1.2911862/20,
    'cam_y': 4.6562521/20,
    'cam_z': 1.35520790/20,
    'cam_qw':-0.1074521,
    'cam_qx':-0.0814614,
    'cam_qy':0.59861538,
    'cam_qz':0.7896060
}
json_params = dumps(parameter, indent=2)

# Send API request
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
cv2.imshow('image',cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Close Connection
ws.close()