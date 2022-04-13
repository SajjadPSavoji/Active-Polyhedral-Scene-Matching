from utils import *
from libs import *
import argparse

def validate_theta():
    pass
def validate_phi():
    pass
    

def get_next_cam_loc(obj_cam_loc, far_corner):
    if far_corner[0] > IMG_H*0.6:
        if far_corner[1]< IMG_W*0.3:
            # decreace phi
            # increace theta
            obj_cam_loc.cam_phi -= DEG_STEP
            obj_cam_loc.cam_theta += DEG_STEP

        elif far_corner[1]> IMG_W*0.6:
            # increace phi
            # decrence theta
            obj_cam_loc.cam_phi += DEG_STEP
            obj_cam_loc.cam_theta -= DEG_STEP

        else:
            # do not change phi
            # increace theta
            obj_cam_loc.cam_phi += 0
            obj_cam_loc.cam_theta += DEG_STEP

    elif far_corner[0]< IMG_H*0.3:
        if far_corner[1]< IMG_W*0.3:
            # decreace phi
            # decrease theta
            obj_cam_loc.cam_phi -= DEG_STEP
            obj_cam_loc.cam_theta -= DEG_STEP

        elif far_corner[1]> IMG_W*0.6:
            # increace phi
            # decrease theta
            obj_cam_loc.cam_phi += DEG_STEP
            obj_cam_loc.cam_theta -= DEG_STEP
        else:
            # do not change phi
            # decrease theta
            obj_cam_loc.cam_phi += 0
            obj_cam_loc.cam_theta -= DEG_STEP
    else:
        if far_corner[1]< IMG_W*0.3:
            # decreace phi
            # do not change theta
            obj_cam_loc.cam_phi -= DEG_STEP
            obj_cam_loc.cam_theta -= 0

        elif far_corner[1]> IMG_W*0.6:
            # increace phi
            # do not change theta
            obj_cam_loc.cam_phi += DEG_STEP
            obj_cam_loc.cam_theta -= 0

        else:
            # do not change phi
            # do not change theta
            obj_cam_loc.cam_phi -= 0
            obj_cam_loc.cam_theta -= 0

    obj_cam_loc.cam_r = 10
    return obj_cam_loc

def capture_plicy(obj1_corners, obj2_corners, obj1_cam_loc, obj2_cam_loc):
    if len(obj1_corners.keys()) < 1:
        nxt_obj1_cam_loc = obj1_cam_loc 
    else:
        corners_1 = obj1_corners[obj1_cam_loc]
        far_corner_1 = corners_1[-1]
        nxt_obj1_cam_loc = get_next_cam_loc(obj1_cam_loc, far_corner_1)

    if len(obj2_corners.keys()) < 1: 
        nxt_obj2_cam_loc  = obj2_cam_loc
    else:
        corners_2 = obj2_corners[obj2_cam_loc]
        far_corner_2 = corners_2[-1]
        nxt_obj2_cam_loc = get_next_cam_loc(obj2_cam_loc, far_corner_2)
        
    return nxt_obj1_cam_loc, nxt_obj2_cam_loc  

def observe_scaled_images(obj1_data, obj2_data, obj1_cam_loc, obj2_cam_loc, obj1_data_path, obj2_data_path, obj1_id, obj2_id):
    img1 = observe(obj1_id, obj1_cam_loc)
    img2 = observe(obj2_id, obj2_cam_loc)
    obj1_cam_loc = adjust_to_fill(img1, obj1_cam_loc)
    obj2_cam_loc = adjust_to_fill(img2, obj2_cam_loc)
    
    obj1_data[obj1_cam_loc] = observe(obj1_id, obj1_cam_loc)
    obj2_data[obj2_cam_loc] = observe(obj2_id, obj2_cam_loc)

    # save image and location detail
    with open(os.path.join(obj1_data_path,f"{hash(obj1_cam_loc)}.camloc"), "wb") as f:
        pkl.dump(obj1_cam_loc, f)
    with open(os.path.join(obj2_data_path,f"{hash(obj2_cam_loc)}.camloc"), "wb") as f:
        pkl.dump(obj2_cam_loc, f)

    cv2.imwrite(os.path.join(obj1_data_path,f"{hash(obj1_cam_loc)}.jpg"), obj1_data[obj1_cam_loc])
    cv2.imwrite(os.path.join(obj2_data_path,f"{hash(obj2_cam_loc)}.jpg"), obj2_data[obj2_cam_loc])



def compare_2objects(obj1_id, obj2_id, data_path) -> str:
    # set up directories for each object
    obj1_data_path = os.path.join(data_path, obj1_id)
    obj2_data_path = os.path.join(data_path, obj2_id)

    if not obj1_id in os.listdir(data_path):
        os.system(f"mkdir {obj1_data_path}")
    if not obj2_id in os.listdir(data_path):
        os.system(f"mkdir {obj2_data_path}")

    #initialized CamLoc and Data containers
    obj1_cam_loc, obj2_cam_loc = CamLoc(), CamLoc()
    obj1_data, obj2_data = {}, {}
    obj1_corners, obj2_corners = {}, {} 
    obj1_graphs, obj2_graphs = {}, {}

    for _ in range(MX_CP):
        obj1_cam_loc, obj2_cam_loc = capture_plicy(obj1_corners, obj2_corners, obj1_cam_loc, obj2_cam_loc)
        observe_scaled_images(obj1_data, obj2_data, obj1_cam_loc, obj2_cam_loc, obj1_data_path, obj2_data_path, obj1_id, obj2_id)
        corners_1 = find_corners(obj1_data[obj1_cam_loc], obj1_cam_loc.cam_r)
        corners_2 = find_corners(obj2_data[obj2_cam_loc], obj2_cam_loc.cam_r)
        obj1_corners[obj1_cam_loc] = corners_1
        obj2_corners[obj2_cam_loc] = corners_2
        draw_corners(obj1_data[obj1_cam_loc], corners_1, obj1_data_path, obj1_cam_loc)
        draw_corners(obj2_data[obj2_cam_loc], corners_2, obj2_data_path, obj2_cam_loc)
        graph_1 = get_graph(corners_1, obj1_cam_loc.cam_r)
        graph_2 = get_graph(corners_2, obj2_cam_loc.cam_r)
        obj1_graphs[obj1_cam_loc] = graph_1
        obj2_graphs[obj2_cam_loc] = graph_2 
    
        
        # sim = similar(obj1, obj2)
        # if sim == "S":
        # elif sim 

            
            


    
    # do your comparision in an active fashion 
    # get images from both objects 
    # extract features
    # decide if they are the same or not
    # if decided -> terminate and return "S" or "D"
    # else: get more images and more features
    # terminate after a fized number of views
    pass


def compare_objs_dataset(csv_path, data_path = "Data"):
    df = pd.read_csv(csv_path)
    for i, row in tqdm(df.iterrows()):
        obj1_id, obj2_id = row["Object_1"], row["Object_2"]
        # make folders to store imgs and features
        # image name will be a hash of object id and camera config
        # so the system will not require many API calls
        
        row["Answer"] = compare_2objects(obj1_id, obj2_id, data_path)

def main(args):
    compare_objs_dataset(args.TestSetPath, args.DataPath)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--TestSetPath', type=str, required=True)
    my_parser.add_argument('--DataPath', type=str, required=True)
    args = my_parser.parse_args()

    main(args)
    