import numpy as np
from PIL import Image

#calculate the intersection over union 
#box format : [x_left,y_top, width, height]
def get_iou(a, b):
    xmin = min(a[0],b[0])
    xmax = max(a[2]+a[0],b[2]+b[0])
    ymin = min(a[1]-a[3], b[1]-b[3])
    ymax = max(a[1],b[1]) 
    union = max(0, xmax - xmin) * max(0, ymax - ymin)
    return (a[2]*a[3] + b[2]*b[3] - union)/union

#currently unimplemented
#desgined to parse a annotation file and return a list of bounding boxes in the format [x_left,y_top, width, height]
def get_bounding_boxes(annotation):
    return []

#currently unimplemented 
#loads the dataset in the form of : images,[scores,scores_and_coordinates]
#returns randomly generated data, for the purpose of testing the training loop.
def dataloader(path_to_dataset = ''):
    return np.float32(np.random.random((64, 128, 128, 3))), [np.float32(np.random.random((64,320,1))),np.float32(np.random.random((64,320,5)))]

#For the purpose of simplicity, currently we have 1 anchor box (aspect ratio 1:1) for each point in the 8*8 and 16*16 feaature map.
#maps the anchors to the original image (128*128*3).
#will be called by dataloader to provide the target scores and coordinates for each anchor. 
def map_anchors(path_to_annotations):
    annotations = os.listdir(path_to_annotations)
    batch_score_8 = []
    batch_loc_8 = []
    batch_score_16 = []
    batch_loc_16 =  []
    for annotation in annotations:  
        #boxes : [x_left,y_top, width, height]
        boxes = get_bounding_boxes(annotation)
        scores_16 = []
        loc_16 = []
        threshold = 0.5
        for x in range(0,128,8):
            for y in range(0,128,8):
                iou = 0
                box = []
                for b in boxes:
                    overlap = get_iou(b,[x,y,8,8])
                    if iou < overlap:
                        iou = overlap
                        box = b
                if iou > threshold:
                    scores_16.append(1)
                    loc_16.append([1, (box[0]+box[2]/2)/128,(box[1]-box[3]/2)/128,box[2]/128,box[3]/128])
                else:
                    scores_16.append(0)
                    loc_16.append([0,0,0,0,0])

        scores_8 = []
        loc_8 = []
        for x in range(0,128,16):
            for y in range(0,128,16):
                iou = 0
                box = []
                for b in boxes:
                    overlap = get_iou(b,[x,y,16,16])
                    if iou < overlap:
                        iou = overlap
                        box = b
                if iou > threshold:
                    scores_8.append(1)
                    loc_8.append([1, (box[0]+box[2]/2)/128,(box[1]-box[3]/2)/128,box[2]/128,box[3]/128])
                else:
                    scores_8.append(0)
                    loc_8.append([0,0,0,0,0]) 
        batch_loc_8.append(loc_8)
        batch_score_8.append(scores_8)
        batch_score_16.append(scores_16)
        batch_loc_16.append(loc_16)     

    # print(np.array(batch_score_8).shape, np.array(batch_loc_8).shape)    
    # return np.array(batch_score_8), np.array(batch_loc_8), np.array(batch_score_16), np.array(batch_loc_16)
    # print(np.concatenate((np.array(batch_score_16),np.array(batch_score_8)),axis=1).shape, np.concatenate((np.array(batch_loc_16),np.array(batch_loc_8)),axis=1).shape)

    return np.concatenate((np.array(batch_score_16),np.array(batch_score_8)),axis=1), np.concatenate((np.array(batch_loc_16),np.array(batch_loc_8)),axis=1)