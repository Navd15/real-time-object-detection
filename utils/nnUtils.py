import cv2
import os.path as path
import numpy as np


'''
utils for neural network like searching for a weights and config file. 
_internal is the wrapper over cv2.dnn process, it is called by the videoUI.py file after we have selected the video file. It takes each frame from the video source and sends it to for the forward pass to the YOLOv3; gets in return the boxes and class confidences  for various labels.

'''


two_up =  path.abspath(path.join(__file__ ,"../.."))


def sendNN(
    weight="weights/yolov3.weights", cfg="configuration/yolov3.cfg"):
    print(two_up)
    net = cv2.dnn.readNet(f'{two_up}/{weight}',f'{two_up}/{cfg}')
    classes = []
    label_names = f"{two_up}/configuration/coco.names"
    with open(label_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return (net, classes, layer_names, output_layers)


def _internal(cap,net,output_layers,classes,out_vid):
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            width = int(cap.get(3))
            height = int(cap.get(4))
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            while cap.isOpened():
                ret, _temp_frame = cap.read()
                print(ret)
                if ret:
                    # _temp_frame=rescale_frame(_temp_frame)
                    cv2image = cv2.cvtColor(_temp_frame, cv2.COLOR_BGR2RGB)

                    blob = cv2.dnn.blobFromImage(
                        cv2image, 1/255.0, (416, 416), (0, 0, 0), True, crop=False
                    )
                    net.setInput(blob)
                    outs = net.forward(output_layers)

                    class_ids = []
                    confidences = []
                    boxes = []
                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)

                            confidence = scores[class_id]
                            if confidence > 0.1:
                                # Object detected
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)

                                # Rectangle coordinates
                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)

                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)
                            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
                            for i in range(len(boxes)):
                                if i in indexes:
                                    x, y, w, h = boxes[i]
                                    label = str(classes[class_ids[i]])
                                    
                                    confidence = confidences[i]
                                    color = colors[class_ids[i]]
                                    cv2.rectangle(
                                        cv2image, (x, y), (x + w, y + h), color, 2
                                    )
                                    cv2.putText(
                                        cv2image,
                                        label + " " + str(round(confidence, 2)),
                                        (x, y + 30),
                                        font,
                                        1,
                                        color,
                                        3,
                                    )
                    out_vid.write(cv2image)
            return