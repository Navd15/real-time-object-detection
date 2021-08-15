import cv2
import os.path as path

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
