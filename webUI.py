from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
from utils.nnUtils import sendNN
import cv2


'''file handles the working of code required for live webcam detection.'''

def genCamLabel(win):
    win.camLabel = ttk.Label(win)
    win.camLabel.grid(row=3,column=0)
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    net,classes,layer_names,output_layers = sendNN()
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    def _startFrame():
        
        _temp_frame=cap.read()[1]
        height,width,channel=_temp_frame.shape
        cv2image = cv2.cvtColor(_temp_frame, cv2.COLOR_BGR2RGB)
        
        # cv2image=cv2.flip(cv2image,1)
        blob = cv2.dnn.blobFromImage(cv2image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
                
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                  
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
                print(label)
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(cv2image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                cv2image,
                label + " " + str(round(confidence, 2)),
                (x, y + 30),
                font,
                1,
                color,
                3,
            )           
        img = Image.fromarray(cv2image)
       
        imgtk = ImageTk.PhotoImage(image=img)
        win.camLabel.imgtk = imgtk
        win.camLabel.configure(image=imgtk)
        win.camLabel.after(20, _startFrame)

    _startFrame()
    def _endFrame():
        win.camLabel.destroy()
        cap.release()
        cv2.destroyAllWindows()
    return _endFrame
