from tkinter import filedialog as fd, ttk
from utils.nnUtils import sendNN
import cv2
import numpy as np


def genImageLabel(win):
    def _selectVideo():
        vidSelect = fd.askopenfilename(
            title="Select video file", filetypes=[("mp4 files", ".mp4")]
        )
        if vidSelect:
            global cap
            cap = cv2.VideoCapture(vidSelect)
            width = int(cap.get(3))
            height = int(cap.get(4))
            net, classes, layer_names, output_layers = sendNN()
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            out_vid = cv2.VideoWriter(
                f"annotated_video.avi",
                cv2.VideoWriter_fourcc(*"XVID"),
                10,
                (width, height)
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            while cap.isOpened():
                ret, _temp_frame = cap.read()
                print(ret)
                if ret:
                    cv2image = cv2.cvtColor(_temp_frame, cv2.COLOR_BGR2RGB)
                    
                    blob = cv2.dnn.blobFromImage(
                        cv2image, 0.00392, (416, 416), (0, 0, 0), True, crop=False
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
                    out_vid.write(cv2image)
            cap.release()
            out_vid.release()
    
    win.button = ttk.Button(win, text="Select video file", command=_selectVideo)
    win.button.grid(row=3, column=0)

    def _destroy():
        win.button.destroy()

    return _destroy
