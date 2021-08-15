import cv2

'''util file for misc tasks'''

'''
rescale_frame takes the frame and percent. converts the given frame into smaller size relative to # percent argument.
'''
def rescale_frame(frame_input, percent=75):
    width = int(frame_input.shape[1] * percent / 100)
    height = int(frame_input.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)
