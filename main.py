import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2 as cv

outpath = './out'
if not os.path.isdir(outpath):
    os.mkdir(outpath)

Tk().withdraw()
filepath = askopenfilename(title='Select a video...', filetypes=[('mp4 {mp4}')])

cap = cv.VideoCapture(filepath)

file_name, file_extension = os.path.splitext(os.path.basename(filepath))
videoSize = "{:n}x{:n}".format(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))
outfile = os.path.join(outpath, "{}({}).txt".format(file_name, videoSize))

with open(outfile, mode='w') as f:
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        RGBframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        cv.imshow('frame', RGBframe)

        for row in RGBframe:
            for pixel in row:
                f.write("{:02X}{:02X}{:02X}".format(pixel[0], pixel[1], pixel[2]))
            f.write('\n')

        if cv.waitKey(1) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
