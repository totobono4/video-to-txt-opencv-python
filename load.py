import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2 as cv
import numpy as np
from progress.bar import Bar

outpath = './out'
if not os.path.isdir(outpath):
    os.mkdir(outpath)

Tk().withdraw()
filepath = askopenfilename(title='Select a txt...', filetypes=[('txt {txt}')])

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')

with open(filepath, mode='r') as f:
    print('reading file...')
    lines = f.readlines()
    frames_size = lines[:2]
    frames_height = int(frames_size[0])
    frames_width = int(frames_size[1])
    print(frames_height)
    print(frames_width)
    out = cv.VideoWriter(os.path.join(outpath, 'output.avi'), fourcc, 30.0, (frames_width, frames_height))
    frameslines = lines[2:2+(frames_height * 200)]
    frames = [[[] for _ in range(frames_height)] for _ in range(len(frameslines)//frames_height)]

    bar = Bar('Reformatting Frames', max=len(frameslines), suffix='%(percent)d%%')

    for index in range(len(frameslines)):
        frameline = frameslines[index]
        framecount = index//frames_height
        framerow = index%frames_height

        newline = []
        for pixelindex in range(frames_width):
            pixelcolorsize = 2
            pixelsize = pixelcolorsize * 3
            pixelstart = pixelindex * pixelsize

            pixel = [
                int(frameline[pixelstart:pixelstart+pixelcolorsize], base=16),
                int(frameline[pixelstart+pixelcolorsize:pixelstart+(pixelcolorsize*2)], base=16),
                int(frameline[pixelstart+(pixelcolorsize*2):pixelstart+pixelsize], base=16)
            ]

            newline.append(pixel)

        frames[framecount][framerow] = newline
        bar.next()
    bar.finish()

    bar = Bar('Recreating Video', max=len(frames), suffix='%(percent)d%%')

    for frame in frames:
        rgbframe = np.ndarray((frames_height,frames_width,3), dtype=np.uint8)
        for row in range(frames_height):
            for pixel in range(frames_width):
                rgbframe[row][pixel][0] = np.uint8(frame[row][pixel][0])
                rgbframe[row][pixel][1] = np.uint8(frame[row][pixel][1])
                rgbframe[row][pixel][2] = np.uint8(frame[row][pixel][2])
        newframe = cv.cvtColor(rgbframe, cv.COLOR_RGB2BGR)

        out.write(newframe)
        cv.imshow('frame', newframe)
        # Release everything if job is finished
        if cv.waitKey(1) == ord('q'):
            break
        bar.next()
    out.release()
    bar.finish()
cv.destroyAllWindows()
