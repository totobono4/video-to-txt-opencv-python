import os
import cv2 as cv
import numpy as np
from progress.bar import Bar

class Encoding:
    encodings = {
        "BASIC"
    }

    def __init__(self, encoding=''):
        self.encoding = encoding

    def encode(self, inpath, outpath):
        file_name, _ = os.path.splitext(os.path.basename(inpath))
        outfile = os.path.join(outpath, "{}.txt".format(file_name))

        with open(outfile, mode='w') as file:
            match self.encoding:
                case "BASIC":
                    self.basic_encoder(inpath, file)
                case _:
                    self.basic_encoder(inpath, file)

    def decode(self, inpath, outpath):
        file_name, _ = os.path.splitext(os.path.basename(inpath))
        outfile = os.path.join(outpath, "{}.avi".format(file_name))

        with open(inpath, mode='r') as file:
            match self.encoding:
                case "BASIC":
                    self.basic_decoder(file, outfile)
                case _:
                    self.basic_decoder(file, outfile)

    def basic_encoder(self, infile, outfile):
        cap = cv.VideoCapture(infile)

        outfile.write("{:n}\n".format(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        outfile.write("{:n}\n".format(cap.get(cv.CAP_PROP_FRAME_WIDTH)))

        totalframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        bar = Bar('Encoding Video', max=totalframes, suffix='%(percent)d%%')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            RGBframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            for row in RGBframe:
                for pixel in row:
                    outfile.write("{:02X}{:02X}{:02X}".format(pixel[0], pixel[1], pixel[2]))
                outfile.write('\n')

            cv.imshow('frame', frame)

            if cv.waitKey(1) == ord('q'):
                break
            bar.next()
        bar.finish()
        cap.release()
        cv.destroyAllWindows()

    def basic_decoder(self, infile, outfile):
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        print('reading file...')
        lines = infile.readlines()
        frames_size = lines[:2]
        frames_height = int(frames_size[0])
        frames_width = int(frames_size[1])
        out = cv.VideoWriter(outfile, fourcc, 30.0, (frames_width, frames_height))
        frameslines = lines[2:]
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

        bar = Bar('Decoding Video', max=len(frames), suffix='%(percent)d%%')

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

            if cv.waitKey(1) == ord('q'):
                break
            bar.next()

        bar.finish()
        out.release()
        cv.destroyAllWindows()
