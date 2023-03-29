import os
import cv2 as cv
import numpy as np
from progress.bar import Bar
import binascii

class Encoding:
    encodings = {
        "BASIC",
        "REPETITION",
        "RLE16",
        "RLE24"
    }

    def __init__(self, encoding=''):
        self.encoding = encoding

    def encode(self, inpath, outpath):
        file_name, _ = os.path.splitext(os.path.basename(inpath))
        outfile = os.path.join(outpath, "{}.txt".format(file_name))

        match self.encoding:
            case "BASIC":
                self.basic_encoder(inpath, outfile)
            case "REPETITION":
                self.repetition_encoder(inpath, outfile)
            case "RLE16":
                self.RLE16_encoder(inpath, outfile)
            case "RLE16G":
                self.RLE16G_encoder(inpath, outfile)
            case "RLE24":
                self.RLE24_encoder(inpath, outfile)
            case _:
                self.basic_encoder(inpath, outfile)

    def decode(self, inpath, outpath):
        file_name, _ = os.path.splitext(os.path.basename(inpath))
        outfile = os.path.join(outpath, "{}.avi".format(file_name))

        match self.encoding:
            case "BASIC":
                self.basic_decoder(inpath, outfile)
            case "REPETITION":
                self.repetition_decoder(inpath, outfile)
            case "RLE":
                self.RLE_decoder(inpath, outfile)
            case _:
                self.basic_decoder(inpath, outfile)

    def basic_encoder(self, infile, outfile):
        with open(outfile, mode='w') as file:
            cap = cv.VideoCapture(infile)

            file.write("{:n}\n".format(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
            file.write("{:n}\n".format(cap.get(cv.CAP_PROP_FRAME_WIDTH)))

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
                        file.write("{:02X}{:02X}{:02X}".format(pixel[0], pixel[1], pixel[2]))
                    file.write('\n')

                cv.imshow('frame', frame)

                if cv.waitKey(1) == ord('q'):
                    break
                bar.next()
            bar.finish()
            cap.release()
            cv.destroyAllWindows()

    def basic_decoder(self, infile, outfile):
        with open(infile, mode='r') as file:
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            print('reading file...')
            lines = file.readlines()
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

    def repetition_encoder(self, infile, outfile):
        with open(outfile, mode='w') as file:
            cap = cv.VideoCapture(infile)

            file.write("{:n}\n".format(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
            file.write("{:n}\n".format(cap.get(cv.CAP_PROP_FRAME_WIDTH)))

            totalframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            bar = Bar('Encoding Video', max=totalframes, suffix='%(percent)d%%')

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                RGBframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                repetition = 1
                last_pixel = ''

                for row in RGBframe:
                    for pixel in row:
                        current_pixel = "{:02X}{:02X}{:02X}".format(pixel[0], pixel[1], pixel[2])
                        if last_pixel == '':
                            last_pixel = current_pixel
                            continue
                        if current_pixel == last_pixel:
                            repetition = repetition+1
                        else:
                            file.write("{},{};".format(repetition, last_pixel))
                            repetition = 1
                            last_pixel = current_pixel

                file.write("{},{}".format(repetition, last_pixel))
                file.write('\n')
                cv.imshow('frame', frame)

                if cv.waitKey(1) == ord('q'):
                    break
                bar.next()
            bar.finish()
            cap.release()
            cv.destroyAllWindows()

    def repetition_decoder(self, infile, outfile):
        with open(infile, mode='r') as file:
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            print('reading file...')
            lines = file.readlines()
            frames_size = lines[:2]
            frames_height = int(frames_size[0])
            frames_width = int(frames_size[1])
            out = cv.VideoWriter(outfile, fourcc, 30.0, (frames_width, frames_height))
            frameslines = lines[2:]

            bar = Bar('Decoding Video', max=len(frameslines), suffix='%(percent)d%%')

            for index in range(len(frameslines)):
                frame = np.ndarray((frames_height,frames_width,3), dtype=np.uint8)
                frameline = frameslines[index]
                repetitions_format = frameline.split(';')
                pixels = []

                for repetition_format in repetitions_format:
                    repetition, pixel_format = repetition_format.split(',')
                    for _ in range(int(repetition)):
                        pixels.append(pixel_format[:6])

                for pixelindex in range(len(pixels)):
                    frame[pixelindex//frames_width][pixelindex%frames_width] = [
                        int(pixels[pixelindex][:2], base=16),
                        int(pixels[pixelindex][2:4], base=16),
                        int(pixels[pixelindex][4:], base=16)
                    ]

                newframe = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                out.write(newframe)
                cv.imshow('frame', newframe)

                if cv.waitKey(1) == ord('q'):
                    break
                bar.next()
            bar.finish()
            out.release()
            cv.destroyAllWindows()

    def RLE16_encoder(self, infile, outfile):
        with open(outfile, mode='wb') as file:
            cap = cv.VideoCapture(infile)

            file.write(binascii.unhexlify("{:04X}".format(int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))))
            file.write(binascii.unhexlify("{:04X}".format(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))))

            totalframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            bar = Bar('Encoding Video', max=totalframes, suffix='%(percent)d%%')

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                RGBframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # La donnée sera de 16 bits pour la répétition et 24 bits pour la couleur.
                # De 0x0000 à 0x8000, les non-répétitions et de 0x8000 à 0xFFFF, les répétitions.
                rle_offset = int("8000", base=16)

                difference = -1
                last_sequence = []

                repetition = 0
                last_pixel = ''

                for row in RGBframe:
                    for pixel in row:
                        current_pixel = "{:02X}{:02X}{:02X}".format(pixel[0], pixel[1], pixel[2])
                        if last_pixel == '':
                            last_pixel = current_pixel
                            continue
                        if current_pixel == last_pixel: # Si le pixel est répété
                            if difference == 0:
                                difference = -1
                            if difference > 0: # Si il y avait une chaîne de pixels différents
                                difference = difference-1
                                last_sequence.pop()
                                file.write(binascii.unhexlify("{:04X}".format(difference)))
                                for last_diff in last_sequence:
                                    file.write(binascii.unhexlify("{}".format(last_diff)))
                                last_sequence = []
                                difference = -1
                            if repetition == rle_offset-1: # Si on atteint la limite de chaîne de répétition
                                file.write(binascii.unhexlify("{:04X}{}".format(rle_offset+repetition, last_pixel)))
                                repetition = 0
                            repetition = repetition+1
                        else: # Si le pixel est différent
                            if repetition > 0: # Si il y avait une chaîne de pixels répétés
                                file.write(binascii.unhexlify("{:04X}{}".format(rle_offset+repetition, last_pixel)))
                                repetition = 0
                                last_sequence = []
                            if difference == rle_offset-1: # Si on atteint la limite de chaîne de différence
                                file.write(binascii.unhexlify("{:04X}".format(difference)))
                                for last_diff in last_sequence:
                                    file.write(binascii.unhexlify("{}".format(last_diff)))
                                last_sequence = []
                                difference = -1
                            difference = difference+1
                            last_sequence.append(current_pixel)
                        last_pixel = current_pixel

                cv.imshow('frame', frame)

                if cv.waitKey(1) == ord('q'):
                    break
                bar.next()
            bar.finish()
            cap.release()
            cv.destroyAllWindows()

    def RLE16_decoder(self, infile, outfile):
        with open(infile, mode='rb') as file:
            pass

    def RLE16G_encoder(self, infile, outfile):
        with open(outfile, mode='wb') as file:
            cap = cv.VideoCapture(infile)

            file.write(binascii.unhexlify("{:04X}".format(int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))))
            file.write(binascii.unhexlify("{:04X}".format(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))))

            totalframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            bar = Bar('Encoding Video', max=totalframes, suffix='%(percent)d%%')

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                RGBframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # La donnée sera de 16 bits pour la répétition et 24 bits pour la couleur.
                # De 0x0000 à 0x8000, les non-répétitions et de 0x8000 à 0xFFFF, les répétitions.
                rle_offset = int("8000", base=16)

                difference = -1
                last_sequence = []

                repetition = 0
                last_pixel = ''

                for row in RGBframe:
                    for pixel in row:
                        current_pixel = "{:02X}".format(pixel[0])
                        if last_pixel == '':
                            last_pixel = current_pixel
                            continue
                        if current_pixel == last_pixel: # Si le pixel est répété
                            if difference == 0:
                                difference = -1
                            if difference > 0: # Si il y avait une chaîne de pixels différents
                                difference = difference-1
                                last_sequence.pop()
                                file.write(binascii.unhexlify("{:04X}".format(difference)))
                                for last_diff in last_sequence:
                                    file.write(binascii.unhexlify("{}".format(last_diff)))
                                last_sequence = []
                                difference = -1
                            if repetition == rle_offset-1: # Si on atteint la limite de chaîne de répétition
                                file.write(binascii.unhexlify("{:04X}{}".format(rle_offset+repetition, last_pixel)))
                                repetition = 0
                            repetition = repetition+1
                        else: # Si le pixel est différent
                            if repetition > 0: # Si il y avait une chaîne de pixels répétés
                                file.write(binascii.unhexlify("{:04X}{}".format(rle_offset+repetition, last_pixel)))
                                repetition = 0
                                last_sequence = []
                            if difference == rle_offset-1: # Si on atteint la limite de chaîne de différence
                                file.write(binascii.unhexlify("{:04X}".format(difference)))
                                for last_diff in last_sequence:
                                    file.write(binascii.unhexlify("{}".format(last_diff)))
                                last_sequence = []
                                difference = -1
                            difference = difference+1
                            last_sequence.append(current_pixel)
                        last_pixel = current_pixel

                cv.imshow('frame', frame)

                if cv.waitKey(1) == ord('q'):
                    break
                bar.next()
            bar.finish()
            cap.release()
            cv.destroyAllWindows()

    def RLE16G_decoder(self, infile, outfile):
        with open(infile, mode='rb') as file:
            pass

    def RLE24_encoder(self, infile, outfile):
        with open(outfile, mode='wb') as file:
            cap = cv.VideoCapture(infile)

            file.write(binascii.unhexlify("{:04X}".format(int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))))
            file.write(binascii.unhexlify("{:04X}".format(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))))

            totalframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            bar = Bar('Encoding Video', max=totalframes, suffix='%(percent)d%%')

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                RGBframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # La donnée sera de 24 bits pour la répétition et 24 bits pour la couleur.
                # De 0x0000 à 0x800000, les non-répétitions et de 0x800000 à 0xFFFFFF, les répétitions.
                rle_offset = int("800000", base=16)

                difference = -1
                last_sequence = []

                repetition = 0
                last_pixel = ''

                for row in RGBframe:
                    for pixel in row:
                        current_pixel = "{:02X}{:02X}{:02X}".format(pixel[0], pixel[1], pixel[2])
                        if last_pixel == '':
                            last_pixel = current_pixel
                            continue
                        if current_pixel == last_pixel: # Si le pixel est répété
                            if difference == 0:
                                difference = -1
                            if difference > 0: # Si il y avait une chaîne de pixels différents
                                difference = difference-1
                                last_sequence.pop()
                                file.write(binascii.unhexlify("{:06X}".format(difference)))
                                for last_diff in last_sequence:
                                    file.write(binascii.unhexlify("{}".format(last_diff)))
                                last_sequence = []
                                difference = -1
                            if repetition == rle_offset-1: # Si on atteint la limite de chaîne de répétition
                                file.write(binascii.unhexlify("{:06X}{}".format(rle_offset+repetition, last_pixel)))
                                repetition = 0
                            repetition = repetition+1
                        else: # Si le pixel est différent
                            if repetition > 0: # Si il y avait une chaîne de pixels répétés
                                file.write(binascii.unhexlify("{:06X}{}".format(rle_offset+repetition, last_pixel)))
                                repetition = 0
                                last_sequence = []
                            if difference == rle_offset-1: # Si on atteint la limite de chaîne de différence
                                file.write(binascii.unhexlify("{:06X}".format(difference)))
                                for last_diff in last_sequence:
                                    file.write(binascii.unhexlify("{}".format(last_diff)))
                                last_sequence = []
                                difference = -1
                            difference = difference+1
                            last_sequence.append(current_pixel)
                        last_pixel = current_pixel

                cv.imshow('frame', frame)

                if cv.waitKey(1) == ord('q'):
                    break
                bar.next()
            bar.finish()
            cap.release()
            cv.destroyAllWindows()

    def RLE24_decoder(self, infile, outfile):
        with open(infile, mode='rb') as file:
            pass
