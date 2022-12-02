import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from encoding import Encoding

outpath = os.path.abspath('./out')
if not os.path.isdir(outpath):
    os.mkdir(outpath)

Tk().withdraw()
inpath = askopenfilename(title='Select a txt...', filetypes=[('txt {txt}')])

Encoding('REPETITION').decode(inpath, outpath)
