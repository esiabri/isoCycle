import numpy as np
import matplotlib.pylab as plt
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
# from alexanderthclark/Matplotlib-for-Storytellers
def color_title(labels, colors, textprops ={'size':'large'}, ax = None, y = 1.013,
               precision = 10**-2):
     
    "Creates a centered title with multiple colors."
        
    if ax == None:
        ax = plt.gca()
        
    plt.gcf().canvas.draw()
    transform = ax.transAxes # use axes coords
    
    # initial params
    xT = 0 # where the text ends in x-axis coords
    shift = 0 # where the text starts
    
    # for text objects
    text = dict()

    while (np.abs(shift - (1-xT)) > precision) and (shift <= xT) :         
        x_pos = shift 
        
        for label, col in zip(labels, colors):

            try:
                text[label].remove()
            except KeyError:
                pass
            
            text[label] = ax.text(x_pos, y, label, 
                        transform = transform, 
                        ha = 'left',
                        color = col,
                        **textprops)
            
            x_pos = text[label].get_window_extent()\
                   .transformed(transform.inverted()).x1
            
        xT = x_pos # where all text ends
        
        shift += precision/2 # increase for next iteration
      
        if x_pos > 1: # guardrail 
            break


def loadFilePath(defaultDataDir,fileExtension="*.dat",fileDesdription="Intan raw files"):
    if not os.path.isdir(defaultDataDir):
        defaultDataDir = "C:\\"
    
    root = Tk()
    root.attributes("-topmost", True)
    root.lift()
    
    root.withdraw()

    dataFileAdd =  askopenfilename(initialdir = defaultDataDir,title = "Select file",\
                                filetypes = ((fileDesdription,fileExtension),("all files","*.*")),parent=root)

    # dataFileName = os.path.basename(dataFileAdd)[:-4]
    dataFileBaseFolder = os.path.dirname(dataFileAdd)
    
    return dataFileAdd, dataFileBaseFolder