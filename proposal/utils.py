import cv2
import matplotlib.pyplot as plt
import glob


def list_files(DIR='/',EXT='*'):
    '''List all files in a given directory DIR and its subfolders. It is 
    possible to add a specific extension EXT (e.g JPG, TIFF). '''
    # check inputs
    if DIR[-1]!='/':
        DIR += '/'
    # list images
    filepath = []
    for filename in glob.glob(DIR + '**/*.' + EXT, recursive=True):
        filepath.append(filename)
    return filepath

def imshow(img, title='image', displaymode = 'matplotlib'):
    '''Display a given image 'img'.'''
    if displaymode == 'matplotlib':
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=plt.cm.gray)
        ax.set_title(title)
        ax.get_xaxis().set_visible(False) # Disable x-axis
        ax.get_yaxis().set_visible(False) # Disable y-axis
    else:
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() # press ENTER on the figure to close it
