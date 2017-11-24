import matplotlib as mpl
import numpy as np
from sys import exc_info
import os 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imp


module_dir = os.path.dirname(__file__)


def load_matplotlib_style(name="default"):
    # loading customized matplotlib style. If not available, it does nothing
    try:
        if name == "default":
            mpl.rcParams = mpl.rc_params_from_file(module_dir + "/matplotlibrc")
        imp.reload(plt)
        #reload(gridspec)
    except:
        print(exc_info())
        pass


def plot_image_and_proj(image, title="", **kwargs):
    """Plots an image and the projections (sums) of it on the x, y axes. 
    Parameters
    ----------
    image: 2-d array
           the image to be plotted
    title: string
           optional title for the plot
    """
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1], height_ratios=[0.2, 3, 1]) 
    ax0 = plt.subplot(gs[1,0])
    plt.title(title)
    ims = plt.imshow(image, aspect="auto", **kwargs)
    
    ax2 = plt.subplot(gs[2,0], sharex=ax0, )
    plt.plot(image.sum(axis=0))
    plt.subplot(gs[1,1], sharey=ax0)
    plt.plot(image.sum(axis=1), list(range(len(image.sum(axis=1)))))
    ax = plt.subplot(gs[0,0])
    plt.colorbar(ims, orientation="horizontal", cax=ax)
    fig.show()


def rebin(a, *args):
    """rebins a numpy array.
    Example: 
    a = np.arange(20).reshape(10, 2)
    b = rebin(a, 5, 2)
    Parameters
    ----------
    a: numpy array
       array to be rebinned
    *args: integers
       the number of bins desidered, for each dimension
    Returns
    -------
    result: numpy array
       the rebinned array
    """
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape) / np.asarray(args)
    #print factor
    evList = ['a.reshape('] + ['args[%d],factor[%d],' % (i, i) for i in range(lenShape)] + [')'] + ['.mean(%d)' % (i + 1) for i in range(lenShape)]
    return eval(''.join(evList))
