# Global namespace

import os
import sys
from qtflow import get_flowcontrol
from instruments import get_instruments
from lib import config as _config
#from data import Data # YS: no longer used
#from plot import Plot, plot, plot3, replot_all # YS: no longer used
from scripts import Scripts, Script

config = _config.get_config()

#data = Data.get_named_list() # YS: data no longer used
instruments = get_instruments()
frontpanels = {}
sliders = {}
scripts = Scripts()

flow = get_flowcontrol()
msleep = flow.measurement_idle
mstart = flow.measurement_start
mend = flow.measurement_end

#from plot import Plot2D, Plot3D
#try:
#    from plot import plot_file
#except:
#    pass # YS: plot no longer used

#plots = Plot.get_named_list() # YS: plot no longer used

def version():
    version_file = os.path.join(config['execdir'], 'VERSION')
    try:
        f = file(version_file,'r')
        str = f.readline()
        str = str.rstrip('\n\r')
        f.close()
    except:
        str = 'NO VERSION FILE'
    return str

class qApp:
    '''Class to fix a bug in matplotlib.pyplot back-end detection.'''
    @staticmethod
    def startingUp():
        return True

