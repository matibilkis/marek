import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from training import Experiment
import os
from datetime import datetime
import pickle


#dict is a dictornary with the labels you want to assign
plt.figure(figsize=(16,10)  , dpi=70)
ax1 = plt.subplot2grid((2,1), (0,0))
ax2 = plt.subplot2grid((2,1), (1,0))

exp = Experiment(number_phases=2, layers=2, resolution=0.1, bound_displacements=1, save_tables=True)
exp.load_data("run_1",tables=True)
# run_color = tuple(np.random.randint(256, size=3)/256)
