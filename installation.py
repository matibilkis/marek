import training
import gc
import numpy as np
#
amplitudesss = np.arange(.01,1,.01)
for amplitude in amplitudesss:
    a = training.Experiment(layers=1,amplitude=np.round(amplitude,2))
    a.compute_optimal_kenn()

    a = training.Experiment(layers=2,amplitude=np.round(amplitude,2))
    a.compute_optimal_2l()
