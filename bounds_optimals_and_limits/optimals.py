import numpy as np
from training import Experiment


amplitudes = np.arange(.01,1,.01)

for a in amplitudes:
    f = Experiment(amplitude = np.round(a,2))
    f.compute_optimal_2l()
    del f
