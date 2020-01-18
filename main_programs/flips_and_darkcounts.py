import training
import gc
from showing import ploting
import pickle
from misc import save_obj, filter_keys
from main import MegaFrontEnd
import numpy as np

mega = MegaFrontEnd(layers=2, guessing_rule="None")
mega.run_phaseflip(total_episodes=10**5, bobs=1)
mega.run_darkcounts(total_episodes=10**5, bobs=1)
