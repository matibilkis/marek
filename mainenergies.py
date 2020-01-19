import training
import gc
from showing import ploting
import pickle
from misc import save_obj, filter_keys
from main import MegaFrontEnd
import numpy as np

mega = MegaFrontEnd(layers=2, guessing_rule="None")
mega.energies(total_episodes=5*10**1, bobs=1)
