import training
import gc
from showing import ploting
import pickle
from misc import save_obj, filter_keys
from main import MegaFrontEnd


mega = MegaFrontEnd(layers=2, guessing_rule="None")
mega.run_epgreedy1_tables(total_episodes=10**7)
mega.run_phaseflip(total_episodes=10**2, bobs=12)
mega.run_darkcounts(total_episodes=10**2, bobs=12)
