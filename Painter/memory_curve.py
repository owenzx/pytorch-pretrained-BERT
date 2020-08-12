import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import json

RESULTS_JSON_PATH = ""

results_dict = json.load(open(RESULTS_JSON_PATH, 'r'))

# Results is in the format： model_type:
