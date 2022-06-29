import torch, os, pprint, copy
import pandas as pd
import time
import json
from torch import nn

import numpy as np

from haven import haven_utils as hu
from haven import haven_utils as hu
from haven import haven_wizard as hw

import argparse
import os
from src import models
#from src import utils as ut
from src import datasets
import exp_configs

from trainval import trainval as trainval_actual
    
#def trainval(exp_dict, savedir, args):
    
def trainval(exp_dict, savedir, args):
    # exp_dict here should just be a single
    # key showing where the experiment is
    # located.
    
    exp_dir = exp_dict['exp_dir']
    actual_exp_dict = json.loads(
        open("{}/exp_dict.json".format(exp_dir)).read()
    )
    savedir = exp_dir
        
    return trainval_actual(actual_exp_dict, savedir, args)