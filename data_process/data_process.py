import random

from utils import sparse_to_adjlist
from scipy.io import loadmat
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
"""
	Read data and save the adjacency matrices to adjacency lists
"""

# this code is based on https://github.com/YingtongDou/CARE-GNN/blob/master/data_process.py


if __name__ == "__main__":
	prefix = './data/amazon/'

	# yelp = loadmat(os.path.join(prefix, 'YelpChi.mat'))
	# net_rur = yelp['net_rur']
	# net_rtr = yelp['net_rtr']
	# net_rsr = yelp['net_rsr']
	# yelp_homo = yelp['homo']
	#
	# sparse_to_adjlist(net_rur, prefix + 'yelp_rur_adjlists.pickle')
	# sparse_to_adjlist(net_rtr, prefix + 'yelp_rtr_adjlists.pickle')
	# sparse_to_adjlist(net_rsr, prefix + 'yelp_rsr_adjlists.pickle')
	# sparse_to_adjlist(yelp_homo, prefix + 'yelp_homo_adjlists.pickle')

	amz = loadmat(prefix + 'Amazon.mat')
	amz = loadmat(os.path.join(prefix, 'Amazon.mat'))
	net_upu = amz['net_upu']
	net_usu = amz['net_usu']
	net_uvu = amz['net_uvu']
	amz_homo = amz['homo']

	sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')
	sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')
	sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
	sparse_to_adjlist(amz_homo, prefix + 'amz_homo_adjlists.pickle')    # 不考虑关系的不同，全部都视为一种关系
