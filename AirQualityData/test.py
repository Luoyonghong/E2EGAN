
# -*- coding: utf-8 -*-
import random
import os
import time
import math
import sys
import numpy as np
from sklearn import preprocessing
raw1 = np.loadtxt("./originData/test.csv", dtype=float,delimiter=",",skiprows=(1), usecols=(0,1,2,3,4,5,6,7))
print(raw1)

