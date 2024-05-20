import numpy as np

F_MIN = 30
F_MAX = 10000

NUM_BINS = int(12*(np.log2(F_MAX/F_MIN)))
print(NUM_BINS)