import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pipe_break_funcs as pb
pd.options.mode.chained_assignment = None  # default='warn'

pipe_data = 'syn_pipe_data.csv'

guess = [20, .5, -1]
bnds = ((1e-5, None), (1e-5, None), (None, None))

data = list(pb.prepare_data(pipe_data))

fit = minimize(pb.fit_leyp, guess, args=(data), bounds=bnds, method='Nelder-Mead', options={'disp': True, 'return_all': True})
print(fit.x)
