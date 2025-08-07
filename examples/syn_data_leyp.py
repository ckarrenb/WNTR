import numpy as np
import pandas as pd
from scipy.optimize import minimize
pd.options.mode.chained_assignment = None  # default='warn'

rng = np.random.default_rng(1234)

m = int(5000) #number of pipes
b_year = 1980 # first year of observations
e_year = 2017 # last year of observations

df_data = pd.read_csv('syn_pipe_data.csv', usecols=['pipe_id', 't'])
df_data_all = pd.read_csv('syn_pipe_data.csv', usecols=['pipe_id', 'diameter', 'a', 'b'], dtype={'pipe_id':str, 'diameter':np.float32, 'a':np.int16, 'b':np.int16})

def gen_n(num_pipes, perc_fail):
    n_repl = int(num_pipes*perc_fail)
    pipes = np.zeros(num_pipes, dtype=int)
    idx = rng.choice(num_pipes, n_repl, replace=False)
    n = rng.integers(1, 6, size=n_repl)
    pipes[idx] = n
    return pipes

def log_non_zero(array):
    mask = array != 0
    logged_array = np.zeros_like(array, dtype=np.float32)
    logged_array[mask] = np.log(array[mask])
    s = np.sum(logged_array)
    return s
    
def g_func(alpha, n):
    k = [list(range(x)) for x in n]
    k_list = np.array([np.sum([np.log(alpha**(-1) + x) for x in y]) for y in k], dtype=np.float32)
    return k_list    

def h_func(a, b, X, alpha, beta, delta):
    transposed = np.array([x.T for x in X])
    # term_0 = np.exp(np.array(np.dot([np.transpose(x), beta for x in X])))
    term_0 = np.exp(np.dot(transposed, beta))
    term_0 = term_0.astype(np.float32)
    new_shape = term_0.shape[0]
    term_0 = np.reshape(term_0, new_shape)
    term_1 = alpha * np.power(b, delta) * term_0
    # term_2 = np.exp(alpha * term_0 * (np.power(a, delta) - np.power(b, delta))) 
    # term_3 = np.exp(-alpha * np.power(b, delta) * term_0)
    return term_1 #+ np.log(1 - term_2 + term_3)
    
    
def fit_leyp(params, data):
    
    alpha = np.array(params[0], dtype=np.float32)
    delta = np.array(params[1], dtype=np.float32)
    beta = np.array([params[2:]], dtype=np.float32)

    n = data[0]
    a = data[1]
    b = data[2]
    X = data[3]
    T = data[4]
    g = g_func(alpha, n)
    h = h_func(a, b, X, alpha, beta, delta)

    term_0 = n * np.log(alpha)
    term_1 = g
    term_2 = (alpha**(-1) + n) * h 
    term_3 = n * np.log(delta)
    term_4a = np.array([np.dot(np.transpose(x), beta) for x in X])
    new_shape = term_4a.shape[0]
    term_4a = term_4a.reshape(new_shape)
    term_4 = n * term_4a
    # term_4 = n * [np.dot(np.transpose(x), beta) for x in X]
    # term_5 = (delta - 1) * np.array([np.sum(np.log(t)) for t in T])
    term_5 = (delta - 1) * np.array([log_non_zero(t) for t in T])
    term_6 = np.array([np.sum(alpha * T[i]**delta * np.exp(np.dot(np.transpose(X[i]),beta))) for i in range(len(X))])
    print('term0: ', np.sum(term_0), 'term1: ', np.sum(term_1), 'term2: ', np.sum(term_2), 'term3: ', np.sum(term_3), 'term4: ', np.sum(term_4), 'term5: ', np.sum(term_5), 'term6: ', np.sum(term_6))

    nll = -(np.sum(term_0 + term_1 - term_2 + term_3 + term_4 + term_5 + term_6))
    return nll

# n0 = np.random.randint(0, 11, m)
# n = gen_n(m, 0.06)
# year_con = rng.integers(1890, 2022, m)
# a0 = np.random.randint(0, 50, m)
# a = np.maximum(b_year - year_con, 0)
# b = np.maximum(e_year - year_con, 0)
# b0 = np.random.randint(0, 50, m)
# b00 = a0 + b0
# X0 = rng.integers(0, 25, size=(m, 2))
# pipe_sizes =   [ 0.75,   1.0,   1.5,     2,     3,     4,     6,      8,    10,    12,    16,    18,    20,    24,    30,    36,    42]
# pipe_sizes_p = [0.003, 0.004, 0.002, 0.003, 0.01, 0.080, 0.371, 0.310, 0.024, 0.11, 0.040, 0.002, 0.020, 0.012, 0.003, 0.004, 0.002]
# x_dia = rng.choice(pipe_sizes, m, p=pipe_sizes_p)
# x_len = rng.integers(1, 1000, m) / 1000
# XX = np.column_stack((x_dia, x_len))
# T = [[np.sort(rng.integers(1, np.maximum(b[i], 3), n[i])) for i in range(m)] for _ in range(m)][0]
# T0 = [[np.sort(rng.integers(1, 50, i)) for i in n] for _ in range(m)][0]
# data = [n, a, b, XX, T]
# data0 = [n0, a0, b00, X0, T]

guess = [20, .5, -1]
bnds = ((1e-5, None), (1e-5, None), (None, None))

df_data1 = df_data.loc[df_data['t'].notna()]
df_data1['t'] = df_data1['t'].astype(np.int16, copy=False)

df_data0 = df_data.loc[df_data['t'].isna()]
df_data0 = df_data0[~df_data0['pipe_id'].isin(df_data1['pipe_id'])]

# df_T1 = df_data1.groupby('pipe_id')['t'].agg(list).reset_index()
pipe_n = df_data1['pipe_id'].value_counts()
df_T1 = df_data1.groupby('pipe_id')['t'].apply(lambda x: np.array(x)).reset_index()
df_T1['n'] = df_T1['pipe_id'].map(pipe_n)

df_data0 = df_data0.fillna(0)
# df_T0 = df_data0.groupby('pipe_id')['t'].agg(list).reset_index()
df_T0 = df_data0.groupby('pipe_id')['t'].apply(lambda x: np.array(x)).reset_index()
df_T0['n'] = 0

df_T = pd.concat([df_T1, df_T0])
df_combined = pd.merge(df_T, df_data_all, 'left', on='pipe_id')
df_combined = df_combined.drop_duplicates(subset='pipe_id')
df_combined['diameter'] = df_combined['diameter'].apply(np.log)#.astype(np.float32)
df_combined['diameter'] = df_combined['diameter'].astype(np.float32)
df_combined = df_combined.dropna(subset=['diameter'])
# df_combined['X'] = df_combined[['diameter']].values
df_combined['X'] = df_combined.apply(lambda row: np.array([row['diameter']]), axis=1)

syn_T = df_combined['t'].to_numpy()
syn_a = df_combined['a'].to_numpy(dtype=np.int16)
syn_b = df_combined['b'].to_numpy(dtype=np.int16)
syn_n = df_combined['n'].to_numpy(dtype=np.int16)

# syn_X = df_combined['diameter'].apply(lambda x: np.array(x))
syn_X = df_combined['X'].to_numpy()

# for i in range(len(syn_T)):
#     print(syn_T[i], syn_a[i], syn_b[i], syn_n[i], syn_X[i])
data = [syn_n, syn_a, syn_b, syn_X, syn_T]
params = guess
alpha = np.array(params[0], dtype=np.float32)
delta = np.array(params[1], dtype=np.float32)
beta = np.array([params[2:]], dtype=np.float32)

n = data[0]
a = data[1]
b = data[2]
X = data[3]
T = data[4]

g = g_func(alpha, n)
h = h_func(a, b, X, alpha, beta, delta)
# term_0 = n * np.log(alpha)
# term_1 = g
# term_2 = (alpha**(-1) + n) * h 
# term_3 = n * np.log(delta)
# term_4a = np.array([np.dot(np.transpose(x), beta) for x in X])
# new_shape = term_4a.shape[0]
# term_4a = term_4a.reshape(new_shape).astype(np.float32)
# term_4 = n * term_4a
# term_5 = (delta - 1) * np.array(np.sum([np.log(t) if not np.all(t == 0) else 0 for t in T]))
# term_5 = (delta - 1) * np.array([log_non_zero(t) for t in T])
# term_6 = np.array([np.sum(alpha * T[i]**delta * np.exp(np.dot(np.transpose(X[i]),beta))) for i in range(len(X))])
# data = []
print(fit_leyp(params, data))
# fit = minimize(fit_leyp, guess, args=(data), bounds=bnds, method='Nelder-Mead', options={'disp': True, 'return_all': True})
# print(fit.x)
