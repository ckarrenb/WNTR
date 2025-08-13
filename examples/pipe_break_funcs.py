import numpy as np
import pandas as pd
from scipy.optimize import minimize
pd.options.mode.chained_assignment = None  # default='warn'


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
    covs = params[2:]
    size = len(covs)
    beta = np.array(covs, dtype=np.float32)
    beta = beta.reshape((size, 1))

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
    term_4a = np.array([np.dot(np.transpose(x), beta) for x in X])
    print(term_4a.isinf())
        
    new_shape = term_4a.shape[0]
    term_4a = term_4a.reshape(new_shape)
    # term_4 = n * term_4a
    # print(np.isinf(term_4a))
    # term_4 = n * [np.dot(np.transpose(x), beta) for x in X]
    # term_5 = (delta - 1) * np.array([np.sum(np.log(t)) for t in T])
    # term_5 = (delta - 1) * np.array([log_non_zero(t) for t in T])
    # term_6 = np.array([np.sum(alpha * T[i]**delta * np.exp(np.dot(np.transpose(X[i]),beta))) for i in range(len(X))])

    # nll = -(np.sum(term_0 + term_1 - term_2 + term_3 + term_4 + term_5 + term_6))
    # return nll


def prepare_data(file, cov=['diameter']):
    df_data = pd.read_csv(file, usecols=['pipe_id', 't'])
    cols = ['pipe_id', 'a', 'b'] + cov
    df_data_all = pd.read_csv(file, usecols=cols)
    
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
    syn_X = df_combined['X'].to_numpy()

    return syn_n, syn_a, syn_b, syn_X, syn_T

        
        
def prepare_data_2(file, cov=['diameter', 'length']):
    df_data = pd.read_csv(file, usecols=['pipe_id', 'install_year', 'break_year']) # pipe break history
    cols = ['pipe_id', 'install_year'] + cov
    df_data_all = pd.read_csv(file, usecols=cols) # all pipe inventory
    beg_obs = df_data['break_year'].min()
    end_obs = df_data['break_year'].max()
    df_data_all['a'] = np.maximum((beg_obs - df_data_all['install_year']), 0)
    df_data_all['b'] = np.maximum((end_obs - df_data_all['install_year']), 0)
        
    df_data['t'] = df_data['break_year'] - df_data['install_year']
    df_data1 = df_data.loc[df_data['t'].notna()]
    df_data1 = df_data1.loc[df_data1['t'] >= 0]
    df_data1['t'] = df_data1['t'].astype(np.int16, copy=False)

    df_data0 = df_data.loc[df_data['t'].isna()]
    df_data0 = df_data0[~df_data0['pipe_id'].isin(df_data1['pipe_id'])]

    # df_T1 = df_data1.groupby('pipe_id')['t'].agg(list).reset_index()
    pipe_n = df_data1['pipe_id'].value_counts()
    df_T1 = df_data1.groupby('pipe_id')['t'].apply(lambda x: np.array(x)).reset_index()
    df_T1['n'] = df_T1['pipe_id'].map(pipe_n)

    
    df_data0 = df_data0.fillna(0)
    df_data0['t'] = df_data0['t'].astype(np.int16, copy=False)
    # df_T0 = df_data0.groupby('pipe_id')['t'].agg(list).reset_index()
    df_T0 = df_data0.groupby('pipe_id')['t'].apply(lambda x: np.array(x)).reset_index()
    df_T0['n'] = 0

    
    df_T = pd.concat([df_T1, df_T0])
    df_combined = pd.merge(df_T, df_data_all, 'left', on='pipe_id')
    df_combined = df_combined.drop_duplicates(subset='pipe_id')
    df_combined = df_combined.loc[df_combined['install_year'] != 0]
    for c in cov: 
        df_combined[c] = df_combined[c].apply(np.log)#.astype(np.float32)
        df_combined[c] = df_combined[c].astype(np.float32)
    df_combined = df_combined.dropna()
    df_combined['X'] = df_combined.apply(lambda row: np.array([row[col] for col in cov]), axis=1)
    # df_combined.to_csv('cleaned_joined_pwsa.csv')

    
    syn_T = df_combined['t'].to_numpy()
    syn_a = df_combined['a'].to_numpy(dtype=np.int16)
    syn_b = df_combined['b'].to_numpy(dtype=np.int16)
    syn_n = df_combined['n'].to_numpy(dtype=np.int16)
    syn_X = df_combined['X'].to_numpy()

    return syn_n, syn_a, syn_b, syn_X, syn_T

        
        
        
        
def mu(t, alpha, delta, X, beta):
    mu = np.exp(alpha * np.power(t, delta) * np.exp(X * beta))
    return mu

def expected_value(alpha, j, t, s, b, a):
    term_0 = np.power(alpha, -1) + j 
    num = mu(t) - mu(s)
    denom = mu(b) - mu(a) + 1

    ev = term_0 * (num / denom)
    return ev
    
    
data = list(prepare_data_2('./data/joined_pwsa_all.csv'))
# data = [syn_n, syn_a, syn_b, syn_X, syn_T]
guess = [10, .5, 1, 1]
fit_leyp(guess, data)
# bnds = ((1e-5, None), (1e-5, None), (None, None), (None, None))
# fit = minimize(fit_leyp, guess, args=(data), bounds=bnds, method='Nelder-Mead', options={'disp': True, 'return_all': True})
# print(fit.x)
