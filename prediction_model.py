import numpy as np
import pandas as pd
from scipy.optimize import minimize
pd.options.mode.chained_assignment = None

# class PredictionModel():
#     def __init__(self, model, data, )

class LinearExtendedYuleProcess():
    def __init__(self, data, covariates: list, obs_a: int, obs_b: int, pred_s: int, pred_t: int):
        self.data = data
        self.covariates = covariates
        self.obs_a = obs_a
        self.obs_b = obs_b
        self.pred_s = pred_s
        self.pred_t = pred_t

        self.alpha = None
        self.delta = None
        self.beta = []

        self.parameters = {
            'alpha': self.alpha,
            'delta': self.delta,
            'beta': self.beta
        }

        if self.obs_a >= self.obs_b:
            raise ValueError("obs_b, the end of the observation period, must be greater than obs_a.")
        if self.pred_s >= self.pred_t:
            raise Value_error("pred_t, the end of the prediction period, must be greater than pred_s")
        

        
        if isinstance(self.data, pd.DataFrame):
            self.T = data['t'].to_numpy()
            self.a = data['a'].to_numpy(dtype=np.int16)
            self.b = data['b'].to_numpy(dtype=np.int16)
            self.n = data['n'].to_numpy(dtype=np.int16)
            self.X = data['X'].to_numpy()
        
        elif isintance(self.data, str) and self.data.endswith('.csv'):
            self.prepare_data()
        else:
            print("Data needs to be a Pandas DataFrame or a CSV file.")

    def update_parameters(self, key, new_value):
        setattr(self, key, new_value)
        self.parameter[key] = new_value
        

    def prepare_data(self):
        cov = self.covariates
        file = self.data
        df_data = pd.read_csv(file, usecols=['pipe_id', 't'])
        cols = ['pipe_id', 'a', 'b'] + cov
        df_data_all = pd.read_csv(file, usecols=cols)
    
        df_data1 = df_data.loc[df_data['t'].notna()]
        df_data1['t'] = df_data1['t'].astype(np.int16, copy=False)

        df_data0 = df_data.loc[df_data['t'].isna()]
        df_data0 = df_data0[~df_data0['pipe_id'].isin(df_data1['pipe_id'])]

        pipe_n = df_data1['pipe_id'].value_counts()
        df_T1 = df_data1.groupby('pipe_id')['t'].apply(lambda x: np.array(x)).reset_index()
        df_T1['n'] = df_T1['pipe_id'].map(pipe_n)

        df_data0 = df_data0.fillna(0)
        df_T0 = df_data0.groupby('pipe_id')['t'].apply(lambda x: np.array(x)).reset_index()
        df_T0['n'] = 0

        df_T = pd.concat([df_T1, df_T0])
        df_combined = pd.merge(df_T, df_data_all, 'left', on='pipe_id')
        df_combined = df_combined.drop_duplicates(subset='pipe_id')
        df_combined['diameter'] = df_combined['diameter'].apply(np.log)#.astype(np.float32)
        df_combined['diameter'] = df_combined['diameter'].astype(np.float32)
        df_combined = df_combined.dropna(subset=['diameter'])
        df_combined['X'] = df_combined.apply(lambda row: np.array([row['diameter']]), axis=1)

        self.T = df_combined['t'].to_numpy()
        self.a = df_combined['a'].to_numpy(dtype=np.int16)
        self.b = df_combined['b'].to_numpy(dtype=np.int16)
        self.n = df_combined['n'].to_numpy(dtype=np.int16)
        self.X = df_combined['X'].to_numpy()
        
    def log_non_zero(self, array):
        mask = array != 0
        logged_array = np.zeros_like(array, dtype=np.float32)
        logged_array[mask] = np.log(array[mask])
        s = np.sum(logged_array)
        return s
    
    def g_func(self):
        k = [list(range(x)) for x in self.n]
        k_list = np.array([np.sum([np.log(self.alpha**(-1) + x) for x in y]) for y in k], dtype=np.float32)
        return k_list    

    def h_func(self):
        transposed = np.array([x.T for x in self.X])
        term_0 = np.exp(np.dot(transposed, self.beta))
        term_0 = term_0.astype(np.float32)
        new_shape = term_0.shape[0]
        term_0 = np.reshape(term_0, new_shape)
        term_1 = self.alpha * np.power(self.b, self.delta) * term_0
        return term_1
    
    
    def log_likelihood(self, params):
    
        self.update_parameters('alpha', np.array(params[0], dtype=np.float32))
        self.update_parameters('delta', np.array(params[1], dtype=np.float32))
        self.update_parameters('beta', np.array([params[2:]], dtype=np.float32))
        # self.alpha = np.array(params[0], dtype=np.float32)
        # self.delta = np.array(params[1], dtype=np.float32)
        # self.beta = np.array([params[2:]], dtype=np.float32)

        # n = data[0]
        # a = data[1]
        # b = data[2]
        # X = data[3]
        # T = data[4]
        
        g = self.g_func()
        h = self.h_func()

        term_0 = self.n * np.log(self.alpha)
        term_1 = g
        term_2 = (self.alpha**(-1) + self.n) * h 
        term_3 = self.n * np.log(self.delta)
        term_4a = np.array([np.dot(np.transpose(x), self.beta) for x in self.X])
        new_shape = term_4a.shape[0]
        term_4a = term_4a.reshape(new_shape)
        term_4 = self.n * term_4a
        term_5 = (self.delta - 1) * np.array([log_non_zero(t) for t in self.T])
        term_6 = np.array([np.sum(self.alpha * self.T[i]**self.delta * np.exp(np.dot(np.transpose(self.X[i]), self.beta))) for i in range(len(self.X))])

        nll = -(np.sum(term_0 + term_1 - term_2 + term_3 + term_4 + term_5 + term_6))
        return nll

    def mu(self):
        mu = np.exp(self.alpha * np.power(t, self.delta) * np.exp(self.X * self.beta))
        self.mu = mu
        return mu

    def expected_value(self, j, t, s, b, a):
        term_0 = np.power(self.alpha, -1) + j 
        num = mu(t) - mu(s)
        denom = mu(b) - mu(a) + 1

        ev = term_0 * (num / denom)
        self.expected_value = ev
        return ev

    def fit_leyp(self, method='Nelder-Mead'):
        fit = minimize(log_likelihood, self.guess, args=(self.data), bounds=bnds, method=method, options={'disp': True, 'return_all': True})
