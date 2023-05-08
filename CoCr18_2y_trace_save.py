import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sympy
import datetime

# для расчетов над tdb
from pycalphad import Database, equilibrium, variables as v, binplot

# для MCMC расчетов
import pymc as pm  # пакет для MCMC расчетов 
import arviz as az # пакет для работы с типом данных arviz
import pytensor
import pytensor.tensor as pt
# import theano
# theano.config.exception_verbosity = 'high' # должно выдавать подробное описание ошибки, но не помогает

import aesara
# import aesara.tensor as аt

# import warnings
# # warnings.filterwarnings("ignore")

# def fxn():
#     warnings.warn("RuntimeWarning", RuntimeWarning)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     fxn()

# import seaborn as sns

# пути к tdb
cc10_path = "tdbs/CoCr-18Cac_with_new_functions.tdb"

print(f"Running on PyMC v{pm.__version__}") # 5.3.1
print(f"Running on NumPy v{np.__version__}") # 1.22.1
print(f"Running on ArviZ v{az.__version__}") # 0.12.1

def from_xarray_to_pandas(xarray_data, component_str, goal_phase_str):

    import numpy as np
    import pandas as pd
    
    cr_tuple = xarray_data.X.sel(component=component_str).data[0][0]
    phase_tuple = xarray_data.Phase.data[0][0]
    t_tuple = xarray_data.T.data
    
    df = pd.DataFrame()

    # создали таблицу со всеми данными
    for i in range(phase_tuple.shape[2]):
        df_temp = pd.DataFrame(columns=['T','num','phase','conc'])
        df_temp['T'] = t_tuple
        df_temp['phase'] = phase_tuple[:, 0, i]
        df_temp['conc'] = cr_tuple[:, 0, i]
        
        df = pd.concat([df, df_temp])

    # смерджили нужные нам данные с имеющимися температурами
    df_res = pd.DataFrame({'T': t_tuple})
    df_res = pd.merge(df_res['T']
                    , df[(df['phase'] == goal_phase_str)][['T','phase','conc']]
                    , how = 'left'
                    , left_on = 'T'
                    , right_on = 'T')

    # заменили NaN значения
    df_res['phase'].fillna(goal_phase_str, inplace=True)
    df_res['conc'].fillna(np.float32(10), inplace=True)

    return df_res

# для 0.5
df_sigma_fcc = pd.read_excel('emp_data/sigma_fcc_allibert.xls')
df_sigma_hcp = pd.read_excel('emp_data/sigma_hcp_allibert.xls')

df_hcp_fcc = pd.concat([df_sigma_fcc, df_sigma_hcp])

df_hcp_fcc['T'] = df_hcp_fcc['T'].round(2)
df_hcp_fcc['cr_conc'] = df_hcp_fcc['cr_conc'].round(6)
df_hcp_fcc = df_hcp_fcc[(df_hcp_fcc['phase'] == 'sigma_old')].reset_index()
df_hcp_fcc.sort_values('T', inplace=True)
# df_hcp_fcc

# для 0.75
df_bcc = pd.read_excel('emp_data/sigma_bcc_allibert.xls')

df_bcc['T'] = df_bcc['T'].round(2)
df_bcc['cr_conc'] = df_bcc['cr_conc'].round(6)
df_bcc = df_bcc[(df_bcc['phase'] == 'sigma_old')].reset_index()
df_bcc.sort_values('T', inplace=True)
# df_bcc

# общие данные для расчетов
db10 = Database(cc10_path)

press = 101325
elements = ['CR', 'CO', 'VA']
component = 'CR'
el_cnt = 1

phase = 'SIGMA_D8B'
parameters_list = ['GSCRCO1', 'GSCOCRCO1', 'GSCOCRCO2', 'GSCRCO2', 'GSCOCR1',  'GSCOCR2', 'GSCOCR3']

# данные для расчетов с концентрацией хрома 0.5
y_obs_05 = df_hcp_fcc['cr_conc'].values

T_05 = df_hcp_fcc['T'].to_numpy()
conditions_05 = {v.X('CR'):0.5, v.P: 101325, v.T: T_05, v.N: el_cnt}

print('T', len(T_05))
print('T', T_05)
print('y_obs', y_obs_05)

# данные для расчетов с концентрацией хрома 0.75
y_obs_75 = df_bcc['cr_conc'].values

T_75 = df_bcc['T'].to_numpy()
conditions_75 = {v.X('CR'):0.75, v.P: 101325, v.T: T_75, v.N: el_cnt}

print('T', len(T_75))
print('T', T_75)
print('y_obs', y_obs_75)

# define a pytensor Op for our likelihood function
class LogLike(pt.Op):
#     определяем тип входящих и исходящих данных
    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.fvector]  # outputs a single scalar value (the log likelihood)

    def __init__(self, db, conditions, phase, elements, component, parameters_list):
        self.db_tdb = db
        self.conditions_dict = conditions
        self.phases_list = list(self.db_tdb.phases.keys())
        self.phase_str = phase
        self.elements_list = elements
        self.component_str = component
        self.parameters_list = parameters_list
        
        self.y_eqs = []
        self.likelihoods = []

    def perform(self, node, inputs, outputs):
        
        (theta,) = inputs  # this will contain my variables
        
        # новая версия
        new_parameters = dict()
        
        for i in range(len(self.parameters_list)):
            new_parameters[self.parameters_list[i]] = inputs[0][i]

        
        # старая версия
        # COCRCO_0, COCRCO_1, COCRCR_0, COCRCR_1 = theta

        # new_parameters = {
        #  'SIGMA_OLD_COCRCO_0' : COCRCO_0
        #  ,'SIGMA_OLD_COCRCO_1': COCRCO_1
        #  ,'SIGMA_OLD_COCRCR_0': COCRCR_0
        #  ,'SIGMA_OLD_COCRCR_1': COCRCR_1
        # }
        # print(new_parameters)
        y_eq = (from_xarray_to_pandas(equilibrium(self.db_tdb
                                            , self.elements_list
                                            , self.phases_list
                                            , self.conditions_dict
                                            , parameters = new_parameters
                                        ), self.component_str, self.phase_str)['conc']
                .astype(np.float32)
                .to_numpy())
        
        # print(len(self.conditions_dict[v.T]))
        # print(self.conditions_dict[v.T])

        outputs[0][0] = y_eq
    
test_model = pm.Model()

logl_05 = LogLike(db10, conditions_05, phase, elements, component, parameters_list)
logl_75 = LogLike(db10, conditions_75, phase, elements, component, parameters_list)
s = 0.000001

with test_model:
    # uniform priors on m and c
    GSCRCO1 = pm.Normal("GSCRCO1", mu=-526000.0, sigma=s) # sigma = 1,
    GSCOCRCO1 = pm.Normal("GSCOCRCO1", mu=-200000.0, sigma=s)
    GSCOCRCO2 = pm.Normal("GSCOCRCO2", mu=20.0, sigma=s)
    GSCRCO2 = pm.Normal("GSCRCO2", mu=49.0, sigma=s) 
    GSCOCR1 = pm.Normal("GSCOCR1", mu=180000.0, sigma=s) 
    GSCOCR2 = pm.Normal("GSCOCR2", mu=348000.0, sigma=s) 
    GSCOCR3 = pm.Normal("GSCOCR3", mu=525000.0, sigma=s) 
    
    theta = pt.as_tensor_variable([GSCRCO1, GSCOCRCO1, GSCOCRCO2, GSCRCO2, GSCOCR1,  GSCOCR2, GSCOCR3])
    
    y_obs_05_pm = pm.ConstantData(name = 'y_obs_05_data', value=y_obs_05)
    y_obs_75_pm = pm.ConstantData(name = 'y_obs_75_data', value=y_obs_75)
    
    # y_norm_05 = pm.Normal("y_norm_05", mu=logl_05(theta), sigma = 0.001, observed=np.float32(y_obs_05))
    # y_norm_75 = pm.Normal("y_norm_75", mu=logl_75(theta), sigma = 0.001, observed=np.float32(y_obs_75))
    y_norm_05 = pm.Normal("y_norm_05", mu=logl_05(theta), sigma = s, observed=y_obs_05_pm) # sigma = 0.001,
    y_norm_75 = pm.Normal("y_norm_75", mu=logl_75(theta), sigma = s,observed=y_obs_75_pm)
                             
def trace_f(test_model):
    pytensor.config.exception_verbosity = 'high'
    # import psutil
    # print('trace done')
    with test_model:
        # trace = pm.sample(5, tune=5, chains = 4, idata_kwargs={"log_likelihood": True}, progressbar=True) # количество ядер на вм
        trace = pm.sample(1000, tune=700, chains = 2, idata_kwargs={"log_likelihood": True}, progressbar=True) # количество ядер на вм
        # trace = pm.sample(draws=2000, tune=500, idata_kwargs={"log_likelihood": True}, progressbar=True)
    trace.to_json('trace_cocr18_2Sx700x1000x2_20230508.json')

    with test_model:
            ppc = pm.sample_posterior_predictive(trace)

    ppc.to_json('ppc_cocr18_2Sx700x1000x2_20230508.json')

    # return trace

def ppc_f(test_model, trace):
    with test_model:
            ppc = pm.sample_posterior_predictive(trace)

    ppc.to_json('ppc_cocr18_2Sx700x1000x2_20230508.json')

def pp_f(test_model):
    with test_model:
        pp = pm.sample_prior_predictive(samples=2000)

    pp.to_json('pp_cocr18_2Sx2000_20230508.json')

if __name__ == '__main__':
    trace_f(test_model)
    # ppc_f(test_model, traces)
    pp_f(test_model)