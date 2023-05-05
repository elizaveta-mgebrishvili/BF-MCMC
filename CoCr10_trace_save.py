print('start')

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

# import seaborn as sns

# пути к tdb
cc10_path = "tdbs/CoCr-01Oik_with_new_functions.tdb"

print(f"Running on PyMC v{pm.__version__}") # 5.1.2
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
    df_res['conc'].fillna(10, inplace=True)

    return df_res
print('from_xarray_to_pandas')

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

        outputs[0][0] = y_eq
print('LogLike')

                             
df_sigma_fcc = pd.read_excel('emp_data/sigma_fcc_allibert.xls')
# df_sigma_bcc = pd.read_excel('emp_data/sigma_bcc_allibert.xls')
df_sigma_hcp = pd.read_excel('emp_data/sigma_hcp_allibert.xls')

df_sigma_fcc = pd.concat([df_sigma_fcc, df_sigma_hcp])

df_sigma_fcc['T'] = df_sigma_fcc['T'].round(2)
df_sigma_fcc['cr_conc'] = df_sigma_fcc['cr_conc'].round(6)
df_sigma_fcc_sigma_old = df_sigma_fcc[(df_sigma_fcc['phase'] == 'sigma_old')].reset_index()
df_sigma_fcc_sigma_old = df_sigma_fcc_sigma_old.sort_values('T')


db10 = Database(cc10_path)

press = 101325
elements = ['CR', 'CO', 'VA']
el_cnt = 1


T = df_sigma_fcc_sigma_old['T'].to_numpy()
phase = 'SIGMA_OLD'

y_obs = df_sigma_fcc_sigma_old['cr_conc'].values
conditions = {v.X('CR'):0.5, v.P: 101325, v.T: T, v.N: el_cnt}
parameters_list = ['SIGMA_OLD_COCRCO_0', 'SIGMA_OLD_COCRCO_1', 'SIGMA_OLD_COCRCR_0', 'SIGMA_OLD_COCRCR_1']
component = 'CR'

print('T', T)
print('y_obs', y_obs)
# print('phases', phases)
print('phase', phase)

test_model = pm.Model()

logl = LogLike(db10, conditions, phase, elements, component, parameters_list)
print('with test_model')

with test_model:
    # uniform priors on m and c
    COCRCO_0 = pm.Normal("SIGMA_OLD_COCRCO_0", mu=-103863.0, sigma=1)
    COCRCO_1 = pm.Normal("SIGMA_OLD_COCRCO_1", mu=47.47, sigma=1)
    COCRCR_0 = pm.Normal("SIGMA_OLD_COCRCR_0", mu=-248108.8, sigma=1)
    COCRCR_1 = pm.Normal("SIGMA_OLD_COCRCR_1", mu=79.12, sigma=1) 
    
    theta = pt.as_tensor_variable([COCRCO_0, COCRCO_1, COCRCR_0, COCRCR_1])
        
    # y_det = pm.Deterministic("y_det", logl(theta))
    y_norm = pm.Normal("y_norm", mu=logl(theta), sigma = 0.001, observed=y_obs)
    pp = pm.sample_prior_predictive(samples=2000)
    # trace = pm.sample(draws=2000, tune=500, idata_kwargs={"log_likelihood": True}, progressbar=True)
pp.to_json('calc_res/pp_cocr10_2000.json')
print('pp saved')
print('start sample')
with test_model:
    trace = pm.sample(1000, tune=700, chains = 4, idata_kwargs={"log_likelihood": True}) # количество ядер на вм

trace.to_json('calc_res/trace_cocr10_700x1000x4.json')
print('trace saved')


with test_model:
    ppc = pm.sample_posterior_predictive(trace)
ppc.to_json('calc_res/ppc_cocr10_700x1000x4.json')
print('ppc saved')
print('finish')

