################################################### ПОДГРУЗКА ПАКЕТОВ ################################################
import pandas as pd
import numpy as np

# для расчетов над tdb
from pycalphad import Database, equilibrium, variables as v

# для MCMC расчетов
import pymc as pm
import aesara
import arviz as az
import pytensor
import pytensor.tensor as pt

# путь к тдб с параметрами, обернутыми в функцию
cc10_path = "tdbs/CoCr-01Oik_with_new_functions.tdb"
# пути к опытным данным
path_sigma_fcc = 'emp_data/sigma_fcc_allibert.xls'
path_sigma_hcp = 'emp_data/sigma_hcp_allibert.xls'
path_bcc = 'emp_data/sigma_bcc_allibert.xls'


# если скрипт не работает, возможно дело в версиях библиотек
print(f"Running on PyMC v{pm.__version__}") # 5.3.1
print(f"Running on NumPy v{np.__version__}") # 1.22.1
print(f"Running on ArviZ v{az.__version__}") # 0.12.1

########################################### СОЗДАНИЕ СПЕЦИФИЧЕСКИХ ФУНКЦИЙ ############################################

def from_xarray_to_pandas(xarray_data, component_str, goal_phase_str):

    '''
    Данная функция позволяет преобразовывать данные формата xarray_data
    в список значений концентрации вещества в равновесном состоянии определенной фазы
    
    Т.о. на вход подаются следующие данные:
        xarray_data - результат расчета функции equilibrium
        component_str - вещество, концентрацию которого необходимо получить
        goal_phase_str - фаза, для равнвоесного состояния которой нужно получить данные
    
    А на выходе функции получаем dataframe из трех стобцов:
        T - температура
        phase - название равновесной фазы (соответствует goal_phase_str)
        conc - концентрация вещества component_str
    
    При этом, в случае, если размерность списка T не совпадает с размерностью списка conc,
    недостающие значения типа nan заменяются на константы:
        для phase - значение goal_phase_str
        для conc - 10
    '''

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

########################################### ПОДГРУЗКА ДАННЫХ ###############################################################

# создание объекта Database
db10 = Database(cc10_path)

# условия для проведения расчетов с помощью equilibrium
press = 101325
elements = ['CR', 'CO', 'VA']
component = 'CR'
el_cnt = 1

phase = 'SIGMA_OLD'
parameters_list = ['SIGMA_OLD_COCRCO_0', 'SIGMA_OLD_COCRCO_1', 'SIGMA_OLD_COCRCR_0', 'SIGMA_OLD_COCRCR_1']

# подгрузка и преобразование опытных данных при концентрации хрома 0.5
df_sigma_fcc = pd.read_excel(path_sigma_fcc)
df_sigma_hcp = pd.read_excel(path_sigma_hcp)

df_hcp_fcc = pd.concat([df_sigma_fcc, df_sigma_hcp])

df_hcp_fcc['T'] = df_hcp_fcc['T'].round(2)
df_hcp_fcc['cr_conc'] = df_hcp_fcc['cr_conc'].round(6)
df_hcp_fcc = df_hcp_fcc[(df_hcp_fcc['phase'] == 'sigma_old')].reset_index()
df_hcp_fcc.sort_values('T', inplace=True)

y_obs_05 = df_hcp_fcc['cr_conc'].values
T_05 = df_hcp_fcc['T'].to_numpy()
conditions_05 = {v.X('CR'):0.5, v.P: 101325, v.T: T_05, v.N: el_cnt}

# подгрузка и преобразование опытных данных при концентрации хрома 0.75
df_bcc = pd.read_excel(path_bcc)

df_bcc['T'] = df_bcc['T'].round(2)
df_bcc['cr_conc'] = df_bcc['cr_conc'].round(6)
df_bcc = df_bcc[(df_bcc['phase'] == 'sigma_old')].reset_index()
df_bcc.sort_values('T', inplace=True)

y_obs_75 = df_bcc['cr_conc'].values
T_75 = df_bcc['T'].to_numpy()
conditions_75 = {v.X('CR'):0.75, v.P: 101325, v.T: T_75, v.N: el_cnt}

########################################### ОПРЕДЕЛЕНИЕ КЛАССА ДЛЯ ПЕРЕГРУЗКИ ФУНКЦИЙ ######################################

class LogLike(pt.Op):
    itypes = [pt.dvector]
    otypes = [pt.fvector]

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
        
        (theta,) = inputs
        
        new_parameters = dict()
        
        for i in range(len(self.parameters_list)):
            new_parameters[self.parameters_list[i]] = inputs[0][i]        
        
        y_eq = (from_xarray_to_pandas(equilibrium(self.db_tdb
                                            , self.elements_list
                                            , self.phases_list
                                            , self.conditions_dict
                                            , parameters = new_parameters
                                        ), self.component_str, self.phase_str)['conc']
                .astype(np.float32)
                .to_numpy())

        outputs[0][0] = y_eq

########################################### ФУНКЦИЯ ДЛЯ ЗАПУСКА В КОМАНДНОЙ СТРОКЕ ######################################

def func(db10, conditions_05, conditions_75, phase, elements, component, parameters_list, y_obs_05, y_obs_75):
    pytensor.config.exception_verbosity = 'high' 
    import psutil

    test_model = pm.Model()

    logl_05 = LogLike(db10, conditions_05, phase, elements, component, parameters_list)
    logl_75 = LogLike(db10, conditions_75, phase, elements, component, parameters_list)

    with test_model:
        COCRCO_0 = pm.Normal("SIGMA_OLD_COCRCO_0", mu=-103863.0, sigma=1)
        COCRCO_1 = pm.Normal("SIGMA_OLD_COCRCO_1", mu=47.47, sigma=1)
        COCRCR_0 = pm.Normal("SIGMA_OLD_COCRCR_0", mu=-248108.8, sigma=1)
        COCRCR_1 = pm.Normal("SIGMA_OLD_COCRCR_1", mu=79.12, sigma=1)

        theta = pt.as_tensor_variable([COCRCO_0, COCRCO_1, COCRCR_0, COCRCR_1])

        y_norm_05 = pm.Normal("y_norm_05", mu=logl_05(theta), sigma = 0.001, observed=y_obs_05)
        y_norm_75 = pm.Normal("y_norm_75", mu=logl_75(theta), sigma = 0.001, observed=y_obs_75)

        trace = pm.sample(1000, tune=700, chains = 4, idata_kwargs={"log_likelihood": True}, progressbar=True)
        trace.to_json('trace_cocr10_2Sx700x1000x4.json')

        with test_model:
            ppc = pm.sample_posterior_predictive(trace)

        ppc.to_json('ppc_cocr10_2Sx700x1000x4.json')

        with test_model:
            pp = pm.sample_prior_predictive(samples=4000)

        pp.to_json('pp_cocr10_2Sx4000.json')

########################################### ЗАПУСК РАСЧЕТА #########################################################
if __name__ == '__main__':
    func(db10, conditions_05, conditions_75, phase, elements, component, parameters_list, y_obs_05, y_obs_75)