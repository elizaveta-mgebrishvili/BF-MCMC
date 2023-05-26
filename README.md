# BF-MCMC

## Используемые сокращения

ТДБ - темродинамическая база данных\
ТДМ - термодинамическая модель

## Краткое описание проекта

В данном репозитории представлена методика, позволяющая производить сравнение термодинамических моделей на основании их статистической значимости.
В частности, представлен пример реализации сравнения для двух моделей описания равновесных свойств SIGMA фазы системы Co-Cr.

## Описание представленных файлов

В репозитории представлено несколько типов файлов:
1. Файлы, которые позволяют произвести семплирование апостериорных распределений параметров моделей, и их предсказания:
    - [CoCr_2y_trace_save.py](CoCr_2y_trace_save.py) - для ТДБ [tdbs/CoCr-01Oik_with_new_functions.tdb](tdbs/CoCr-01Oik_with_new_functions.tdb)
    - [CoCr18_2y_trace_save.py](CoCr18_2y_trace_save.py) - для ТДБ [tdbs/CoCr-18Cac_with_new_functions.tdb](tdbs/CoCr-18Cac_with_new_functions.tdb)
2. Файлы, где представлен пример оценки полученных распределний:
    - [CoCr10_chains_analysis.ipynb](CoCr10_chains_analysis.ipynb) - для цепей, полученных в [CoCr_2y_trace_save.py](CoCr_2y_trace_save.py)
    - [CoCr18_chains_analysis.ipynb](CoCr18_chains_analysis.ipynb) - для цепей, полученных в [CoCr18_2y_trace_save.py](CoCr18_2y_trace_save.py)
3. Файл, в котором реализовано сравнение моделей на основании полученных цепей: [models_comparison.ipynb](models_comparison.ipynb)

## Описание подходов, реализуемых в файлах

Для сравнения ТДМ используются методы байесовской статистики (WAIC, LOO, отношение апостериорных вероятностей), основанные на получении байесвских распределений: распределении правдопдобия и апостериорных распределений параметров моделей.

Для получения указанных распределений в файлах типа 1 реализовано соответсвующее семплирование.
Для реализации семлпирования использовались методы библиотеки PyMC (v.5), для реализации термодинамических вычислений - функция equilibrium библиотеки pyCALPHAD. Также для реализации вычислений используется класс LogLike: данный класс был разработан для реализации перегрузки функций PyTensor (библиотека, на основании которой производятся вычисления в PyMC). Необходимость разработки данного класса обусловлена тем, что для семлирования апостериорных распределений в PyMC модели необходимо задать явное соответсвие между предсказаниями модели, зависящих от распределений парамтеров, и опытными данными.

Затем, в файлах 2 типа производится анализ полученных распределений. Если цепи сходятся и не имеют сильной автокорреляции, то мы считаем их пригодными для совершения дальнейшего анализа.

В завершении, в файле 3 типа происходит расчет целевых метрик. При этом, для вычисления WAIC и LOO используются встроенные методы библиотеки ArviZ, а для расчета отношения апостериорных вероятностей была разработана методика, которая позволяет вычислить апостериорную вероятность модели по методу наивной аппроксимации маржинального распределения.

## Рекомендации по запуску скриптов

В виду того, что в файлах 2 типа производятся вычисления, которые сильно нагружают систему, рекомендуется производить запуск этих файлов из терминала. \
Также стоит отметить, что время выполнения данных файлов довольно длительное, поэтому лучше запускать их на машинах с большим объемом оперативной памяти и при этом подбирать параметр core (метода pm.sample) в соответствии с возможностями машины. К слову, при запуске скриптов из терминала можно наблюдать progressbar, который прогнозирует примерное время расчета.\
Для корректной работы скриптов необходимо и достаточно иметь ПО следующих версий:
1. PyMC: 5.3.1 или 5.1.2
2. ArviZ: 0.12.1
3. NumPy: 1.22.1

Кроме того, рекомендуется установить g++, в соответсвии с тербованиями PyMC, так как это может ускорить реализацию расчетов. Наиболее удобный способ установки описан [здесь](https://stackoverflow.com/questions/30069830/how-can-i-install-mingw-w64-and-msys2) и [здесь](https://www.msys2.org/).


