from sklearn.datasets import *
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz


@st.cache
def load_data():
    '''
    Загрузка данных
    '''
    pre_data = pd.read_csv('Admission_Predict.csv', sep=",")

    data = pre_data.drop(["Serial No."], axis=1)
    data_X = data.drop(["Chance of Admit "], axis=1)
    data_Y = data[["Chance of Admit "]]

    sc = MinMaxScaler()
    data_X_sc = sc.fit_transform(data_X)
    data_Y_sc = sc.fit_transform(data_Y)

    return data_X_sc, data_Y_sc, data.shape[0], data


st.header('Обучение модели ближайших соседей')

data_load_state = st.text('Загрузка данных...')
data_X, data_Y, data_len, data  = load_data()
data_load_state.text('Данные загружены!')

st.write(data.head())

cv_slider = st.slider('Количество фолдов:', min_value=3, max_value=10, value=5, step=1)

#Вычислим количество возможных ближайших соседей
rows_in_one_fold = int(data_len / cv_slider)
allowed_knn = int(rows_in_one_fold * (cv_slider-1))
st.write('Количество строк в наборе данных - {}'.format(data_len))
st.subheader('Метод ближайших соседей')
st.write('Максимальное допустимое количество ближайших соседей с учетом выбранного количества фолдов - {}'.format(allowed_knn))

cv_knn = st.slider('Количество ближайших соседей:', min_value=1, max_value=allowed_knn, value=5, step=1)


scores = cross_val_score(KNeighborsRegressor(n_neighbors=cv_knn),
    data_X, data_Y, scoring='neg_mean_squared_error', cv=cv_slider)


st.subheader('Оценка качества модели метода ближайших соседей')
st.write('Значения neg_mean_squared_error для отдельных фолдов')
st.bar_chart(scores)
st.write('Усредненное значение neg_mean_squared_error по всем фолдам - {}'.format(-np.mean(scores)))

st.subheader('Деревья решений')
cv_max_depth = st.slider('Максимальная гоубина:', min_value=3, max_value=6, value=3, step=1)
cv_min_samples_leaf = st.slider('Минимальное количество листьев:', min_value=0.04, max_value=0.08, value=0.04, step=0.02)


scores1 = cross_val_score(DecisionTreeRegressor(random_state=1, max_depth=cv_max_depth, min_samples_leaf=cv_min_samples_leaf),
    data_X, data_Y, scoring='neg_mean_squared_error', cv=cv_slider)


st.subheader('Оценка качества модели деревья решений')
st.write('Значения neg_mean_squared_error для отдельных фолдов')
st.bar_chart(scores1)
st.write('Усредненное значение neg_mean_squared_error по всем фолдам - {}'.format(-np.mean(scores)))
