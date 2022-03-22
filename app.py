import streamlit as st
import pandas as pd
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import math
import logging
import subprocess
import sys
from matplotlib.ticker import PercentFormatter
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.stats.outliers_influence
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn import metrics
import joblib
import pickle


def main():
    st.header("Cell Culture Calculator")
    values_to_predict = ['Viability (%)', 'Time (days)', 'Glucose (gLˆ-1)', 'pH', 'Temperature (°C)']
    predict_value = st.sidebar.selectbox(
        "Choose the variable you want to predict", values_to_predict)

    input_variables = [el for el in values_to_predict if el != predict_value]
    variables_dict = {}
    for var in input_variables:
        if var == 'Viability (%)':
            viability_values = st.text_input(var)
            variables_dict['Viability'] = viability_values
        elif var == 'Time (days)':
            time_values = st.text_input(var)
            variables_dict['Time'] = time_values
        elif var == 'Glucose (gLˆ-1)':
            glucose_values = st.text_input(var)
            variables_dict['Glucose'] = glucose_values
        elif var == 'pH':
            pH_values = st.text_input(var)
            variables_dict['pH'] = pH_values
        elif var == 'Temperature (°C)':
            sucrose_values = st.text_input(var)
            variables_dict['Temperature'] = sucrose_values

    if(st.button("Calculate")):
        if predict_value == 'Viability (%)':
            X = pd.DataFrame([variables_dict])
            st.subheader("Input values: ")
            st.dataframe(X)
            st.subheader("Viability calculated: " + str(95) + "%")
        
        elif predict_value == 'Time (days)':
            X = pd.DataFrame([variables_dict])
            st.subheader("Input values: ")
            st.dataframe(X)
            st.subheader("Time calculated: " + str(7) + " days")
        
        elif predict_value == 'Glucose (gLˆ-1)':
            X = pd.DataFrame([variables_dict])
            st.subheader("Input values: ")
            st.dataframe(X)
            st.subheader("Glucose calculated: " + str(2.00) + " gLˆ-1")
        
        elif predict_value == 'pH':
            X = pd.DataFrame([variables_dict])
            st.subheader("Input values: ")
            st.dataframe(X)
            st.subheader("pH calculated: " + str(5.35))

        elif predict_value == 'Temperature (°C)':
            X = pd.DataFrame([variables_dict])
            st.subheader("Input values: ")
            st.dataframe(X)
            st.subheader("Temperature calculated: " + str(26) + " (°C)")

if __name__ == "__main__":
    main()

# if predict_value == 'PCV':
# X = pd.DataFrame([variables_dict])
# pcv_scaler = joblib.load('pcv_scaler.bin')
# scaling_cols = ['Time', 'Conductivity', 'pH',
#                 'Glucose', 'Fructose', 'Viability', 'Sucrose']
# X_scaled = pcv_scaler.transform(X[scaling_cols])
# scaled_df = pd.DataFrame(data=X_scaled, columns=scaling_cols)
# pcv_pcaer = pickle.load(open("pcv_pca.pkl", 'rb'))
# pca_cols = ['Time', 'Conductivity',
#             'pH', 'Glucose', 'Fructose', 'Sucrose']
# pcaed_components = pcv_pcaer.transform(scaled_df[pca_cols])
# pcaed_df = pd.DataFrame(data=pcaed_components, columns=pca_cols)
# final_scaled_cols = [pcaed_df['Time'],
#                      pcaed_df['Conductivity'], pcaed_df['pH'],
#                      pcaed_df['Glucose'], pcaed_df['Fructose'],
#                      scaled_df['Viability'], pcaed_df['Sucrose']]
# final_scaled_df = pd.concat(final_scaled_cols, axis=1)
# regr = pickle.load(open("pcv_linear_rgr.pkl", 'rb'))
# y_pred = regr.predict(final_scaled_df)
# st.subheader("Input values: ")
# st.dataframe(X)
# st.subheader("PCV calculated: " + str(round(y_pred[0], 2)))