import os
import sys
import numpy as np
import pandas as pd
import pickle

def load_st_dataset(dataset):
    print(f"-_+_+_+_+_+_+_+_+_+_+_+_+_+{dataset}+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+")
    #output B, N, D
    if 'DGEV_order_100' in dataset:
        df = pd.read_csv('../EVData/DGEV_v1208/100_order_series_matrix_filtered.csv', header=None)
        time_data = df.iloc[0, 1:].to_numpy()
        id_data = df.iloc[1:, 0].to_numpy()
        data = df.iloc[1:, 1:].to_numpy(dtype=np.float64).T
        df = pd.read_csv('../EVData/DGEV_v1208/EVData_filtered.csv')
        first_dt, first_tp = df.loc[0, "dt"], df.loc[0, "time_period"]
        df = df[(df['dt'] == first_dt) & (df['time_period'] == first_tp)]
        mapping = dict(zip(df['station_one_id'].values, df['fast_onli_connector_id'].values))
        capacity = np.array([mapping.get(i, 0) for i in id_data])
    elif 'DGEV_kt_100' in dataset:
        df = pd.read_csv('../EVData/DGEV_v1208/100_kt_series_matrix_filtered.csv', header=None)
        time_data = df.iloc[0, 1:].to_numpy()
        id_data = df.iloc[1:, 0].to_numpy()
        data = df.iloc[1:, 1:].to_numpy(dtype=np.float64).T
        capacity = data.max(axis=0) - data.min(axis=0)
    elif 'BJEV_order_100' in dataset:
        df = pd.read_csv('../EVData/BJEV_v1208/100_order_series_matrix_filtered.csv', header=None)
        time_data = df.iloc[0, 1:].to_numpy()
        id_data = df.iloc[1:, 0].to_numpy()
        data = df.iloc[1:, 1:].to_numpy(dtype=np.float64).T
        df = pd.read_csv('../EVData/BJEV_v1208/EVData_filtered.csv')
        first_dt, first_tp = df.loc[0, "dt"], df.loc[0, "time_period"]
        df = df[(df['dt'] == first_dt) & (df['time_period'] == first_tp)]
        mapping = dict(zip(df['station_one_id'].values, df['fast_onli_connector_id'].values))
        capacity = np.array([mapping.get(i, 0) for i in id_data])
    elif 'BJEV_kt_100' in dataset:
        df = pd.read_csv('../EVData/BJEV_v1208/100_kt_series_matrix_filtered.csv', header=None)
        time_data = df.iloc[0, 1:].to_numpy()
        id_data = df.iloc[1:, 0].to_numpy()
        data = df.iloc[1:, 1:].to_numpy(dtype=np.float64).T
        capacity = data.max(axis=0) - data.min(axis=0)
    elif 'HZEV_order_100' in dataset:
        df = pd.read_csv('../EVData/HZEV_v1208/100_order_series_matrix_filtered.csv', header=None)
        time_data = df.iloc[0, 1:].to_numpy()
        id_data = df.iloc[1:, 0].to_numpy()
        data = df.iloc[1:, 1:].to_numpy(dtype=np.float64).T
        df = pd.read_csv('../EVData/HZEV_v1208/EVData_filtered.csv')
        first_dt, first_tp = df.loc[0, "dt"], df.loc[0, "time_period"]
        df = df[(df['dt'] == first_dt) & (df['time_period'] == first_tp)]
        mapping = dict(zip(df['station_one_id'].values, df['fast_onli_connector_id'].values))
        capacity = np.array([mapping.get(i, 0) for i in id_data])
    elif 'HZEV_kt_100' in dataset:
        df = pd.read_csv('../EVData/HZEV_v1208/100_kt_series_matrix_filtered.csv', header=None)
        time_data = df.iloc[0, 1:].to_numpy()
        id_data = df.iloc[1:, 0].to_numpy()
        data = df.iloc[1:, 1:].to_numpy(dtype=np.float64).T
        capacity = data.max(axis=0) - data.min(axis=0)
    elif 'DGEV_order' in dataset:
        df = pd.read_csv('../EVData/DGEV_v1208/order_series_matrix_filtered.csv', header=None)
        time_data = df.iloc[0, 1:].to_numpy()
        id_data = df.iloc[1:, 0].to_numpy()
        data = df.iloc[1:, 1:].to_numpy(dtype=np.float64).T
    elif 'DGEV_kt' in dataset:
        df = pd.read_csv('../EVData/DGEV_v1208/kt_series_matrix_filtered.csv', header=None)
        time_data = df.iloc[0, 1:].to_numpy()
        id_data = df.iloc[1:, 0].to_numpy()
        data = df.iloc[1:, 1:].to_numpy(dtype=np.float64).T
    elif 'BJEV_order' in dataset:
        df = pd.read_csv('../EVData/BJEV_v1208/order_series_matrix_filtered.csv', header=None)
        time_data = df.iloc[0, 1:].to_numpy()
        id_data = df.iloc[1:, 0].to_numpy()
        data = df.iloc[1:, 1:].to_numpy(dtype=np.float64).T
    elif 'BJEV_kt' in dataset:
        df = pd.read_csv('../EVData/BJEV_v1208/kt_series_matrix_filtered.csv', header=None)
        time_data = df.iloc[0, 1:].to_numpy()
        id_data = df.iloc[1:, 0].to_numpy()
        data = df.iloc[1:, 1:].to_numpy(dtype=np.float64).T
    elif 'HZEV_order' in dataset:
        df = pd.read_csv('../EVData/HZEV_v1208/order_series_matrix_filtered.csv', header=None)
        time_data = df.iloc[0, 1:].to_numpy()
        id_data = df.iloc[1:, 0].to_numpy()
        data = df.iloc[1:, 1:].to_numpy(dtype=np.float64).T
    elif 'HZEV_kt' in dataset:
        df = pd.read_csv('../EVData/HZEV_v1208/kt_series_matrix_filtered.csv', header=None)
        time_data = df.iloc[0, 1:].to_numpy()
        id_data = df.iloc[1:, 0].to_numpy()
        data = df.iloc[1:, 1:].to_numpy(dtype=np.float64).T
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data.astype(np.float32), capacity.astype(np.float32)
