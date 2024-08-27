import pandas as pd
import vitaldb
import glob
import numpy as np
from multiprocessing import Pool, freeze_support
import neurokit2 as nk
import os
import matplotlib.pyplot as plt
import warnings
import yaml

warnings.filterwarnings("ignore")
sampling_rate = 500
sampling_rate_after = 100
hospital_name = "SNUH"

with open(f'{hospital_name}_variables.yaml', 'r') as file:
    data = yaml.safe_load(file)

path = data["path"]
variables = data['variables']
waveforms = data["waveforms"]

def process_file(vital_path):
    category, vitalfile = vital_path.split("\\")[-2:]
    print(category, vitalfile)
    
    
    # CNUH 경로 다른 이슈 
    if hospital_name == "CNUH" :
        category = category[-4:]
        print(category, vitalfile)

    #multi-processing으로 이미 생성된 파일은 생성 안하기 위해
    # if os.path.isfile(f"./processed/waveform/{category}/{hospital_name}/" + vitalfile + ".npy") :
    #     return 
    
    # vital file에서 필요한 trks 가져오기
    trk = vitaldb.vital_trks(vital_path)
    single_sensor_list, single_factor_list = [], []
    wave_sensor_list, wave_factor_list = [], []

    for measurement in variables:
        for sensor_name, cali_factor in measurement:
            if sensor_name in trk:
                single_sensor_list.append(sensor_name)
                single_factor_list.append(cali_factor)
                break
    
    for measurement in waveforms:
        for wave_name, cali_factor in measurement :
            if wave_name in trk:
                wave_sensor_list.append(wave_name)
                wave_factor_list.append(cali_factor)
                break

    single_sensor_list.insert(0, "EVENT")


    # 개수 다르면 안쓰기로
    if len(single_sensor_list) != len(variables) + 1:
        return
    if len(wave_sensor_list) != len(waveforms) :
        return
    # vital load
    vf = vitaldb.VitalFile(vital_path)
    recs_df = vf.to_pandas(",".join(single_sensor_list), interval=1, return_datetime=True)
    wave_df = vf.to_pandas(",".join(wave_sensor_list), interval=1 / sampling_rate, return_datetime=True)
    
    # time index 맞춰주기
    recs_df.index = recs_df["Time"].dt.floor("S")
    wave_df.index = wave_df["Time"]

    ## 각 channel별로 time으로 할당
    for waveform in wave_sensor_list :
        recs_df[waveform] = wave_df[waveform].resample("S").apply(lambda x: np.array(x).flatten().tolist())


    # time to range index
    data_df = recs_df.reset_index(drop=True)
 
    # # EVENT LABELING
    # ### 1) beak : ['95':'end'] & <95

    data_df["Label"] = 0
    start_indices = data_df[data_df["EVENT"] == "시작"].index
    end_indices = data_df[data_df["EVENT"] == "끝"].index

    if len(start_indices) == 0 or len(end_indices) == 0:
        print(vital_path)
        return
    for start_idx, end_idx in zip(start_indices, end_indices):
        for ii in range(start_idx, end_idx + 1):
            label = 1 if data_df.loc[ii, single_sensor_list[1]] < 95 else 0  # spo2
            data_df.loc[ii, "Label"] = label


    # Label이 EVENT 이므로
    single_sensor_list.remove("EVENT")
    
    first_index, last_index = 0, len(data_df)
    first_index = np.where(np.any(pd.isna(data_df[single_sensor_list].ffill()), axis=1) == False)[0][0]
    last_index = np.where(np.any(pd.isna(data_df[single_sensor_list].bfill()), axis=1) == True)[0][0] + 1
    
    # Trim
    data_df = data_df.iloc[first_index : last_index]
    data_df = data_df.iloc[: len(data_df) // 2 * 2]
    #print(data_df)

    # recs(11 variables), waveforms, label
    recs = np.array(data_df[single_sensor_list])  # 11 variables (n,11)
    label = np.array(data_df["Label"])  # (n,) 

    #print(recs.shape, label.shape)
    ## 길이 너무 작으면 ppg_clean 안될까봐
    if len(data_df[wave_sensor_list[0]]) < 1000 :
        print(vital_path)
        return 
    #try : 
    # clean & downsampling .value
    def flatten(series) :
        return [item for sublist in series for item in sublist]
    

    recs_ppg = flatten(data_df[wave_sensor_list[0]])
    recs_ppg = np.array(pd.Series(recs_ppg).ffill().bfill(), dtype=float) * np.array(wave_factor_list[0])
    recs_awp = flatten(data_df[wave_sensor_list[1]])
    recs_awp = np.array(pd.Series(recs_awp).ffill().bfill(), dtype=float) * np.array(wave_factor_list[1])
    recs_co2 = flatten(data_df[wave_sensor_list[2]])
    recs_co2 = np.array(pd.Series(recs_co2).ffill().bfill(), dtype=float) * np.array(wave_factor_list[2])

    
    # 전체적으로 다 nan인 경우 
    if np.isnan(recs_ppg).sum() == recs_ppg.shape[0] :
        return
    if np.isnan(recs_ppg).sum() >= 1 or np.isnan(recs_awp).sum() >= 1 or np.isnan(recs_co2).sum() >= 1:
        return


    recs_ppg = nk.ppg_clean(recs_ppg.reshape(-1,), sampling_rate=sampling_rate) # (n,500)
    recs_ppg = np.mean(recs_ppg.reshape(-1, sampling_rate_after, int(sampling_rate/sampling_rate_after)), axis=2) # (n,100)
    recs_awp = np.mean(recs_awp.reshape(-1, sampling_rate_after, int(sampling_rate/sampling_rate_after)), axis=2) # (n,100)
    recs_co2 = np.mean(recs_co2.reshape(-1, sampling_rate_after, int(sampling_rate/sampling_rate_after)), axis=2) # (n,100)

    # Add single measurements
    recs = np.array(pd.DataFrame(recs).ffill().bfill())
    recs = recs * np.array(single_factor_list)  # Adjust unit
    recs = np.concatenate([recs, label[:, np.newaxis]], axis=1)  # Concat (n,12)

    # Add waveforms
    recs_waveforms = np.stack([recs_ppg, recs_awp, recs_co2], axis=-1) # n x 100 x 3
    spo2_included = np.tile(recs[:,0][:, np.newaxis, np.newaxis], (1,sampling_rate_after, 1))
    label_included = np.tile(recs[:,-1][:, np.newaxis, np.newaxis], (1,sampling_rate_after,1))
    recs_waveforms = np.concatenate([recs_waveforms, spo2_included, label_included], axis=-1) # n x 100 x 5

    recs_waveforms = recs_waveforms.reshape(-1, recs_waveforms.shape[2]).astype(np.float32) # (n x 100) x 5
    #print(recs.shape, recs_waveforms.shape)
    
    recs = recs[::2]  # Resample for every 2 seconds
    assert recs.shape[0] * sampling_rate_after * 2 == recs_waveforms.shape[0] 
    np.save(f"./processed/waveform/{category}/{hospital_name}1/" + vitalfile + ".npy", recs_waveforms)
    np.save(f"./processed/single/{category}/{hospital_name}1/" + vitalfile + ".npy", recs)


if __name__ == "__main__" :
    freeze_support()
    with Pool() as pool:
        counters = pool.map(process_file, glob.glob(path))

    #process_file("./raw/SNUH/train/P10_190703_125228.vital")
    # for idx, vital_path in enumerate(glob.glob(path)):
    #     print(vital_path)
    #     break

