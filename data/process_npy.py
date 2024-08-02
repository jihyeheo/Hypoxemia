import pandas as pd
import vitaldb
import glob
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from multiprocessing import Pool
import neurokit2 as nk
import scipy
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")



sampling_rate = 125
hospital_name = "SNUH"


if hospital_name == "SNUH" :
    path = f"./raw/{hospital_name}/*/*.vital"
    variables = [  # [sensor name, calibration factor] n = 11
        [["Intellivue/PLETH_SAT_O2", 1.0], ["Solar8000/PLETH_SPO2", 1.0], ["Root/SPO2", 1.0]],  # O2 sat, same
        [["Intellivue/PLETH_HR", 1.0], ["Solar8000/PLETH_HR", 1.0]],  # HR, same
        [["Primus/ETCO2", 1.0], ["Datex-Ohmeda/ETCO2", 7.5], ["Solar8000/ETCO2", 1.0]],  # ETCO2
        [["Primus/FIO2", 1.0], ["Datex-Ohmeda/FIO2", 1.0], ["Solar8000/FIO2", 1.0]],  # FIO2, same
        [["Primus/TV", 1.0], ["Datex-Ohmeda/TV_EXP", 1.0], ["Solar8000/VENT_TV", 1.0]],  # TV, same
        [["Primus/SET_PIP", 0.98067], ["Datex-Ohmeda/SET_PINSP", 1.], ["Solar8000/VENT_SET_PIP", 1.]], # SET_PIP
        [["Primus/PIP_MBAR", 1.0], ["Datex-Ohmeda/PIP", 1.01972], ["Solar8000/VENT_PIP", 1.0]],  # PIP
        [["Primus/PEEP_MBAR", 1.0], ["Datex-Ohmeda/SET_PEEP", 1.01972], ["Solar8000/VENT_PEEP", 1.0]],  # PEEP
        [["Primus/MV", 1.0], ["Datex-Ohmeda/MV_EXP", 1.0], ["Solar8000/VENT_MV", 1.0]],  # MV, same
        [["Solar8000/NIBP_SBP", 1.0], ["Intellivue/NIBP_SYS", 1.0]],  # SBP, same
        [["Solar8000/NIBP_DBP", 1.0], ["Intellivue/NIBP_DIA", 1.0]],  # DBP, same
    ]
    ppg_list = ["SNUADC/PLETH", "Intellivue/PLETH"]

else : 
    path = f"./raw/{hospital_name}/*.vital"
    variables = [  # [sensor name, calibration factor] n = 10
        [["Intellivue/PLETH_SAT_O2", 1.0], ["X002/SPO2", 1.0], ["Radical7/SPO2", 1.0], ["Root/SPO2", 1.0], ["Bx50/PLETH_SPO2", 1.0]],  # O2 sat, same
        [["Intellivue/PLETH_HR", 1.0], ["Bx50/PLETH_HR", 1.0]],  # HR, 애매함. 
        [["Primus/ETCO2", 1.0], ["Datex-Ohmeda/ETCO2", 7.5], ["Flow-i/ETCO2", 1.0]],  # ETCO2
        [["Primus/FIO2", 1.0], ["Datex-Ohmeda/FIO2", 1.0], ["Flow-i/FIO2", 1.0]],  # FIO2, same
        [["Primus/TV", 1.0], ["Datex-Ohmeda/TV_EXP", 1.0], ["Flow-i/TV_EXP", 1.0]],  # TV, same
        [["Primus/SET_PIP", 0.98067], ["Datex-Ohmeda/SET_PINSP", 1.]], # SET_PIP
        [["Primus/PIP_MBAR", 1.0], ["Datex-Ohmeda/PIP", 1.01972], ["Flow-i/PIP", 1.01972]],  # PIP
        [["Primus/PEEP_MBAR", 1.0], ["Datex-Ohmeda/SET_PEEP", 1.01972]],  # PEEP
        [["Primus/MV", 1.0], ["Datex-Ohmeda/MV_EXP", 1.0], ["Flow-i/MV_EXP", 1.0]],  # MV, 애매함 다시 확인.
        [["Intellivue/NIBP_SYS", 1.0], ["Bx50/NIBP_SBP", 1.0]],  # SBP, same
        [["Intellivue/NIBP_DIA", 1.0], ["Bx50/NIBP_DBP", 1.0]],  # DBP, same
    ]
    ppg_list = ["Intellivue/PLETH", "X002/PLETH", "Bx50/PLETH"]



def process_file(vital_path):
    trk = vitaldb.vital_trks(vital_path)
    sensor_list = []
    factor_list = []

    for measurement in variables:
        for sensor_name, cali_factor in measurement:
            if sensor_name in trk:
                sensor_list.append(sensor_name)
                factor_list.append(cali_factor)
                break

    for ppg_name in ppg_list:
        if ppg_name in trk:
            break

    if len(sensor_list) != len(variables):
        return

    sensor_list.insert(0, "EVENT")

    # vital load
    vf = vitaldb.VitalFile(vital_path)
    recs_df = vf.to_pandas(",".join(sensor_list), 
                      interval=1,
                      return_datetime=True)

    ppgsqi_df = vf.to_pandas(ppg_name, 
                      interval=1/sampling_rate,
                      return_datetime=True)

    print(recs_df.shape, ppgsqi_df.shape)
    # time index
    recs_df.index = recs_df["Time"].dt.floor("S")
    ppgsqi_df.index = ppgsqi_df["Time"]

    
    ppgsqi_df = ppgsqi_df.resample("S").apply(lambda x:x.tolist())
    ppgsqi_df = ppgsqi_df.reindex(recs_df.index)
    
    recs_df[ppg_name] = ppgsqi_df[ppg_name] 

    # time to range index
    data_df = recs_df.reset_index(drop=True)


    # # EVENT LABELING
    # ### 1) beak : ['95':'end'] & <95

    data_df["Label"] = 0
    start_indices = data_df[data_df["EVENT"] == "시작"].index
    end_indices = data_df[data_df["EVENT"] == "끝"].index

    for start_idx, end_idx in zip(start_indices, end_indices):
        for ii in range(start_idx, end_idx +1) :
            label = 1 if data_df.loc[ii, sensor_list[1]] < 95 else 0 # spo2
            #print(df.loc[ii, "Intellivue/PLETH_SAT_O2"])
            data_df.loc[ii, 'Label'] = label
    

    sensor_list.remove("EVENT")
    first_index = np.where(np.any(pd.isna(data_df[sensor_list].ffill()), axis=1) == False)[0][0]
    last_index = np.where(np.any(pd.isna(data_df[sensor_list].bfill()), axis=1) == True)[0][0]

    # Trim
    data_df = data_df.iloc[first_index:last_index+1]
    
    # recs(11 variables), ppgsqi(ppg convert ppgsqi), label
    recs = np.array(data_df[sensor_list]) # 11 variables
    label = np.array(data_df["Label"])
    ppgsqi = np.array(data_df[ppg_name].tolist()).reshape(-1,).astype(float)
    ppgsqi = nk.ppg_clean(ppgsqi, sampling_rate=sampling_rate) # clean
    ppgsqi = nk.ppg_quality(ppgsqi, sampling_rate=sampling_rate) # sqi
    ppgsqi = np.mean(ppgsqi.reshape(-1, sampling_rate), axis=1) # mean
    
     
    # Add all variables
    recs = recs * np.array(factor_list)  # Adjust unit
    recs = np.concatenate([np.array(recs), np.array(ppgsqi).reshape(-1, 1), np.array(label).reshape(-1, 1)], axis=1) # Concate
    recs = recs[::2]  # Resample for every 2 seconds
    print(recs.shape)

    vitalfile = vital_path.split("/")[-1]
    if hospital_name == "SNUH" :
        category = vital_path.split("/")[-2]
        if not os.path.isdir(f"./processed/{hospital_name}/{category}/"):
            os.makedirs(f"./processed/{hospital_name}/{category}/")
        np.save(f"./processed/{hospital_name}/{category}/" + vitalfile + ".npy", recs)
    else :
        np.save(f"./processed/{hospital_name}/" + vitalfile + ".npy", recs)



with Pool() as pool:
    counters = pool.map(process_file, os.listdir(path))

# for vital_path in glob.glob(path) :
#     process_file(vital_path)
#     break
    

# plt.plot(recs[:, 0])
        # colors = ["red","blue"]
        # for i in range(len(start_indices)):
        #     plt.axvline(start_indices[i], color= colors[i], linestyle="--")
        #     plt.axvline(end95_indices[i], color= colors[i], linestyle="--")
        #     #plt.axvline(end_indices[i], color="red", linestyle="--")
        # plt.savefig("spo2.png")