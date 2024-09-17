import pandas as pd
import vitaldb
import glob
import numpy as np
from multiprocessing import Pool
import neurokit2 as nk
import warnings
import yaml
from pathlib import Path
import pickle
import os
from datetime import datetime

warnings.filterwarnings("ignore")
sampling_rate = 500
sampling_rate_after = 100



def process_file(vital_path):
    try:
        vital_path = Path(vital_path)
        category = vital_path.parent.name
        dst_path = str((Path("./processed") / category / hospital_name / vital_path.name).with_suffix(".pkl"))
        # if os.path.exists(dst_path):
        #     return

        # Track
        trk = vitaldb.vital_trks(str(vital_path))
        single_sensor_list, single_factor_list = [], []
        wave_sensor_list, wave_factor_list = [], []

        for measurement in variables:
            for sensor_name, cali_factor in measurement:
                if sensor_name in trk:
                    single_sensor_list.append(sensor_name)
                    single_factor_list.append(cali_factor)
                    break

        for measurement in waveforms:
            for wave_name, cali_factor in measurement:
                if wave_name in trk:
                    wave_sensor_list.append(wave_name)
                    wave_factor_list.append(cali_factor)
                    break

        # 개수 다르면 안쓰기로
        if len(single_sensor_list) != len(variables):
            print("Not enough sensors for single measurement!, ", vital_path.name)
            return
        if len(wave_sensor_list) != len(waveforms):
            print("Not enough sensors for waveform data!, ", vital_path.name)
            return

        single_sensor_list.insert(0, "EVENT")

        # vital load & crop
        opstart, opend = get_timestamp(meta_info.loc[vital_path.name]["opstart"]), get_timestamp(meta_info.loc[vital_path.name]["opend"])
        meta = meta_info.loc[vital_path.name, ["sex", "age", "weight", "height"]].astype(float).values
        vf = vitaldb.VitalFile(str(vital_path))


        single_df = vf.to_pandas(
            ",".join(single_sensor_list), interval=1, return_datetime=True
        )
        wave_df = vf.to_pandas(
            ",".join(wave_sensor_list), interval=1 / sampling_rate, return_datetime=True
        )

        # time index 맞춰주기
        single_df.index = pd.to_datetime(single_df["Time"].dt.tz_localize(None).dt.floor("S"))
        wave_df.index = pd.to_datetime(wave_df["Time"].dt.tz_localize(None))

        ## 각 channel별로 time으로 할당
        for waveform in wave_sensor_list:
            single_df[waveform] = (
                wave_df[waveform]
                .resample("S")
                .apply(lambda x: np.array(x).flatten().tolist())
            )
        del wave_df

        opstart = single_df.index.get_loc(opstart)
        opend = single_df.index.get_loc(opend) + 1

        # time to range index
        data_df = single_df.reset_index(drop=True)

        # single measurements
        single_sensor_list.remove("EVENT")
        single = pd.DataFrame(data_df[single_sensor_list]).applymap(
            lambda x: float(x) if isinstance(x, (int, float, str)) else np.nan
        )
        single = single.ffill().bfill().to_numpy()
        single = single * np.array(single_factor_list)  # Adjust unit

        # # EVENT LABELING
        data_df["Label"] = 0
        start_indices = data_df[data_df["EVENT"] == "시작"].index
        end_indices = data_df[data_df["EVENT"] == "끝"].index

        for start_idx, end_idx in zip(start_indices, end_indices):
            data_df.loc[start_idx : end_idx, "Label"] = (single[start_idx : end_idx + 1, 0] < 95).astype(int)


        first_index = np.where(
            np.any(pd.isna(data_df[single_sensor_list].ffill()), axis=1) == False
        )[0][0]
        last_index = (
            np.where(np.any(pd.isna(data_df[single_sensor_list].bfill()), axis=1) == True)[
                0
            ][0]
            + 1
        )

        # Trim
        data_df = data_df.iloc[max(opstart, first_index): min(opend, last_index)]
        single = single[max(opstart, first_index): min(opend, last_index)]

        if len(np.concatenate(data_df[wave_sensor_list[0]].values)) < 1000:
            print("Too short, ", vital_path.name)
            return

        # Wave
        recs_ppg = np.concatenate(data_df[wave_sensor_list[0]].values)
        recs_ppg = np.array(pd.Series(recs_ppg).ffill().bfill(), dtype=float) * np.array(
            wave_factor_list[0]
        )
        recs_awp = np.concatenate(data_df[wave_sensor_list[1]].values)
        recs_awp = np.array(pd.Series(recs_awp).ffill().bfill(), dtype=float) * np.array(
            wave_factor_list[1]
        )
        recs_co2 = np.concatenate(data_df[wave_sensor_list[2]].values)
        recs_co2 = np.array(pd.Series(recs_co2).ffill().bfill(), dtype=float) * np.array(
            wave_factor_list[2]
        )
        recs_ecg = np.concatenate(data_df[wave_sensor_list[3]].values)
        recs_ecg = np.array(pd.Series(recs_ecg).ffill().bfill(), dtype=float) * np.array(
            wave_factor_list[3]
        )

        try:
            recs_ppg = nk.ppg_clean(recs_ppg, sampling_rate=sampling_rate)
            recs_ecg = nk.ecg_clean(recs_ecg, sampling_rate=sampling_rate)
        except:
            print("Failed to clean, ", vital_path.name)
            return

        wave = np.stack([recs_ppg, recs_awp, recs_co2, recs_ecg], axis=-1)  # n x 4
        wave = np.mean(
            wave.reshape(
                -1,
                int(sampling_rate / sampling_rate_after),
                wave.shape[1],
            ),
            axis=1,
        ).astype(np.float32)  # n x 4

        # Add waveforms
        label = np.kron(np.array(data_df["Label"]), np.ones(sampling_rate_after))

        data = {"wave": wave, "single": single, "label": label, "meta": meta}
        for v in data.values():
            if np.any(np.isnan(v)):
                print("NaN value detected, ", vital_path.name)
                return

        print(dst_path)
        with open(dst_path, "wb") as f:
            pickle.dump(data, f)
    except:
        return


def get_timestamp(time):
    time = str(time)
    if hospital_name == "CNUH":
        return datetime.strptime(time, "%Y-%m-%d %I:%M:%S %p")
    else:
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    for hospital_name in ["SNUH", "CNUH"]:
        with open(f"./{hospital_name}_variables.yaml", "r") as file:
            config = yaml.safe_load(file)

            path = config["path"]
            variables = config["variables"]
            waveforms = config["waveforms"]
            meta_info = pd.read_excel("hypoxemia_meta_info.xlsx", sheet_name=hospital_name, index_col=0)

            if hospital_name == "CNUH":
                meta_info.replace("오전", "AM", inplace=True, regex=True)
                meta_info.replace("오후", "PM", inplace=True, regex=True)

                str2year = {"Y" : 1, "M" : 1 / 12, "D" : 1 / 365}
                meta_info["age"] = meta_info["age"].map(lambda x: float(x[:-1]) * str2year[x[-1]])

            meta_info["sex"] = meta_info["sex"].map(lambda x : 0 if x == "M" else 1)

        with Pool() as pool:
            counters = pool.map(process_file, glob.glob(path))

        # for p in glob.glob(path):
        #     process_file(p)