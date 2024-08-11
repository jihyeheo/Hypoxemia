import pandas as pd
import shutil
import os


data = pd.read_excel("./raw/exclude_all(최종).xlsx")
data = data.dropna(subset=["나이", "키", "몸무게", " Patient Sex"]).reset_index(drop=True)
print(data.shape)
print(data.columns)
print(data["filename"])

real_data_list = os.listdir("./raw/CNUH/test2/")

num = []
for i in range(len(data)) :
    raw = data.loc[i]
    if raw["filename"] in real_data_list :
        num.append(raw) 
print(len(num))
data_final = pd.DataFrame(num).reset_index(drop=True)
data_final.columns = ['ID', 'NAME', '나이', '몸무게', '키', 'ASA', '병원구분', 'ROOM_CODE', '스케쥴_y',
       '스케쥴 시작', '스케쥴 종료', '마취타입', '마취 시작', '마취 종료', '부서코드', '부서명',
       'Export Path', ' Patient Sex', '년', '월', '일', 'filename']
print(data_final.shape)

for real_data_name in real_data_list :
    if len(data_final[data_final["filename"] == real_data_name]) >= 1 :
        shutil.move("./raw/CNUH/test2/"+real_data_name, "./raw/CNUH/test/"+real_data_name)

num = []
real_data_list = os.listdir("./raw/CNUH/test/")
for i in range(len(data_final)) :
    raw = data_final.loc[i]
    print(raw)
    if raw["filename"] in real_data_list :
        num.append(raw) 
print(len(num))
data_final = pd.DataFrame(num).reset_index(drop=True)
print(data_final.shape)
data_final.to_csv("CNUH_final_vitaldb.csv", index=False)







