import pandas as pd
import shutil
import os


data = pd.read_excel("./raw/raw data(원본).xlsx", header=2)

vitalfiles1 = [file for file in os.listdir("./raw/SNUH/train/")]
vitalfiles2 = [file for file in os.listdir("./raw/SNUH/test/")]
vitalfiles = vitalfiles1 + vitalfiles2
filtered_df = data[data['fileid'].isin(vitalfiles)]
print(len(filtered_df))
sorted_df = filtered_df.sort_values(by="orin")
sorted_df['date'] = pd.to_datetime(sorted_df['orin'], errors='coerce')
sorted_df['dataset'] = sorted_df['date'].apply(lambda x: 'train' if x < pd.Timestamp('2020-07-01') else 'test')
sorted_df = sorted_df.drop(columns=['date'])
sorted_df.to_csv("./raw/SNUH_final_vitaldb.csv", index=False)



#파일 옮기기
# os.makedirs("./raw/SNUH/train/")
# os.makedirs("./raw/SNUH/test/")

# source_folder = "./raw/SNUH/"
# destination_folder = source_folder + "train/"
# for fileid in until_june_2020_count.index:
#     # 파일 경로 생성
#     source_path = os.path.join(source_folder, fileid)
#     destination_path = os.path.join(destination_folder, fileid)
    
#     shutil.move(source_path, destination_path)

# destination_folder = source_folder + "test/"
# for fileid in from_july_2020_count.index:
#     # 파일 경로 생성
#     source_path = os.path.join(source_folder, fileid)
#     destination_path = os.path.join(destination_folder, fileid)
    
#     shutil.move(source_path, destination_path)




