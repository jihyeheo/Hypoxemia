
import numpy as np
import pandas as pd



snuh = pd.read_csv("./data/SNUH_final_vitaldb.csv")
cnuh = pd.read_csv("./data/CNUH_final_vitaldb.csv", encoding="cp949")

# print(snuh.shape, snuh.columns)
# print(cnuh.shape, cnuh.columns)


snuh_need = snuh[["fileid", "sex", "age", "weight", "height"]]
cnuh_need = cnuh[["filename", " Patient Sex", "나이", "몸무게", "키"]]
cnuh_need.columns = ["fileid", "sex", "age", "weight", "height"]
snuh_need['sex'] = snuh_need['sex'].map({'M': 0, 'F': 1})
cnuh_need['age'] = cnuh_need['age'].str[:-1].astype(float)
cnuh_need['sex'] = cnuh_need['sex'].map({'M': 0, 'F': 1})

data = pd.concat([snuh_need, cnuh_need], ignore_index=True)
print(data)

data.to_csv("./data/data_final_vitaldb.csv", index=False)


print(snuh_need.head(3))
print(cnuh_need.head(3))