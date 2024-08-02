import glob
from collections import Counter
import vitaldb
from multiprocessing import Pool
import csv

# Function to process each file and return a Counter of trks
def process_file(data_file_path):
    vf = vitaldb.VitalFile(data_file_path)
    trks_list = vitaldb.vital_trks(data_file_path)
    return Counter(trks_list)

# Function to merge multiple Counter objects into one
def merge_counters(counter_list):
    total_counter = Counter()
    for counter in counter_list:
        total_counter.update(counter)
    return total_counter


# 입력
hospital_name = "SNUH" 
data_file_list = glob.glob(f"./raw/{hospital_name}/*.vital")

# Use multiprocessing to process files in parallel
with Pool() as pool:
    counters = pool.map(process_file, data_file_list)

# Merge all Counter objects into one
count_dictionary = merge_counters(counters)

# Write the results to a CSV file
with open(f'{hospital_name}_trk_counts.csv', 'w', newline='') as csvfile:
    fieldnames = ['Parameter', 'Count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for trk, count in count_dictionary.items():
        writer.writerow({'Parameter': trk, 'Count': count})


