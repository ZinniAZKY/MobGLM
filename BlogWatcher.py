import pandas as pd
import re
import os

from sklearn.cluster import DBSCAN
from skmob import TrajDataFrame
from skmob.preprocessing import clustering

# nrows = 1000
# # Read the CSV file
# df = pd.read_csv('/mnt/large/data/BlogWatcher/Tokyo_202307_202308/log_000000000000.csv')
#
# # Convert the 'datetime' column to datetime format
# df['datetime'] = pd.to_datetime(df['datetime'])
#
# # Sort the DataFrame by 'hashed_adid' and 'datetime'
# df_sorted = df.sort_values(by=['hashed_adid', 'datetime'])
#
# # Define the target date
# target_date = '2023-07-25'
#
# # Filter the DataFrame for rows where the date is '2023-07-25'
# filtered_df = df_sorted[df_sorted['datetime'].dt.date == pd.to_datetime(target_date).date()]
# filtered_df.to_csv('/home/zhangky/Documents/ZhangKY/BlogWatcher/2023-07-25.csv')
# print(filtered_df)


# def process_file(input_file, output_file):
#     with open(input_file, 'r') as file:
#         lines = file.readlines()
#
#     # Remove blank rows and replace "bus" and "taxi" tokens
#     processed_sentences = []
#     for line in lines:
#         stripped_line = line.strip()
#         if stripped_line:  # Only process non-blank lines
#             tokens = stripped_line.split()  # Split the line into tokens
#             modified_tokens = [
#                 "Bus" if token == "bus" else "Taxi" if token == "taxi" else token
#                 for token in tokens
#             ]
#             processed_sentences.append(" ".join(modified_tokens))  # Join tokens back into a sentence
#
#     # Write the processed sentences to the output file
#     with open(output_file, 'w') as file:
#         for sentence in processed_sentences:
#             file.write(sentence + "\n")
#
#
# def process_folder(input_folder, output_folder):
#     # Ensure the output folder exists
#     os.makedirs(output_folder, exist_ok=True)
#
#     # Loop through all .txt files in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".txt"):
#             input_file_path = os.path.join(input_folder, filename)
#             output_file_path = os.path.join(output_folder, filename)
#             process_file(input_file_path, output_file_path)
#             print(f"Processed {filename}")
#
#
# # Specify your input and output folder paths
# input_folder = "/home/zhangky/Downloads/Nationwide_Spetial/"
# output_folder = "/home/zhangky/Downloads/Nationwide_Hierarchical/"
#
# process_folder(input_folder, output_folder)

# folder_path = "/home/zhangky/Downloads/Nationwide_Hierarchical_Removed_72"
# expected_length = 723
#
#
# # Function to check the format of each row
# def check_format_and_length(filename):
#     correct_format = True
#     correct_length = True
#
#     with open(filename, 'r') as file:
#         for line_number, line in enumerate(file, start=1):
#             tokens = line.strip().split()
#             if len(tokens) != expected_length:
#                 correct_length = False
#                 print(f"Row {line_number} in {filename} does not have {expected_length} tokens.")
#
#             # Check if the 5th, 15th, 25th... tokens are "<region>"
#             for i in range(4, len(tokens), 10):
#                 if tokens[i] != "<region>":
#                     correct_format = False
#                     print(f"Row {line_number} in {filename} does not have '<region>' at position {i + 1}.")
#
#     return correct_format, correct_length
#
#
# # Iterate through all text files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith(".txt"):
#         file_path = os.path.join(folder_path, filename)
#         format_check, length_check = check_format_and_length(file_path)
#
#         if format_check and length_check:
#             print(f"{filename} is correctly formatted and all rows have {expected_length} tokens.")
#         else:
#             print(f"{filename} has formatting issues or incorrect row lengths.")

# import pandas as pd
# from datetime import datetime, timedelta
#
# file_path = '/mnt/large/data/BlogWatcher/Tokyo_202307_202308/BW_Tokyo_202307_202308.csv'
# output_dir = '/home/zhangky/Documents/ZhangKY/BlogWatcher/'
# start_date = datetime(2023, 7, 29)
# end_date = datetime(2023, 8, 4)
# chunk_size = 10 ** 7
#
# # Iterate through each day in the date range
# current_date = start_date
# while current_date <= end_date:
#     date_str = current_date.strftime('%Y-%m-%d')
#     output_file_path = f'{output_dir}{date_str}_All.csv'
#     chunk_count = 0
#
#     with open(output_file_path, 'w') as output_file:
#         for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, on_bad_lines='warn'):
#             chunk_count += 1
#             print(f'Processing chunk {chunk_count} for date {date_str}...')
#
#             # Convert the 'datetime' column to datetime format, invalid parsing will be set as NaT
#             chunk['datetime'] = pd.to_datetime(chunk['datetime'], errors='coerce')
#             chunk = chunk.dropna(subset=['datetime'])
#
#             # Filter rows where the datetime column matches the current date
#             filtered_chunk = chunk[chunk['datetime'].dt.strftime('%Y-%m-%d') == date_str]
#             filtered_chunk.to_csv(output_file, mode='a', index=False, header=output_file.tell() == 0)
#
#             print(f'Finished processing chunk {chunk_count} for date {date_str}, filtered {len(filtered_chunk)} rows.')
#
#     print(f'Data extraction complete for {date_str}. Filtered data saved to {output_file_path}.')
#
#     # Move to the next date
#     current_date += timedelta(days=1)
#
# print('All data extraction complete.')

# import os
# import pandas as pd
# from collections import defaultdict
#
# input_folder = '/home/zhangky/Documents/ZhangKY/BlogWatcher'
# output_folder = '/home/zhangky/Documents/ZhangKY/BlogWatcherResults'
# os.makedirs(output_folder, exist_ok=True)
#
# files_to_process = ['2023-07-22_All.csv', '2023-07-23_All.csv', '2023-07-24_All.csv',
#                     '2023-07-25_All.csv', '2023-07-26_All.csv', '2023-07-27_All.csv', '2023-07-28_All.csv']
#
#
# def extract_unique_id_data(files, batch_size=10000):
#     all_unique_ids = set()
#     id_occurrences = defaultdict(set)  # Keep track of which files each ID appears in
#
#     # First pass: Collect all unique IDs and track their occurrences in each file
#     for file in files:
#         input_path = os.path.join(input_folder, file)
#         df = pd.read_csv(input_path, usecols=['hashed_adid'])
#         unique_ids = df['hashed_adid'].unique()
#
#         # Update unique ID set and track occurrences
#         all_unique_ids.update(unique_ids)
#         for uid in unique_ids:
#             id_occurrences[uid].add(file)
#
#     # Check for IDs that appear in multiple files and print them
#     for uid, file_set in id_occurrences.items():
#         if len(file_set) > 1:
#             print(f"ID {uid} appears in multiple files: {file_set}")
#
#     all_unique_ids = list(all_unique_ids)
#
#     # Process unique IDs in batches
#     for i in range(0, len(all_unique_ids), batch_size):
#         batch_ids = all_unique_ids[i:i + batch_size]
#
#         batch_data = []
#
#         for file in files:
#             input_path = os.path.join(input_folder, file)
#             df = pd.read_csv(input_path)
#
#             # Filter the DataFrame for the current batch of IDs
#             filtered_df = df[df['hashed_adid'].isin(batch_ids)]
#             batch_data.append(filtered_df)
#
#         batch_df = pd.concat(batch_data)
#
#         # Output the batch to a new CSV file
#         output_path = os.path.join(output_folder, f'unique_ids_batch_{i // batch_size + 1}.csv')
#         batch_df.to_csv(output_path, index=False)
#         print(f"Saved batch {i // batch_size + 1} to {output_path}")
#
#
# # Call the function to process files and extract batches
# extract_unique_id_data(files_to_process)

# import os
# import pandas as pd
# from skmob import TrajDataFrame
# from skmob.preprocessing import detection
# from skmob.preprocessing import filtering
#
# input_folder = '/home/zhangky/Documents/ZhangKY/BlogWatcherResults'
# output_folder = '/home/zhangky/Documents/ZhangKY/BlogWatcherSP'
# os.makedirs(output_folder, exist_ok=True)
# csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
#
#
# # Function to process each file: load, filter, and detect stay points
# def process_file(input_path, output_path):
#     df = pd.read_csv(input_path)
#     tdf = TrajDataFrame(df, latitude='latitude_anonymous', longitude='longitude_anonymous', user_id='hashed_adid', datetime='datetime')
#     ftdf = filtering.filter(tdf, max_speed_kmh=300.)
#     stdf = detection.stay_locations(ftdf, stop_radius_factor=1, minutes_for_a_stop=5.0, spatial_radius_km=0.1,
#                                     leaving_time=True)
#
#     # Save the processed data with stay points to a new CSV
#     stdf.to_csv(output_path, index=False)
#     print(f"Processed and saved: {output_path}")
#
#
# # Loop over all files and process each one
# for file in csv_files:
#     input_path = os.path.join(input_folder, file)
#     output_path = os.path.join(output_folder, file.replace(".csv", "_stay_points.csv"))
#     process_file(input_path, output_path)
#
# print("All files processed and saved.")

# import os
# import pandas as pd
#
# input_folder = '/home/zhangky/Documents/ZhangKY/BlogWatcherSP'
# output_file = '/home/zhangky/Documents/ZhangKY/clustered_SP_0722-0728.csv'
#
# # List all CSV files in the stay points folder
# csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
#
#
# # Function to load and combine all stay point CSV files in chunks and change uid to numeric
# def combine_stay_points_in_chunks_with_uid_mapping(files, folder, output_file, chunk_size=25000):
#     # Dictionary to map long uid to numeric
#     uid_mapping = {}
#     uid_counter = 1
#
#     with open(output_file, 'w') as outfile:
#         first_file = True
#
#         for file in files:
#             input_path = os.path.join(folder, file)
#             print(f"Processing file: {file}")
#
#             # Read and process the file in chunks
#             for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
#                 # Map each unique uid to a numeric value
#                 for uid in chunk['uid'].unique():
#                     if uid not in uid_mapping:
#                         uid_mapping[uid] = uid_counter
#                         uid_counter += 1
#
#                 # Replace the uid column with the numeric mapping
#                 chunk['uid'] = chunk['uid'].map(uid_mapping)
#
#                 if first_file:
#                     # Write the header for the first file
#                     chunk.to_csv(outfile, index=False, mode='a', header=True)
#                     first_file = False
#                 else:
#                     # Append without writing the header for subsequent files
#                     chunk.to_csv(outfile, index=False, mode='a', header=False)
#
#                 # Print progress for each chunk
#                 print(f"  Processed chunk {i + 1} of {file}")
#
#     print(f"Finished concatenating all files into {output_file}")
#     print("UID mapping:", uid_mapping)
#
#
# # Call the function to combine the CSV files chunk-wise and map uid to numeric
# combine_stay_points_in_chunks_with_uid_mapping(csv_files, input_folder, output_file)

# import pandas as pd
# from skmob import TrajDataFrame
# from skmob.preprocessing import clustering
#
# combined_stay_points_file = '/home/zhangky/Documents/ZhangKY/clustered_SP_0722-0728.csv'
# df = pd.read_csv(combined_stay_points_file)
# df['datetime'] = pd.to_datetime(df['datetime'])
#
# # Filter data for July 22
# df_july22 = df[df['datetime'].dt.strftime('%Y-%m-%d') == '2023-07-22']
# print(f"Filtered data for 2023-07-22, number of rows: {len(df_july22)}")
#
# tdf_july22 = TrajDataFrame(df_july22, latitude='lat', longitude='lng', datetime='datetime', user_id='uid')
# clustered_tdf_july22 = clustering.cluster(tdf_july22, cluster_radius_km=0.01, min_samples=1)
# output_file = '/home/zhangky/Documents/ZhangKY/clustered_output_july22.csv'
# clustered_tdf_july22.to_csv(output_file, index=False)
# print(f"Clustering completed for July 22 and results saved to {output_file}")


# import pandas as pd
# from skmob import TrajDataFrame
# from skmob.preprocessing import clustering
#
# combined_stay_points_file = '/home/zhangky/Documents/ZhangKY/clustered_SP_0722-0728.csv'
# df = pd.read_csv(combined_stay_points_file)
# print('finished reading')
# tdf = TrajDataFrame(df, latitude='lat', longitude='lng', datetime='datetime', user_id='uid')
# clustered_tdf = clustering.cluster(tdf, cluster_radius_km=0.01, min_samples=1)
# output_file = '/home/zhangky/Documents/ZhangKY/clustered_output_weekly.csv'
# clustered_tdf.to_csv(output_file, index=False)
#
# print(f"Clustering completed and results saved to {output_file}")


# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the clustered file (only necessary columns to save memory)
# clustered_file = '/home/zhangky/Documents/ZhangKY/clustered_output_july22.csv'
#
# # Read only the cluster column to reduce memory usage
# df_clustered = pd.read_csv(clustered_file, usecols=['cluster'])
#
# # Use value_counts to get the frequency of each cluster
# cluster_counts = df_clustered['cluster'].value_counts().sort_index()
#
# # Plot the frequency of each cluster
# plt.figure(figsize=(10, 6))
# cluster_counts.plot(kind='bar')
# plt.title('Frequency of Each Cluster')
# plt.xlabel('Cluster ID')
# plt.ylabel('Frequency')
# plt.xticks(rotation=45)
# plt.tight_layout()
#
# # Show the plot
# plt.show()
#
# print("Cluster frequency calculation and plotting completed.")


# def replace_special_tokens(sentence):
#     special_tokens = ['<region>', '<prefecture>', '<municipality>', '<small_zone>']
#
#     # Replace each special token
#     for token in special_tokens:
#         sentence = sentence.replace(token, '<UNK>')
#
#     return sentence
#
#
# def process_dataset(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#         for line in infile:
#             line = line.strip()
#             processed_line = replace_special_tokens(line)
#             outfile.write(processed_line + '\n')
#
#
# def process_folder(input_folder, output_folder):
#     # Iterate through all files in the input folder
#     for filename in os.listdir(input_folder):
#         # Skip files that contain "PAD" in the filename
#         if "PAD" not in filename:
#             input_file = os.path.join(input_folder, filename)
#             # Replace "_Hierarchical_Fin_" with "_UNK_" in the filename for the output file
#             output_filename = filename.replace("_Hierarchical_Fin_", "_UNK_")
#             output_file = os.path.join(output_folder, output_filename)
#             print(f"Processing file: {filename} -> {output_filename}")
#             process_dataset(input_file, output_file)
#
#
# input_folder = '/mnt/free/zhangky/NationwidePT_Hierarchical/'
# output_folder = '/mnt/free/'
# os.makedirs(output_folder, exist_ok=True)
# process_folder(input_folder, output_folder)


# import pandas as pd
# import geopandas as gpd
# import fiona
# from scipy.spatial import KDTree
#
#
# # Function to load shapefiles using fiona
# def fast_load_shapefile(shapefile_path):
#     with fiona.open(shapefile_path) as src:
#         gdf = gpd.GeoDataFrame.from_features(src, crs=src.crs)
#     return gdf
#
#
# # Load the processed GPS data
# gps_data_path = '/home/zhangky/Documents/ZhangKY/clustered_output_july22.csv'
# gps_df = pd.read_csv(gps_data_path)
#
# # Load the three shapefiles containing POIs using fiona for faster loading
# shapefile1 = '/home/zhangky/Downloads/NearestPOIReduced/NearestPOIReduced/TokyoPOIOriAll_reduced.shp'
# shapefile2 = '/home/zhangky/Downloads/NearestPOIReduced/NearestPOIReduced/building_reduced.shp'
# shapefile3 = '/home/zhangky/Downloads/NearestPOIReduced/NearestPOIReduced/merged_shapefile_reduced.shp'
#
# gdf1 = fast_load_shapefile(shapefile1)
# gdf2 = fast_load_shapefile(shapefile2)
# gdf3 = fast_load_shapefile(shapefile3)
#
# # Extract the POI coordinates (lat/lon) from each shapefile
# coords_poi1 = list(zip(gdf1.geometry.y, gdf1.geometry.x))
# coords_poi2 = list(zip(gdf2.geometry.centroid.y, gdf2.geometry.centroid.x))
# coords_poi3 = list(zip(gdf3.geometry.centroid.y, gdf3.geometry.centroid.x))
#
# # Build KDTree for each POI dataset
# kdtree_poi1 = KDTree(coords_poi1)
# kdtree_poi2 = KDTree(coords_poi2)
# kdtree_poi3 = KDTree(coords_poi3)
#
#
# # Function to find the nearest POI using KDTree
# def find_nearest_poi(lat, lon, kdtree, gdf):
#     distance, index = kdtree.query([lat, lon], k=1)
#     return gdf.iloc[index]
#
#
# # Function to attach the nearest POI data to the GPS DataFrame
# def attach_nearest_pois(gps_df, kdtree_poi1, kdtree_poi2, kdtree_poi3, gdf1, gdf2, gdf3):
#     nearest_poi1 = []
#     nearest_poi2 = []
#     nearest_poi3 = []
#
#     for _, row in gps_df.iterrows():
#         lat, lon = row['lat'], row['lng']
#
#         # Find nearest POI from each KDTree
#         poi1 = find_nearest_poi(lat, lon, kdtree_poi1, gdf1)
#         poi2 = find_nearest_poi(lat, lon, kdtree_poi2, gdf2)
#         poi3 = find_nearest_poi(lat, lon, kdtree_poi3, gdf3)
#
#         # Append the relevant POI data
#         nearest_poi1.append(poi1['TypeCode'])  # Add TypeCode from POI 1
#         nearest_poi2.append(poi2['bldg_use'])  # Add building use from POI 2
#         nearest_poi3.append(poi3['L03b_002'])  # Add custom attribute from POI 3
#
#     # Attach the nearest POI information to the GPS DataFrame
#     gps_df['poi1_typecode'] = nearest_poi1
#     gps_df['poi2_bldg_use'] = nearest_poi2
#     gps_df['poi3_land_use'] = nearest_poi3
#
#     return gps_df
#
#
# # Run the nearest POI search and attach results to the GPS dataset
# gps_with_pois_df = attach_nearest_pois(gps_df, kdtree_poi1, kdtree_poi2, kdtree_poi3, gdf1, gdf2, gdf3)
#
# # Save the updated GPS dataset with attached POIs
# output_path = '/home/zhangky/Documents/ZhangKY/gps_with_nearest_pois.csv'
# gps_with_pois_df.to_csv(output_path, index=False)
#
# print(f"GPS data with nearest POIs saved to {output_path}")


import json
import random

input_json_file = "/mnt/free/zhangky/NationwidePT/Tokyo2008PTChain_Nationwide_Train.json"
output_json_file = "/mnt/free/Tokyo2008PTChain_Nationwide_GenInput2.json"
data = []
with open(input_json_file, 'r') as infile:
    for line in infile:
        data.append(json.loads(line.strip()))

# Randomly select 1000 sequences from the data
random.seed(42)  # Set a seed for reproducibility
selected_data = random.sample(data, 20)

# Write the randomly selected 1000 sequences to a new JSON file
with open(output_json_file, 'w') as outfile:
    for entry in selected_data:
        json.dump(entry, outfile)
        outfile.write('\n')  # Write each entry on a new line

print(f"Randomly selected 5000 sequences and saved to {output_json_file}")
