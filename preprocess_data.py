import rasterio
from rasterio.warp import Resampling
import numpy as np
import os
import pandas as pd


input_dataset = 'BigEarthNet-S2'
reference_dataset = 'Reference_Maps'
output_dataset = 'out_dataset'

os.makedirs(os.path.join(output_dataset, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dataset, 'validation'), exist_ok=True)
os.makedirs(os.path.join(output_dataset, 'test'), exist_ok=True)

for country in ['Austria', 'Belgium', 'Luxembourg']:
  metadata_csv = pd.read_csv('metadata.csv')
  metadata_csv = metadata_csv[metadata_csv['country'].isin([country])]
  ### 'Kosovo', 'Switzerland' as independent test set. Change "output_dataset" and remove "row['split']" from final line if so. 
  
  metadata_cols = metadata_csv.columns
  
  for row in metadata_csv.iterrows():
      row = row[1]
  
      patch_folder = '_'.join(row['patch_id'].split('_')[:-2])
  
      input_data = []
      for band_path in os.listdir(os.path.join(input_dataset, patch_folder, row['patch_id'])):
          with rasterio.open(os.path.join(input_dataset, patch_folder, row['patch_id'], band_path)) as src:
  
              src_band = src.read(1, out_shape=(120, 120), resampling=Resampling.bilinear)
  
              input_data.append(src_band)
  
      reference_path = os.listdir(os.path.join(reference_dataset, patch_folder, row['patch_id']))[0]
      with rasterio.open(os.path.join(reference_dataset, patch_folder, row['patch_id'], reference_path)) as src_ref:
          src_ref = src_ref.read(1, out_shape=(120, 120), resampling=Resampling.bilinear)
  
          input_data.append(src_ref)
  
      input_data = np.array(input_data)
      # np.save(os.path.join(output_dataset, row['split'], f"{row['patch_id']}.npy"), arr=input_data)
      np.save(os.path.join(output_dataset, row['split'], f"{row['patch_id']}.npy"), arr=input_data)
