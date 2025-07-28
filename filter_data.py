import rasterio
from rasterio.warp import Resampling
import numpy as np
import os
import pandas as pd

### In total, four datasets are created. Two with the Train/Val/Test splits combining data from Austria, Belgium, & Luxembourg,
### and two independent test sets combinding data from Kosovo and Switzerland.
input_dataset = 'BigEarthNet-S2'
reference_dataset = 'Reference_Maps'

output_dataset = 'out_dataset'
moved_dataset = 'natural_ecosystem'


for country in ['Austria', 'Belgium', 'Luxembourg']:
    metadata_csv = pd.read_csv('metadata.csv')
    metadata_csv = metadata_csv[metadata_csv['country'].isin([country])]

    print(country)

    ### These classes are used to check for overlap, with the class labels.
    # Natural Ecosystem classes: [311, 312, 313, 321, 322, 324, 333, 411, 412, 511, 512]
    # Artificial Structures classes: [111, 112, 121, 122, 131, 133, 141, 142, 211, 221, 222, 231, 242, 243]
    unique_classes = [311, 312, 313, 321, 322, 324, 333, 411, 412, 511, 512]
    class_pixels = {k: 0 for k in unique_classes}

    total_imgs = 0
    kept_imgs = 0

    for row in metadata_csv.iterrows():
        row = row[1]

        try:
            bands_path = os.path.join(output_dataset, f"{row['patch_id']}.npy")
            bands = np.load(bands_path)
            bands = bands[-1]
        except: continue

        ### When counting the number of pixels in each country belonging to each class, use this code. Otherwise, exclude it.
        # Natural Ecosystem excluded classes: [111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142, 211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244, 323, 331, 332, 334, 421, 422, 423, 521, 522, 523, 999]
        # Artificial Structures excluded classes: [123, 124, 132, 212, 213, 223, 241, 244, 311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335, 411, 412, 421, 422, 423, 511, 512, 521, 522, 523, 999]
        # bands[np.isin(bands, [111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142, 211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244, 323, 331, 332, 334, 421, 422, 423, 521, 522, 523, 999])] = 0


        ### Check for overlap in class labels and keep all overlapping images.
        if bool(set(unique_classes) & set(np.unique(bands[-1]))):
            np.save(os.path.join(moved_dataset, row['split'], f'{row["patch_id"]}.npy'), bands)
            kept_imgs += 1

        ### Count number of pixels per class.
        # u_vals, counts = np.unique(bands, return_counts=True)
        # for u_val, count in zip(u_vals, counts):
        #     class_pixels[u_val] += int(count)

        total_imgs += 1

    # print(f'Class Values: {class_pixels.items()}\nTotal Images: {total_imgs}\n')
    #
    # print(class_pixels[0])
    # for c, p in list(class_pixels.items())[1:]:
    #     print(c, p / (total_imgs * 120 * 120 - class_pixels[0]))

    # print(f'Original Images: {total_imgs}\nKept Images: {kept_imgs}')
