### this script correlates the csv with the downloaded images index to ensure that they are proprely aligned as not all images were able to be downloaded.

import os
import pandas as pd


CSV_PATH = "1/student_resource/dataset/train.csv"
IMAGE_DIR = "amazon_images"
OUTPUT_CSV = "1/student_resource/dataset/aligned_train.csv"


df = pd.read_csv(CSV_PATH)


image_files = os.listdir(IMAGE_DIR)
image_indices = {
    int(os.path.splitext(f)[0])
    for f in image_files
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
}

df["image_index"] = df.index
df_aligned = df[df["image_index"].isin(image_indices)].copy()

df_aligned["image_path"] = df_aligned["image_index"].apply(
    lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg")
)


df_aligned.to_csv(OUTPUT_CSV, index=False)

print(f"Original rows: {len(df)}")
print(f"Aligned rows:  {len(df_aligned)}")
print(f"Missing images: {len(df) - len(df_aligned)}")
