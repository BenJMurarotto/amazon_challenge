import pandas as pd
import os
import requests
import random
import csv
from tqdm import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def data_loader(num_samples=None, price_cap=None, verbose = False):
    root = os.getcwd()
    data_dir = os.path.join(root, "1/student_resource/dataset")
    csv_path = os.path.join(data_dir, "aligned_train.csv")

    df = pd.read_csv(csv_path)

    if price_cap is not None:
        df = df[df["price"] <= price_cap]

    if num_samples is not None:
        df = df.head(num_samples)

    # reset index so everything is clean
    df = df.reset_index(drop=True)

    if verbose:
        prices = df["price"]

        print("Price summary:")
        print(f"min:   {prices.min():.2f}")
        print(f"p50:   {np.median(prices):.2f}")
        print(f"p90:   {np.quantile(prices, 0.90):.2f}")
        print(f"p95:   {np.quantile(prices, 0.95):.2f}")
        print(f"p99:   {np.quantile(prices, 0.99):.2f}")
        print(f"max:   {prices.max():.2f}")

        if price_cap is not None:
            prices_plot = prices[prices <= price_cap]
            title = f"Price Distribution (≤ ${price_cap})"
        else:
            prices_plot = prices
            title = "Price Distribution (All Prices)"

        plt.figure(figsize=(8, 4))
        plt.hist(prices_plot, bins=100)
        plt.xlabel("Price ($)")
        plt.ylabel("Count")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


        


    return df


    
def download_images(image_links, image_folder):
    for idx, image_url in tqdm(enumerate(image_links), total=len(image_links)):
        filename = f"{idx}.jpg"
        filepath = os.path.join(image_folder, filename)

        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()

            with open(filepath, "wb") as out_file:
                shutil.copyfileobj(response.raw, out_file)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")




# def get_image_tensors(num_samples = None):
#     root = os.getcwd()
#     image_folder = os.path.join(root, f"amazon_images")
#     os.makedirs(image_folder, exist_ok=True)
#     if len(os.listdir(image_folder)) < 1:
#         image_links = image_extractor(num_samples=num_samples)
#         download_images(image_links=image_links, image_folder=image_folder)


# get_image_tensors(num_samples=None)
        

