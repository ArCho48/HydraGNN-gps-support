import os
import tarfile
import requests
from tqdm import tqdm

def download_with_progress(url, download_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 

    os.makedirs(os.path.dirname(download_path), exist_ok=True)

    with open(download_path, 'wb') as f, tqdm(
        desc=f"Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            f.write(data)
            bar.update(len(data))

def extract_tar_gz(file_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_to)

def download_and_extract_tar_gz(url, download_path, extract_to):
    download_with_progress(url, download_path)
    print(f"\nExtracting to {extract_to}...")
    extract_tar_gz(download_path, extract_to)
    os.remove(download_path)
    print(f"Removed archive: {download_path}")

url = "https://zenodo.org/records/13891643/files/ARCMOF_20241004.tar.gz?download=1"
download_path = "./dataset/raw/ARCMOF_20241004.tar.gz"       
extract_to = "./dataset/raw"                     

download_and_extract_tar_gz(url, download_path, extract_to)


