import os, pdb
import tarfile
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

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

def preprocess(dir_path, save_path):
    list_of_cif = []
    print(f"\nPreprocessing ...")
    for file in tqdm(Path(dir_path).glob('*.cif'), total=242517):
         dico = MMCIF2Dict(file)
         temp = pd.DataFrame.from_dict(dico, orient='index')
         temp = temp.transpose()
         temp.insert(0, 'Filename', Path(file).stem)
         list_of_cif.append(temp)
    df = pd.concat(list_of_cif)
    df = df[['Filename','_cell_length_a', '_cell_length_b', 
                '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta',
                '_cell_angle_gamma', '_cell_volume', '_atom_site_label',
                '_atom_site_type_symbol', '_atom_site_fract_x', '_atom_site_fract_y', 
                '_atom_site_fract_z', '_atom_type_partial_charge']]
    
    df.to_parquet(save_path+'/niaid.parquet.gzip',compression='gzip')
    print(f"\nSaved data to disk ...")


url = "https://zenodo.org/records/13891643/files/ARCMOF_20241004.tar.gz?download=1"
download_path = "./dataset/raw/ARCMOF_20241004.tar.gz"       
extract_to = "./dataset/raw"
dir_path = extract_to+'/ARCMOF_20241004/'                     

if not os.path.isdir(dir_path):
    download_and_extract_tar_gz(url, download_path, extract_to)
preprocess(dir_path, extract_to)


