import os
import shutil
import gzip
import tempfile
import subprocess

def decompress_gz_file(gz_path, output_dir):
    # Extract the filename without .gz
    output_path = os.path.join(output_dir, os.path.basename(gz_path)[:-3])
    with gzip.open(gz_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)  # Delete the .gz file after extraction

def process_repo(download_link, datadir):
    # Create a temporary directory to clone the repo
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_name = download_link.split("/")[-1].replace(".git", "")
        clone_path = os.path.join(tmpdir, repo_name)
        
        # Clone the repository
        subprocess.run(["git", "clone", download_link, clone_path], check=True)

        # Define source and destination paths
        src_folder = os.path.join(clone_path, datadir)
        dst_folder = os.path.abspath("raw_temp")

        # Move the target folder out
        shutil.move(src_folder, dst_folder)

        # Remove the rest of the cloned repo
        shutil.rmtree(clone_path)

    # Decompress all .gz files in the moved folder
    for file in os.listdir(dst_folder):
        if file.endswith(".gz"):
            gz_path = os.path.join(dst_folder, file)
            decompress_gz_file(gz_path, dst_folder)

    # Rename the folder to 'raw'
    final_dst = os.path.abspath("dataset/raw")
    if os.path.exists(final_dst):
        shutil.rmtree(final_dst)
    os.rename(dst_folder, final_dst)

    print(f"Done. Folder '{datadir}' extracted and processed as 'raw'.")

download_link = "https://github.com/uiocompcat/tmQM.git"
datadir = "tmQM"

process_repo(download_link, datadir)

