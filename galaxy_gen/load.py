import gdown
import shutil
import os

def load_data():
    # Google Drive folder IDs (replace these with actual IDs)
    folder1_id = "15gDQl3P0Tu3LbosuwvpqPFNZSyP8gCvx"
    folder2_id = "18Mpjs4qNKLONGvMZ-rqtSynpxDNGs8Ui"

    # Destination paths
    dest_folder1 = "model"
    dest_folder2 = "data"
    # Ensure destination exists
    os.makedirs(dest_folder1, exist_ok=True)
    os.makedirs(dest_folder2, exist_ok=True)

    # Download each folder
    gdown.download_folder(f"https://drive.google.com/drive/folders/{folder1_id}", output=dest_folder1, quiet=False)
    gdown.download_folder(f"https://drive.google.com/drive/folders/{folder2_id}", output=dest_folder2, quiet=False)

    print("Folders downloaded and moved successfully!") 
