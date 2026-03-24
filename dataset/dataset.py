import urllib.request
import zipfile
import os
import ssl
from pathlib import Path
from srs.LLM_classification.config import Config


url = Config.url
zip_path = Config.zip_path
extracted_path = Config.DATA_RAW_PATH
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
print(data_file_path)

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """
    Function that download dataset

    Returns:
    path where file is download
    """
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Create an unverified SSL context
    ssl_context = ssl._create_unverified_context()

    # Downloading the file
    with urllib.request.urlopen(url, context=ssl_context) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)