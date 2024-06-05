import requests
import os
import subprocess
import argparse
from zipfile import ZipFile
import shutil

#%%

def download_files(local_path):
    """
    Download a file from the web and save it locally.

    Parameters:
    - local_path: Path where the file should be saved locally.
    """
    if local_path[-1] != '/':
        local_path = local_path + '/'
#%%        
    url = 'https://drive.google.com/uc?export=download&id=1--C6MJf7G1OrM8IHj55OJOI5jQuozp1u'
    file_name = 'neuralhydrology.zip'
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path + file_name, 'wb') as file:
            file.write(response.content)
        print(f"{file_name} folder downloaded successfully")
    else:
        raise Exception(f"Failed to download file: {response.status_code}")

    fp = local_path + file_name
    with ZipFile(fp, 'r') as f:
        f.extractall()
        
    os.remove(fp)
        
    # Move the folder
    try:
        shutil.move(local_path + 'neuralhydrology', local_path + 'Results/neuralhydrology')
        print(f"Folder neuralhydrology moved to Results")
        print('1/3 folders')
    except Exception as e:
        print(f"Error moving folder: {e}")
        

            
#%%        
    url = 'https://drive.google.com/uc?export=download&id=13qwiI8Sj4XplXQt712y8NUcdOMU35cJs'
    file_name = 'RF_mean_0.0.0.0.zip'
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path + file_name, 'wb') as file:
            file.write(response.content)
        print(f"{file_name} folder downloaded successfully")
    else:
        raise Exception(f"Failed to download file: {response.status_code}")

    fp = local_path + file_name
    with ZipFile(fp, 'r') as f:
        f.extractall()
        
    os.remove(fp)
        
    # Move the folder
    try:
        shutil.move(local_path + 'RF_mean_0.0.0.0', local_path + 'Results/RF_mean_0.0.0.0')
        print(f"Folder RF_mean_0.0.0.0 moved to Results")
        print('2/3 folders')
    except Exception as e:
        print(f"Error moving folder: {e}")
    

        
#%%        

    url = 'https://drive.google.com/uc?export=download&id=16wTinNOgK9ItixG0DsD1qqLawKP5OitE'
    file_name = 'hydroLSTM.zip'
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path + file_name, 'wb') as file:
            file.write(response.content)
        print(f"{file_name} folder downloaded successfully")
    else:
        raise Exception(f"Failed to download file: {response.status_code}")

    fp = local_path + file_name
    with ZipFile(fp, 'r') as f:
        f.extractall()
        
    os.remove(fp)
        
    # Move the folder
    try:
        shutil.move(local_path + 'hydroLSTM', local_path + 'Results/hydroLSTM')
        print(f"Folder neuralhydrology moved to Results")
        print('3/3 folders. All de folder were downloaded')
    except Exception as e:
        print(f"Error moving folder: {e}")
        



        
#%%

# parser = argparse.ArgumentParser()  # Create an ArgumentParser object to handle command-line arguments
# parser.add_argument('--path', type=str, help='local path of the cloned repo')  

# cfg = vars(parser.parse_args())  # Parse the command-line arguments and convert them to a dictionary
# cfg["path"] == os.getcwd()
download_files(os.getcwd())
