import os

def assure_folder_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
