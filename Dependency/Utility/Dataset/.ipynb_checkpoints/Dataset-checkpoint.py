
import csv
import rasterio as rs
import json
import os, shutil, glob
from torch.utils.data import Dataset
import numpy as np

class FinalDataset(Dataset):
    """
        preprocessed image and label with channel first

    """
    def __init__(self, name, image_folder, label_folder, window_size = 512, channel_last = False,
                status_file = "./Status/dataset_status.json", command_file = "./Command/command.json", command_section ="",
                restart_all = False):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.window_size = window_size
        self.channel_last = channel_last
        self.picker = "index" # random, 
        self.last_index = -1
        self.status_file = status_file
        self.command_file = command_file
        self.command_section = command_section
        self.name = name
        if restart_all:
            self.restart()
        
        
        # if training_length is not None:
        #     self.training_length = training_length
            
        self.load()

    def restart(self):
        
        self.save_status()
        
    def __reload__(self):
        length = len(glob.glob(os.path.join(self.image_folder, "*")))
        self.length = length
        
    def save_status(self):
        status = {}
        status[self.name] = {}
        
        status[self.name]["last_index"] = self.last_index
        status[self.name]["picker"] = self.picker

        with open(self.status_file, "w") as dest:
            json.dump(status, dest, indent = 4)
    def load_status(self):
        with open(self.status_file) as src:
            status = json.load(src)
        self.last_index = status[self.name]["last_index"]
        self.picker = status[self.name]["picker"]
    def load_command(self):
        with open(self.status_file) as src:
            status = json.load(src)
        # self.last_index = status["State"]["last_index"]
        self.picker = status[self.command_section]["picker"]

    def load(self):
        self.__reload__()
        self.load_status()
        self.load_command()
        
    def __len__(self):
        return self.length  

    def pick_index(self, idx):
        # self.last_index = idx
        self.last_index = self.last_index + 1
        if self.last_index >= self.length:
            self.last_index = 0
        return self.get_by_index(self.last_index)

    def pick_random(self, idx):
        # ignore index, pick random
        idx = np.random.randint(low = 0, high = self.length, size = 1)[0]
        self.last_index = idx
        return self.get_by_index(idx)

    def pick_hard(self, idx):
        """not implemented"""
        self.last_index = idx
        return self.get_by_index(idx)

    def get_by_index(self, idx):
        image_file = os.path.join(self.image_folder, f"image_{idx}.npy")
        label_file = os.path.join(self.label_folder, f"label_{idx}.npy")
        return np.load(image_file), np.load(label_file)

    def __getitem__(self, idx):
        self.save_status()
        match self.picker:
            case "random":
                return self.pick_random(idx)
            case "index":
                return self.pick_index(idx)
            case "hard":
                return self.pick_hard(idx)
            case _:
                return self.get_by_index(idx)
        
        