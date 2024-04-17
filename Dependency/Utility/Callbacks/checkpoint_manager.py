from datetime import datetime
import json
import numpy as np
import os
import glob, shutil
class Checkpoint_manager():
    def __init__(self, name = "train", status_file = "./Status/checkpoint_manager_status.json", 
                 period = 100, metrics = {}, metrics_target = {}, 
                 clean_file = False,
                 checkpoint_extension = ".pth", time_format = "m%md%dH%HM%M", max_to_keep = 100, do_default_status = True):
        for metric in metrics.keys():
            if metric not in metrics_target.keys():
                metrics_target[metric] = "min" # default non-set target to min
        self.name = name
        self.period = period
        self.metrics = metrics.copy()
        self.metrics_target = metrics_target.copy()
        self.status_file = status_file
        self.last_index = 0
        self.checkpoint_extension = checkpoint_extension
        self.time_format = time_format
        self.max_to_keep = max_to_keep
            
        if do_default_status:
            self.create_default_status(force = clean_file)
        self.load_status()
        self.create_folder()

        if clean_file:
            self.clean_file()
            self.save_status()

    def clean_file(self):
        for folder in self.subfolders.values():
            # folder = os.path.join(self.checkpoint_folder, folder)
            for f in glob.glob(os.path.join(folder, f"*{self.checkpoint_extension}")):
                os.remove(f)
            print(f"cleaned {folder}")
        print(f"cleaned self ({self.name}) status")
        os.remove(self.status_file)

    def create_default_status(self, force = False):
        if force or not os.path.exists(self.status_file):
            status = {}
            status["Paths"] = {}
            status["Paths"]["checkpoint_folder"] = f"./Checkpoints/{self.name}_checkpoints"
            status["Vars"] = {}
            status["Vars"]["last_index"] = 0
            with open(self.status_file, "w") as dest:
                json.dump(status, dest, indent = 4)
            

    def create_folder(self):
        self.subfolders = {}
        period_folder = os.path.join(self.checkpoint_folder, "Period")
        self.subfolders["Period"] = period_folder
        os.makedirs(period_folder, exist_ok = True)
        for metric in self.metrics.keys():
            metric_folder = os.path.join(self.checkpoint_folder, f"Best_{metric}")
            os.makedirs(metric_folder, exist_ok = True)
            self.subfolders[metric] = metric_folder
                    
    def clean_all(self):
        for folder in self.subfolder.values():
            shutil.rmtree(folder)
    def load_status(self):
        with open(self.status_file) as src:
            status = json.load(src)
        self.checkpoint_folder = status["Paths"]["checkpoint_folder"]
        self.last_index = status["Vars"]["last_index"]
    def save_status(self):
        status = {}
        status["Paths"] = {}
        status["Paths"]["checkpoint_folder"] = self.checkpoint_folder
        status["Vars"] = {}
        status["Vars"]["last_index"] = self.last_index
        
        with open(self.status_file, "w") as dest:
            json.dump(status, dest, indent = 4)
        
    def get_name_period_last(self, now, new = True, remove_old = False):
        """
            return name of last period checkpoint
        """
        folder_name = self.subfolders["Period"]
        n = len(glob.glob(os.path.join(folder_name, "*")))
        
        # full = True if n >= self.max_to_keep else False
        if not new:
            n -=1
            if n <= 0:
                return None
            try:
                return glob.glob(os.path.join(folder_name, f"{self.name}_period_last_index{n}_*{self.checkpoint_extension}"))[0]
            except:
                return None

        if n >= self.max_to_keep:
                n = 0 ## back to index 0 if full
        checkpoint_path = os.path.join(folder_name, f"{self.name}_period_last_index{n}_{now}{self.checkpoint_extension}")
        if remove_old:
            for path in glob.glob(os.path.join(folder_name, f"{self.name}_period_last_index{n}_*{self.checkpoint_extension}")):
                os.remove(path)

        return checkpoint_path
    def get_name_metric_best(self, now, metric_name, new = True, remove_old = False):

        folder_name = self.subfolders[metric_name]
        n = len(glob.glob(os.path.join(folder_name, "*")))
        # full = True if n >= self.max_to_keep else False
        if not new:
            n -=1
            if n <= 0:
                return None
            try:
                return glob.glob(os.path.join(folder_name, f"{self.name}_best_{metric_name}_index{n}_*{self.checkpoint_extension}"))[0]
            except:
                return None
            
        if n >= self.max_to_keep:
            n = 0 ## back to index 0 if full
        checkpoint_path = os.path.join(folder_name, f"{self.name}_best_{metric_name}_index{n}_{now}{self.checkpoint_extension}")
        if remove_old:
            for path in glob.glob(os.path.join(folder_name, f"{self.name}_best_{metric_name}_index{n}_*{self.checkpoint_extension}")):
                os.remove(path)
        
        return checkpoint_path

    def get_save_paths(self):
        checkpoint_paths = {}
        now = datetime.now().strftime(self.time_format)
        checkpoint_paths["last_period"] = self.get_name_period_last(now = now, new = False)
        for metric, val in self.metrics.items():
            try:
                checkpoint_paths[f"best_{metric}"] = self.get_name_metric_best(now = now, metric_name = metric, new = False)
            except:
                pass
        return checkpoint_paths
    def get_save_paths_new(self, index, metrics = {}, remove_old = True):
        checkpoint_paths = {}
        now = datetime.now().strftime(self.time_format)
        checkpoint_last = checkpoint_paths["last_period"] = self.get_name_period_last(now = now, new = False)
        if (index - self.last_index < self.period) and checkpoint_last is not None:
            checkpoint_paths["last_period"] = checkpoint_last ## overwrite current checkpoint
        else:
            checkpoint_paths["last_period"] = self.get_name_period_last(now = now, new = True, remove_old = remove_old) ## create/overwrite next index checkpoint
            self.last_index = index
            
        for metric, val in self.metrics.items():
            try:
                if (metrics[metric] > val and self.metrics_target[metric] == "max")\
                or (metrics[metric] < val and self.metrics_target[metric] == "min"):
                    checkpoint_paths[f"best_{metric}"] = self.get_name_metric_best(now = now, metric_name = metric, new = True, remove_old = remove_old)
                    self.metrics[metric] = val

            except:
                pass
        return checkpoint_paths

        

            