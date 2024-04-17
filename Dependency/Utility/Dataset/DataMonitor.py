import numpy as np
import pandas as pd
import json
import os, glob
import csv
from ..Preprocess.Normalize import preprocess_info
import rasterio as rs

class DataMonitor():
    def __init__(self, monitor_file):
        self.monitor_file = monitor_file
        
    def clean(self):
        df = pd.read_csv(self.monitor_file)
        # df_sorted = df.sort_values(by = ["item_index"], ascending=True)
        df = df.drop_duplicates(subset='item_index', keep='last')
        df.to_csv(self.monitor_file)

    def get_hard(self, num, metric_name):
        df = pd.read_csv(self.monitor_file)
        df = df.sort_values(by = [metric_name], ascending=False)
        return np.array(df["item_index"][:num])


class RawDataMonitor():
    def __init__(self, monitor_file, meta_file, input_image_folder):
        self.monitor_file = monitor_file
        self.monitor_file_json = monitor_file.replace(".csv", ".json")
        self.meta_file = meta_file
        self.input_image_folder = input_image_folder
        self.header = ("image_index", "lows", "highs", "means", "shape")

    def wipe(self):
        with open(self.monitor_file_json, "w") as dest:
            json.dump({}, dest, indent = 4)
        
        # with open(self.monitor_file, "w") as dest:
        #     writer = csv.writer(dest)
        #     writer.writerow(self.header)
            
            
    def append_data_stats(self, current_stats, image_index, image_path):
            """
                
            """
            # images_path = glob.glob(os.path.join(self.input_image_folder, "*.tif"))
                                    
            with open(self.monitor_file_json, "w") as dest:
                
                # dest.write(f"{item_index};{image_index};{corX};{corY}", delimiter = ";")
                writer = csv.writer(dest)
                with rs.open(image_path) as src:
                    meta = src.meta
                    image = src.read()
                    image = np.transpose(image, (1, 2, 0))
                    lows, highs, means = preprocess_info(image)
                # writer.writerow(f"{item_index};{image_index};{corX};{corY}", delimiter = ";")
                # writer.writerow((image_index, lows, highs, means, image.shape))
                current_stats.update({f"{image_index}":
                    dict(
                        lows = list(lows),
                        highs = list(highs),
                        means = list(means),
                        shape = image.shape,
                    )
                })
                json.dump(current_stats, dest, indent = 4)
            return current_stats

    def append_stats(self):
        if os.path.exists(self.monitor_file_json):
            with open(self.monitor_file_json) as src:
                stats = json.load(src)
        else:
            stats = {}

        images_path = glob.glob(os.path.join(self.input_image_folder, "*.tif"))
        for index, path in enumerate(images_path):
            stats = self.append_data_stats(current_stats = stats, image_index = index, image_path = path)
        
        # with open(self.monitor_file_json) as src:
        #     stats = json.load(src)
        
        # df = pd.DataFrame(data)
        # df.to_csv(self.monitor_file, index = False)
        self.update_meta()
        return stats

    def update_meta(self):
        # df = pd.read_csv(self.monitor_file_json)
        with open(self.monitor_file_json) as src:
            list_stats = json.load(src)
        list_stats = list(list_stats.values())
        lows = list(np.mean([stats["lows"] for stats in list_stats], axis = 0))
        highs = list(np.mean([stats["highs"] for stats in list_stats], axis = 0))
        means = list(np.mean([stats["means"] for stats in list_stats], axis = 0))
        meta = {}
        meta["stats"] = {}
        meta["stats"]["lows"] = lows
        meta["stats"]["highs"] = highs
        meta["stats"]["means"] = means
        with open(self.meta_file, "w") as dest:
            json.dump(meta, dest, indent = 4)
        
    def execute(self, mode = "none"):
        match mode:
            case "none":
                return
            case "stats":
                self.append_stats()
            case "wipe":
                self.wipe()
            case "reset":
                self.wipe()
                self.append_stats()
    



