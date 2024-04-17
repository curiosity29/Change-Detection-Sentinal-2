# origional
# origin_index -> set_index, image_index, corX, corY
import json
import csv
import rasterio as rs
from ..Inference.Window import WindowExtractor
from rasterio.windows import Window as rWindow
from ..Preprocess.Normalize import scale, preprocess, preprocess_info
import pandas as pd
# augmented
# item_index -> origin_index, loss, metrics, augment
import os, shutil, glob
import numpy as np
# 
class RawDataset():
    def __init__(self, input_image_folder, input_label_folder, output_image_folders, output_label_folders, 
                ratio_list = None,
                window_size = 512, channel_last = False, training_length = None,
                image_preprocess = None, label_preprocess = None,
                status_file = "./Status/raw_dataset_status.json", command_file = "./Command/command.json",
                image_meta_file = "./Status/image_meta.csv",
                item_meta_file = "./Status/item_meta.csv", status_folder = "./Status/raw_dataset_status", auto = True,
                dynamic_preprocess = False, preprocess_info_file = None,
                restart_all = False, 
                ):
        """
            save by a prob list to a list of output folder
        """
        self.input_image_folder = input_image_folder
        self.input_label_folder = input_label_folder
        self.output_image_folders = output_image_folders
        self.output_label_folders = output_label_folders
        if ratio_list is None:
            self.ratio_list = [1./len(output_image_folders)] * len(output_image_folders) # equal chances by default
        else:
            self.ratio_list = ratio_list
        
        self.window_size = window_size
        self.channel_last = channel_last

        self.mode = "none"  
        self.extracted_set = []
        self.extracted_set_file = ""
        self.status_file = status_file
        self.status_folder = status_folder
        self.command_file = command_file
        self.image_preprocess = image_preprocess
        self.label_preprocess = label_preprocess
        self.item_meta_header = ("folder_index", "item_index", "image_index", "corX", "corY")
        self.image_meta_header = ("folder_index", "image_index", "lows", "highs", "means", "shape")
        # 
        self.item_meta_file = item_meta_file
        self.image_meta_file = image_meta_file
        self.auto = auto
        self.dynamic_preprocess = dynamic_preprocess
        self.prerprocess_info_file = preprocess_info_file


        if auto:
            self.auto_setup()
            
        self.__build__()
        if restart_all:
            self.__restart__()

        self.load_status()

    def output_folder_rotate(self):
        ### by prob
        # val = np.random.rand()
        # total_prob = 0.
        # for idx, prob in enumerate(self.prob_list):
        #     total_prob += prob
        #     if total_prob > val:
        #         return idx
        # return idx

        ## by count
        counts = np.array([len(glob.glob(os.path.join(folder, "*"))) for folder in self.output_image_folders])
        counts = counts / np.array(self.ratio_list)
        return np.argmin(counts)

    def auto_setup(self):
        """
            not impplemented
        """
        return
        # self.image_preprocess = lambda x: x
        # if os.path.exists(self.image_meta_file):
        #     return
            

    def dynamic_preprocess_info(self, image_index):
        with open(self.preprocess_info_file) as src:
            info = json.load()
        
        return info[image_index]
        
    def create_preprocess(self):
        """ not implemented """
        return
    def __restart__(self):
        os.makedirs(self.status_folder, exist_ok=True)
        self.extracted_set_file = os.path.join(self.status_folder, "extracted_set.npy")
        
        # create_item_status
        with open(self.item_meta_file, "w") as dest:
            # dest.write(f"{item_index};{image_index};{corX};{corY}", delimiter = ";")
            writer = csv.writer(dest)
            # writer.writerow(f"{item_index};{image_index};{corX};{corY}", delimiter = ";")
            writer.writerow(self.item_meta_header)
        with open(self.image_meta_file, "w") as dest:
            # dest.write(f"{item_index};{image_index};{corX};{corY}", delimiter = ";")
            writer = csv.writer(dest)
            # writer.writerow(f"{item_index};{image_index};{corX};{corY}", delimiter = ";")
            writer.writerow(self.image_meta_header)


        os.makedirs(self.input_image_folder, exist_ok=True)
        os.makedirs(self.input_label_folder, exist_ok=True)
        for folder in self.output_image_folders:
            os.makedirs(folder, exist_ok=True)
            for f in glob.glob(os.path.join(folder, "*"), recursive=True):
                os.remove(f)
        for folder in self.output_label_folders:
            os.makedirs(folder, exist_ok=True)
            for f in glob.glob(os.path.join(folder, "*"), recursive=True):
                os.remove(f)

        # create self status
        self.save_status()

    def __build__(self):
        return
        
    def append_data_stats(self, folder_index, image_index, image_path):
        """
            
        """
        # images_path = glob.glob(os.path.join(self.input_image_folder, "*.tif"))
                                
        with open(self.image_meta_file, "a") as dest:

            # dest.write(f"{item_index};{image_index};{corX};{corY}", delimiter = ";")
            writer = csv.writer(dest)
            with rs.open(image_path) as src:
                meta = src.meta
                image = src.read()
                image = np.transpose(image, (1, 2, 0))
                lows, highs, means = preprocess_info(image)

            # writer.writerow(f"{item_index};{image_index};{corX};{corY}", delimiter = ";")
            writer.writerow((folder_index, image_index, lows, highs, means, image.shape))
                            
        
    def load_status(self):
        with open(self.status_file) as src:
            status = json.load(src)
        self.extracted_set = status["State"]["extracted_set"]
        self.mode = status["State"]["mode"]
        # self.picker = status["State"]["picker"]
    def load_command(self):
        with open(self.command_file) as src:
            status = json.load(src)
        # self.last_index = status["State"]["last_index"]
        self.mode = status["Raw_data"]["mode"]
        # self.extracted_set_file = status["Raw_data"]["extracted_set_file"]
        # self.extracted_set = np.load(self.extracted_set_file)

    def save_status(self):
        status = {}
        status["State"] = {}
        
        status["State"]["mode"] = self.mode
        status["State"]["extracted_set"] = self.extracted_set
        # np.save(self.extracted_set_file, self.extracted_set)

        with open(self.status_file, "w") as dest:
            json.dump(status, dest, indent = 4)
            
    def get_info(self, image_path):
        with rs.open(image_path) as src:
            meta = src.meta
        return meta
    
    def execute(self, mode = None):
        if mode is None:
            self.load_command()
        else:
            self.mode = mode
        match self.mode:
            case "done": 
                return
            case "extract":
                self.mode = "extracting"
                self.save_status()
                self.extract_image()
                self.mode = "done extract"
                self.save_status()

       
    def get_data_file(self):
        image_files = glob.glob(os.path.join(self.input_image_folder, "*"), recursive=True)
        label_files = []
        matching_image_files = []
        for image_file in image_files:
            if image_file in self.extracted_set:
                continue
            head, image_name = os.path.split(image_file)
            label_name = image_name.replace("image", "label")
            label_file = os.path.join(self.input_label_folder, label_name)
            if os.path.exists(label_file):
                matching_image_files.append(image_file)
                label_files.append(label_file)
        return matching_image_files, label_files

    def create_extractor(self, image_file):
        info = self.get_info(image_file)
        extractor = WindowExtractor(image_shape = (info["width"], info["height"]), window_shape = (self.window_size, self.window_size), step_divide = 1)
        return extractor
    def save_item_status(self, folder_index, item_index, image_index, corX, corY):
        with open(self.item_meta_file, "a") as dest:
            # dest.write(f"{item_index};{image_index};{corX};{corY}", delimiter = ";")
            writer = csv.writer(dest)
            # writer.writerow(f"{item_index};{image_index};{corX};{corY}", delimiter = ";")
            writer.writerow((folder_index, item_index, image_index, corX, corY))

    def remove_image(self, image_files):
        print("not implemented")

    def input_name_mapping(self, image_index):
        image_name = f"image_{image_index}.tif"
        label_name = f"label_{image_index}.tif"
        return image_name, label_name
        
    def output_name_mapping(self, item_index):
        image_name = f"image_{item_index}.npy"
        label_name = f"label_{item_index}.npy"
        return image_name, label_name

    def extract_image(self):
        image_files, label_files = self.get_data_file()
        if len(image_files) == 0:
            self.save_status()
            return "extracted: 0"
        image_index = len(self.extracted_set)
        item_indexs = [len(glob.glob(os.path.join(folder, "*"), recursive=True))
                      for folder in self.output_image_folders]

        for image_file, label_file in zip(image_files, label_files):
            folder_index = self.output_folder_rotate() ## decide train or eval for each image

            extractor = self.create_extractor(image_file=image_file)
            
            ## save meta data
            self.append_data_stats(folder_index = folder_index, image_path=image_file, image_index=image_index)

            ## extract to small image
            while True:

                (corX, corY), corner_type = extractor.next()
                if corX is None:
                    break
                
                window = rWindow(corX, corY, self.window_size, self.window_size)
                with rs.open(image_file) as src:
                    image = src.read(window = window)
                    if self.dynamic_preprocess:
                        info = self.dynamic_preprocess_info(image_index=image_index)
                        image = np.transpose(image, (1,2,0))
                        image = self.image_preprocess(image, info)
                        image = np.transpose(image, (2,0,1))
                    elif self.image_preprocess is not None and not self.dynamic_preprocess:
                        image = np.transpose(image, (1,2,0))
                        image = self.image_preprocess(image)
                        image = np.transpose(image, (2,0,1))

                    
                with rs.open(label_file) as src:
                    label = src.read(window = window)
                    if self.label_preprocess is not None:
                        label = self.label_preprocess(label)
            
                if self.channel_last:
                    return np.transpose(image, (1,2,0)), np.transpose(label, (1,2,0))

                # if corner_type != [-1, -1]:
                #     return image, label, -1.

                item_index = item_indexs[folder_index]
                image_name, label_name = self.output_name_mapping(item_index)
                np.save(os.path.join(self.output_image_folders[folder_index], image_name), image)
                np.save(os.path.join(self.output_label_folders[folder_index], label_name), label)
                self.save_item_status(folder_index, item_index, image_index, corX, corY)
                item_indexs[folder_index] += 1
                    # return self[idx + 1 if idx < self.length else 0]
                # return image, label, 0.
            image_index += 1
            self.extracted_set.append(image_file)
        self.save_status()
        return f"extracted: {len(image_files)}"