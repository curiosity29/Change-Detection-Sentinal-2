import numpy as np
import glob, os
import csv
import rasterio as rs
import geopandas as gd
from rasterio.features import rasterize
from ..Utility.Preprocess.Normalize import preprocess_info, preprocess
import configparser
import pandas as pd
from functools import partial
import torch
import json

def prepare_test(pair_image_folders, raw_image_folder, raw_label_folder, class_attribute = "class", index = None, n_class = 4):
    if index is None:
        index = len(glob.glob(os.path.join(raw_image_folder, "*.tif")))
    for folder in pair_image_folders:
        label_paths = glob.glob(os.path.join(folder, "*.shp"))
        for path in label_paths:
            if "box" in path:
                box_path = path
            else:
                label_path = path
                
        output_image_path = os.path.join(raw_image_folder, f"image_{index}.tif")
        output_label_path = os.path.join(raw_label_folder, f"label_{index}.tif")
        
        image_path1 =  glob.glob(os.path.join(folder, "*1.tif"))[0]
        image_path2 =  glob.glob(os.path.join(folder, "*2.tif"))[0]
        with rs.open(image_path1) as src:
            meta = src.meta
            image1 = src.read()
        with rs.open(image_path2) as src:
            image2 = src.read()
            
        image = np.concatenate((image1, image2), axis = 0)


        
        mask_transform = meta["transform"]
        mask_shape = image.shape[1:]

        image_meta = meta.copy()
        image_meta['count'] = image_meta['count'] * 2
        image_meta["dtype"] = "float32"

        index += 1
        with rs.open(output_image_path, "w", **image_meta) as dest:
            dest.write(image)    
        
        if class_attribute is None:
            gdf = gd.read_file(label_path)[["geometry"]].dropna()
            gdf.to_crs(crs = meta["crs"], inplace = True)
            label = list(gdf["geometry"])
        else:
            gdf = gd.read_file(label_path)[[class_attribute, "geometry"]].dropna()
            gdf.to_crs(crs = meta["crs"], inplace = True)
            label = list(zip(gdf["geometry"], gdf[class_attribute].astype(int)))


        mask = rasterize(label, out_shape = mask_shape, transform = mask_transform, fill = 0)

        label_meta = meta.copy()
        label_meta["count"] = 1
        label_meta["dtype"] = "uint8"
        with rs.open(output_label_path, "w", **label_meta) as dest:
            dest.write(mask[np.newaxis, ...])


def prepare(pair_image_folders, raw_image_folder, raw_label_folder, box = False, class_attribute = "class", index = None, n_class = 4):
    if index is None:
        index = len(glob.glob(os.path.join(raw_image_folder, "*.tif")))
    for folder in pair_image_folders:
        images_path = glob.glob(os.path.join(folder, "*.tif"))
        label_path = glob.glob(os.path.join(folder, "*.shp"))[0]
        output_image_path = os.path.join(raw_image_folder, f"image_{index}.tif")
        output_label_path = os.path.join(raw_label_folder, f"label_{index}.tif")
        
        image_path1 = images_path[0]
        image_path2 = images_path[1]
        with rs.open(image_path1) as src:
            meta = src.meta
            image1 = src.read()
        with rs.open(image_path2) as src:
            image2 = src.read()
            
        image = np.concatenate((image1, image2), axis = 0)

        if box:
            print("not implemented - unknown box name") 
            return
        else:
            
            mask_transform = meta["transform"]
            mask_shape = image.shape[1:]

        image_meta = meta.copy()
        image_meta['count'] = image_meta['count'] * 2
        image_meta["dtype"] = "float32"

        index += 1
        with rs.open(output_image_path, "w", **image_meta) as dest:
            dest.write(image)    
        
        if class_attribute is None:
            gdf = gd.read_file(label_path)[["geometry"]].dropna()
            gdf.to_crs(crs = meta["crs"], inplace = True)
            label = list(gdf["geometry"])
        else:
            gdf = gd.read_file(label_path)[[class_attribute, "geometry"]].dropna()
            gdf.to_crs(crs = meta["crs"], inplace = True)
            label = list(zip(gdf["geometry"], gdf[class_attribute].astype(int)))

        if box:
            print("not implemented - unknown box name") 
            return
        else:
            mask = rasterize(label, out_shape = mask_shape, transform = mask_transform, fill = 0)

        label_meta = meta.copy()
        label_meta["count"] = 1
        label_meta["dtype"] = "uint8"
        with rs.open(output_label_path, "w", **label_meta) as dest:
            dest.write(mask[np.newaxis, ...])


def parser_float_list(s):
    arr = []
    for val in s[1:-1].split(" "):
        try:
            arr.append(float(val.strip()))
        except:
            pass
    return arr

def mean_attr(df, attr_name):
    val_list = []
    for vals in df[attr_name]:
        val_list.append(parser_float_list(vals))
    mean_vals = np.mean(np.array(val_list), axis = 0)
    return mean_vals
def preprocess_write_info(image_meta_file, output_file = "./Data/Meta/preprocess.ini"):
    df = pd.read_csv(image_meta_file)
    lows, highs, means = mean_attr(df, "lows"), mean_attr(df, "highs"), mean_attr(df, "means")
    configs = configparser.ConfigParser()
    # configs.read(self.output_file)
    configs["Stats"] = {}
    configs["Stats"]["lows"] = str(lows)
    configs["Stats"]["highs"] = str(highs)
    configs["Stats"]["means"] = str(means)
    with open(output_file, "w") as dest:
        configs.write(dest)

    return lows, highs, means

def get_image_preprocess(stats_file = "./Data/Meta/meta.json"):
    with open(stats_file) as src:
        stats = json.load(src)

    lows = stats["stats"]["lows"]
    highs = stats["stats"]["highs"]
    return partial(preprocess, lows = lows, highs = highs, map_to = (-1, 0, 1))     
        
def data_stats(raw_image_folder, output_file = "./output.csv"):
    """
        
    """
    images_path = glob.glob(os.path.join(raw_image_folder, "*.tif"))
    lows, highs = preprocess_info(image_path)[0]
    for image_path in images_path:
        lows, highs = preprocess_info(image_path)
                            

# def image_preprocess(image1, image2):
#     """
#         temporary test using farm
#     """
    
#     return np.concatenate([image1, image2], axis =-1)

def get_label_preprocess(n_class = 4):
    """
        to one_hot of n class
    """
    if n_class == 1:
        return lambda x: x.astype(np.uint8)
    else:
        return partial(label_preprocess, n_class = n_class)

def label_preprocess(label, n_class = 4):
    return np.transpose(np.eye(n_class)[label.astype(np.uint8)][0], (2,0,1))

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

# transforms.RandomApply(AddGaussianNoise(args.mean, args.std), p=0.5)

def get_augmenter():

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        AddGaussianNoise(0., 1.),
    ])

    return transform