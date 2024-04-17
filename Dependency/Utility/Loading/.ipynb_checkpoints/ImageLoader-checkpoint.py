import rasterio as rs
import rasterio.features
import rasterio.mask
import geopandas as gd

def load_segment(pathTif, pathShape, pathBox = None, class_attribute = "class", oneBox = False):
  """
  load data for segmentation taask using paths, aligns shape
  according to tif image and extract class attribute to create label

  Args: 
  pathTif: tif file path for image
  pathShape: shape file path for mask
  pathBox: shape file path for box, the region of interest in the image
  class_attribute: the attribute in fileShape contain the class information, dtype is string or int
  one_box: return the first box or all the boxs
  Return:
  list of boxs and labels with channel last, 
  each have dimesion of 3, or 2 if oneBox is True
  """

  ## rasterio-1.3.9
  image = rs.open(pathTif)
  gdf = gd.read_file(pathShape)[[class_attribute, "geometry"]].dropna()#.drop(labels = "id", axis = 1)
  if class_attribute is None:
    label = list(gdf["geometry"])
  else:
    label = list(zip(gdf["geometry"], gdf[class_attribute].astype(int)))
  gdf.to_crs(crs = image.crs, inplace = True)

  if pathBox is None:
    # masks = image
    mask = rs.features.rasterize(label, out_shape = image.shape, transform = image.transform, fill = 0)
    # print(box.shape[1:])
    return image.read(), mask



  else:
    gdf_boxs = gd.read_file(pathBox)
    gdf_boxs.to_crs(crs = image.crs, inplace = True)
    masks = rs.features.rasterize(label, out_shape = image.shape, transform = image.transform, fill = 0)
  
    mask_arr = []
    box_arr = []
    for index, row in gdf_boxs.iterrows():
        # print(row)
        box, transform = rs.mask.mask(image, shapes = [row.geometry], crop = True)
        mask = rs.features.rasterize(label, out_shape = box.shape[1:], transform = transform, fill = 0)
        # print(box.shape[1:])
        mask_arr.append(mask)

        # print(box.shape)
        # for channel in range(len(box)):
        #   print(box[channel, ...].shape)
        #   box[channel, ...] = box[channel, ...]*1.0/MAX[channel]
        # print(np.max(box))

        box_arr.append(box.transpose((1,2,0)))
  
  if oneBox:
    box_arr, mask_arr = box_arr[0], mask_arr[0]

  return box_arr, mask_arr



def check_load_segment(pathTif, pathShape, pathBox = None, class_attribute = "class"):

  image = rs.open(pathTif)
  gdf = gd.read_file(pathShape)[[class_attribute, "geometry"]].dropna()#.drop(labels = "id", axis = 1)
  gdf.to_crs(crs = image.crs, inplace = True)
  if pathBox is None:
    return image, gdf
  else:
    gdf_boxs = gd.read_file(pathBox)
    gdf_boxs.to_crs(crs = image.crs, inplace = True)
    return image, gdf, gdf_boxs


def load_segment_list(pathList, class_attribute = "class"):
  """
  load data for segmentation taask using paths, aligns shape
  according to tif image and extract class attribute to create label

  Args: 
  pathTif: list of tif file path for image
  pathShape: list of shape file path for mask
  pathBox: list of shape file path for box, the region of interest in the image
  class_attribute: the attribute in fileShape contain the class information, dtype is string or int
  
  Return:
  list of boxs and labels with channel last, 
  each have dimesion of 3, or 2 if oneBox is True
  """
  images, labels = [], []
  for pathTif, pathShape, pathBox in pathList:
    image, label = load_segment(pathTif, pathShape, pathBox, class_attribute)
    images += image
    labels += label

  return images, labels