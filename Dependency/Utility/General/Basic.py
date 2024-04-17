import glob, os, cv2
import numpy as np


def varsToDict(**args):
  return args

def dictToVars(dictArgs):
  return dictArgs.values()

def findFolder(rootFolder, *names, sub_folder = False):
  for name_ in names:
    #glob.iglob
    folders_ = glob.glob(f"*{name_}*", root_dir = rootFolder, recursive=True)
    if len(folders_) >= 1:
      return os.path.join(rootFolder, folders_[0])
    else:
      return ""


def poly_to_mask(mask_shape, contours):
  mask = np.zeros((mask_shape))
  # for contour in contours:
  mask = cv2.fillPoly(mask, pts = contours, color=(1))

  return mask