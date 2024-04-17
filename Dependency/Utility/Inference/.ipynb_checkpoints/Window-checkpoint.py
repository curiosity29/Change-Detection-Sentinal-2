import numpy as np
import rasterio as rs
# import tensorflow as tf
from rasterio.windows import Window as rWindow
from tqdm import tqdm

class WindowExtractor():
  def __init__(self, image_shape, window_shape, step_divide = 1):
    self.image_shape = image_shape
    self.window_shape = window_shape
    self.index = 0
    self.step_divide = step_divide
    self.n_row = (image_shape[0] - window_shape[0])  // (window_shape[0] // step_divide) + 2 # + 2 at start and finish
    self.n_col = (image_shape[1] - window_shape[0]) // (window_shape[1] // step_divide) + 2 # + 2 at start and finish
    self.row = 0
    self.col = 0
    self.total = self.getTotal()

  def getTotal(self):
    return int(self.n_col * self.n_row)

  def getRowCol(self, index):
    return index // self.n_col, index % self.n_col

  def next(self):
    self.row, self.col = self.getRowCol(self.index)
    self.index += 1
    if self.index > self.total:
      return (None, None), (None, None)
    return self.getWindow(self.row, self.col)

  def toRowCol(self, corX, corY):
    """
      get row and col index from pixel coordinate this does account for the last image in each row
    """
    row = corX // (self.window_shape[0] // self.step_divide)
    col = corY // (self.window_shape[0] // self.step_divide)
    # corX = row * self.window_shape[0] // self.step_divide
    # corY = col * self.window_shape[0] // self.step_divide
    return row, col

  def getWindow(self, row, col):
    """
    return top left coordinate and corner type: None, (0, 0), (0,1), ...
    -1: not corner
    0: first (left or top)
    1: last (right or bottem)
    """
    corner_type = [-1, -1]
    # print("col: ", col, self.n_col)
    # if col == self.n_col:
    #   print("none")
    #   return (None, None), (None, None)

    # print(row, col)
    # corX, corY = 0, 0
    # posY, posX = self.index // self.n_col, self.index % self.n_col
    if row == self.n_row-1:
      corner_type[1] = 1
      corX = self.image_shape[0] - self.window_shape[0]
    else:
      corX = row * self.window_shape[0] // self.step_divide

    if col == self.n_col-1:
      corner_type[0] = 1
      corY = self.image_shape[1] - self.window_shape[1]
    else:
      corY = col * self.window_shape[1] // self.step_divide

    if row == 0:
      corner_type[0] = 0
    if col == 0:
      corner_type[1] = 0

    return (corX, corY), corner_type

  def getWindowIndex(self, index):
    row, col = self.getRowCol(index)
    return self.getWindow(row, col)
        

# windowExtractor = WindowExtractor(image_shape = (5000, 5000), window_shape = (512, 512), step_divide = 1)
# for _ in range(110):
#   window = windowExtractor.next()
#   print(window)
#   # print(window[0])
#   if window[0][0] is None:
#     break


def predict_windows(pathTif, pathSave, predictor, preprocess, window_size = 512, input_dim = 3, predict_dim = 1, output_type = "int8", batch_size = 4):

  with rs.open(pathTif) as src:
    # get meta
    out_meta = src.meta
    out_transform = src.transform
    profile = src.profile
    profile["transform"] = out_transform
    out_meta.update({"driver": "GTiff",
              "count": predict_dim, "dtype": output_type})

    #####   CURRENTLY ONLY WORK FOR step_divide = 2
    step_divide = 2
    image_W, image_H = out_meta["width"], out_meta["height"]
    extractor = WindowExtractor(image_shape=(image_W, image_H), window_shape = (window_size, window_size), step_divide = 2)
    with rs.open(pathSave, "w", **out_meta) as dest:
      total = extractor.getTotal()
      total = total // batch_size if total % batch_size == 0 else total // batch_size + 1
      pogbar = tqdm(total = total) #, disable = True
      while True:
        batch = []
        windows = []
        for _ in range(batch_size):
          (corX, corY), corner_type = extractor.next()
          # print(corX, corY, corner_type)
          if corX is None:
            break
          window = rWindow(corX, corY, window_size, window_size)
          if corner_type == [-1, -1]:
            windowWrite = rWindow(
              corX + window_size // 4, corY + window_size // 4, window_size//2, window_size//2)
            windows.append([windowWrite, True]) # is corner, write full size
          else:
            windows.append([window, False]) # not corner, write center

          image = src.read(window = window)
          image = np.transpose(image[:input_dim, ...], (1,2,0))
          image = preprocess(image)
          batch.append(image)
        if len(batch) == 0:
          break
        predicts = predictor(np.array(batch))

        for predict, window in zip(predicts, windows):
          predict = np.transpose(predict, (2,0,1))
          if window[1]:
            predict = predict[
              :, window_size // 4 : - window_size // 4, window_size // 4: -window_size // 4]

          dest.write(predict, window = window[0])

        pogbar.update(1)
      pogbar.close()



def predict(full_image, model, folder, patch_size = (512,512), frag = 1,return_image = False, save_image = True, verbose = 0):

  shapeX, shapeY = full_image.shape[0] //frag, full_image.shape[1] // frag
  for x in range(frag + 1):
    for y in range(frag + 1):
      if x == frag:
        pos_X = full_image.shape[0] - shapeX
      else:
        pos_X = shapeX * x
      if y == frag:
        pos_Y = full_image.shape[1] - shapeY
      else:
        pos_Y = shapeY * y

      patch_image = full_image[pos_X: pos_X + shapeX, pos_Y: pos_Y + shapeY, :]

      # def stitch(image, patch_size = (256,256)):
      image_x, image_y = patch_image.shape[:2]
      view_x = patch_size[0]
      view_y = patch_size[1]
      size_x = int(view_x/2)
      size_y = int(view_y/2)
      cut_x = int(size_x/2)
      cut_y = int(size_y/2)

      # gather image

      batch = []

      for pos_x in range(0, image_x - view_x, size_x):
        for pos_y in range(0, image_y - view_y, size_y):
          batch.append(patch_image[pos_x: pos_x + view_x, pos_y: pos_y + view_y, ...] )
      batch= np.array(batch)

      # idx = 0
      preds = []
      for idx in range(0, len(batch), 4):
        pred_ = model.predict(batch[idx: idx + 4], verbose = verbose)
        for p in pred_:
          preds.append(p)
      if len(batch)%4 != 0:
        pred_ = model.predict(batch[-(len(batch)%4):], verbose = verbose)

        for p in pred_:
          # print(p.shape)
          preds.append(p)
      preds = np.array(preds)

      top = []
      pos_y = 0
      for pos_x in range(0, image_x - view_x, size_x):
        top.append(patch_image[pos_x: pos_x + view_x, pos_y: pos_y + view_y, ...] )
      batch= np.array(top)
      preds_top = model.predict(batch, verbose = verbose)


      right = []
      pos_x = image_x - view_x
      for pos_y in range(0, image_y - view_y, size_y):
        right.append(patch_image[pos_x: pos_x + view_x, pos_y: pos_y + view_y, ...] )
      batch= np.array(right)
      preds_right = model.predict(batch, verbose = verbose)


      bottem = []
      pos_y = image_y - view_y
      for pos_x in range(image_x - view_x, 0, -size_x):
        bottem.append(patch_image[pos_x: pos_x + view_x, pos_y: pos_y + view_y, ...] )
      batch= np.array(bottem)
      preds_bottem = model.predict(batch, verbose = verbose)

      left = []
      pos_x = 0
      for pos_y in range(image_y - view_y, 0, -size_y):
        left.append(patch_image[pos_x: pos_x + view_x, pos_y: pos_y + view_y, ...] )
      batch= np.array(left)
      preds_left = model.predict(batch, verbose = verbose)

      predict_dim = 1 if len(preds.shape) < 4 else preds.shape[-1]
      stitched = np.zeros((image_x, image_y, predict_dim))
      # print(preds.shape)
      idx = 0
      for pos_x in range(cut_x, image_x - view_x + cut_x, size_x):
        for pos_y in range(cut_y, image_y - view_y + cut_y, size_y):
          # print([pos_x, pos_y, size_x, patch_image.shape])
          stitched[pos_x: pos_x + size_x, pos_y: pos_y + size_y, :] = preds[idx, cut_x: cut_x + size_x, cut_y: cut_y + size_y, :]
          idx = idx + 1

      # top
      idx = 0
      pos_y = 0
      for pos_x in range(0, image_x - view_x, size_x):
        stitched[pos_x: pos_x + view_x, 0: view_y, :] = preds_top[idx, :, :view_y, :]
        idx = idx + 1

      # right
      idx = 0
      pos_x = image_x - size_x - cut_x
      for pos_y in range(0, image_y - view_y, size_y):
        stitched[-view_x:, pos_y: pos_y + view_y, : ] = preds_right[idx, -view_x:, :, :]
        idx = idx + 1

      # bottem
      idx = 0
      pos_y = image_y - view_y-cut_y
      for pos_x in range(image_x - view_x, 0, -size_x):
        stitched[pos_x: pos_x + view_x, -view_y:, :] = preds_bottem[idx, :, -view_y:, :]
        idx = idx + 1

      # left
      idx = 0
    
      pos_x = 0
      for pos_y in range(image_y - view_y, 0, -size_y):
        stitched[:view_x, pos_y: pos_y + view_y, : ] = preds_left[idx, :view_x, :, :]
        idx = idx + 1

      np.save(f"{folder}data/{pos_X}_{pos_Y}.npy", stitched)
      return predict_dim


def predict_middle(full_image, model, folder, patch_size = (512,512),frag = 1, return_image = False, save_image = True, verbose = 0):

  shapeX, shapeY = full_image.shape[0] //frag, full_image.shape[1] // frag
  for x in range(frag + 1):
    for y in range(frag + 1):
      if x == frag:
        pos_X = full_image.shape[0] - shapeX
      else:
        pos_X = shapeX * x
      if y == frag:
        pos_Y = full_image.shape[1] - shapeY
      else:
        pos_Y = shapeY * y

      patch_image = full_image[pos_X: pos_X + shapeX, pos_Y: pos_Y + shapeY, :]

      # def stitch(image, patch_size = (256,256)):
      image_x, image_y = patch_image.shape[:2]
      view_x = patch_size[0]
      view_y = patch_size[1]
      size_x = int(view_x/2)
      size_y = int(view_y/2)
      cut_x = int(size_x/2)
      cut_y = int(size_y/2)

      # gather image

      batch = []

      for pos_x in range(0, image_x - view_x, size_x):
        for pos_y in range(0, image_y - view_y, size_y):
          batch.append(patch_image[pos_x: pos_x + view_x, pos_y: pos_y + view_y, ...] )
      batch= np.array(batch)

      # idx = 0
      preds = []
      for idx in range(0, len(batch), 4):
        pred_ = model.predict(batch[idx: idx + 4], verbose = verbose)
        for p in pred_:
          preds.append(p)
      if len(batch)%4 != 0:
        pred_ = model.predict(batch[-(len(batch)%4):], verbose = verbose)

        for p in pred_:
          # print(p.shape)
          preds.append(p)
      preds = np.array(preds)


      predict_dim = 1 if len(preds.shape) < 4 else preds.shape[-1]
      stitched = np.zeros((image_x, image_y, predict_dim))
      # print(preds.shape)
      idx = 0
      for pos_x in range(cut_x, image_x - view_x + cut_x, size_x):
        for pos_y in range(cut_y, image_y - view_y + cut_y, size_y):
          # print([pos_x, pos_y, size_x, patch_image.shape])
          stitched[pos_x: pos_x + size_x, pos_y: pos_y + size_y, :] = preds[idx, cut_x: cut_x + size_x, cut_y: cut_y + size_y, :]
          idx = idx + 1


      np.save(f"{folder}data/{pos_X}_{pos_Y}.npy", stitched)
      return predict_dim



# load
def load_predict(image_shape, folder, predict_dim = 1, frag = 5):

  # sh = np.array([7931, 7023, predict_dim])
  shapeX, shapeY = image_shape[0]//frag, image_shape[1] //frag
  paths = folder + "data/{pos_x}_{pos_y}.npy"
  full_predict = np.zeros(image_shape[:2] + (predict_dim,))
  for x in range(frag+1):
    for y in range(frag+1):
      if x == frag:
        pos_x = image_shape[0] - shapeX
      else:
        pos_x = shapeX*x
      if y == frag:
        pos_y = image_shape[1] - shapeY
      else:
        pos_y = shapeY*y
      path = paths.format(pos_x = pos_x, pos_y = pos_y)
      full_predict[pos_x: pos_x + shapeX, pos_y: pos_y+shapeY, :] = np.load(path)
  return full_predict


def save_result(filename, predict, data_type = "uint16", window = None, info_ref_path = None):
  if info_ref_path is not None:
    with rs.open(info_ref_path) as src:
        out_meta = src.meta

        out_transform = src.transform
        # out_image, out_transform = rasterio.mask.mask(src, shapes=shapes, crop=True)
        profile = src.profile

        # channel first
        profile["height"] = src.shape[0]
        profile["width"] = src.shape[1]
        profile["transform"] = out_transform
        out_meta.update({"driver": "GTiff",
                  "count": predict.shape[0], "dtype": data_type})
  else:
    print("not implemented, do nothing")
    return 
  with rs.open(filename, "w", **out_meta) as dest:
    if window is None:
      dest.write(predict)
    else:
      dest.write(predict, window = window)