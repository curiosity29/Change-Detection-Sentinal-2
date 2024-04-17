import numpy as np

def scale(x, lows, highs):
  x = np.clip(x, lows, highs)
  return (x-lows)/(highs-lows)

def preprocess_info(image):
  """
    take statistic from an image to perform processing later 

    Return:
      lows, highs quantile 0.02, 0.98
  """
  lows, highs, means = [], [], []
  for channel in image.transpose((2,0,1)):
    vals = channel[channel != 0]
    lows.append(np.quantile(vals, q = 0.02, axis = 0))
    highs.append(np.quantile(vals, q = 0.98, axis = 0))
    means.append(np.mean(vals))
  lows, highs, means = np.array(lows), np.array(highs), np.array(means)
  return lows, highs, means

def preprocess(image, lows, highs, map_to = (-1, 0, 1)):
  """
    preprocess with quantiles lows and highs, then scale with map_to

    Args:
      lows: list of value for low quantiles (0.02)
      highs: list of value for high quantiles (0.98)
      map_to: [x, y, z] with x, y is the lowest/highest value and z is the value for all 0 pixel, after scaling

    Return:
      scaled image with min, max is x, y and 0 pixels is mapped to z
  """
  # save zeros position
  lows, highs = np.array(lows), np.array(highs)
  zeros_mask = np.where((image == 0).all(axis=1))

  image = np.clip(image, a_min = lows, a_max = highs)
  # to (0,1)
  image = scale(image, lows, highs)

  # nodata to -1
  image[zeros_mask] = -1
  # to (0.1, 1)
  # image = np.where(image >= 0, image * 0.9 + 0.1, image)
  # nodata to 0
  return image

def preprocess_inplace(image, map_to = (-1, 0, 1)):
  lows, highs = preprocess_info(image)
  return preprocess(image, lows, highs, map_to = map_to)