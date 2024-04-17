from Utility.Dataset.TFRecords import read, create_name_dataset, file_iterate_auto
# from functools import partial
def set_shape(x, y, image_shape, label_shape):
  x.set_shape(image_shape)
  y.set_shape(label_shape)
  return x, y

def TFDataset(root, n, file_iterate, image_shape, label_shape):
  if not callable(file_iterate):
    file_iterate = file_iterate_auto(name = str(file_iterate))
    
  raw_dataset = create_name_dataset(n = n, root = root, file_iterate = file_iterate)

  dataset = (
      raw_dataset
      .map(read)
      # .repeat()
      .map(lambda x, y: set_shape(x, y, image_shape, label_shape))
  )
  return dataset

def TFDataset_inplace(root, n, file_iterate, image_shape, label_shape):
  """
    auto save and read
  """
  return


