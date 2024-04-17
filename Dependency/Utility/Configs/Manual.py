def todict(**args):
  return args

channels = 3
n_class = 1
SIZE = 512
filters = (32, 64, 128, 256)
args = todict(
    channels = channels,
    n_class = n_class,
    input_size = SIZE,
    output_size = SIZE,
    filters = filters
)

def get_args():
  return args