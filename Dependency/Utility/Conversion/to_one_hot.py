def to_one_hot(label): 
    return np.transpose(np.eye(n_class)[label.astype(np.uint8)][0], (2,0,1)) ## to one hot (1, size, size) -> (n_class, size, size)