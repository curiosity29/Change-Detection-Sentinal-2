import torch


def input_adapter(image1, image2):
    """
        stack 2 image
    """
    return torch.stack([image1, image2], dim=-1)

def output_adapter(segmap_list):
    return segmap_list[0]
    