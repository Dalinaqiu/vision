from basic_transforms import *

class TrainAugmentation():
    def __init__(self, image_size, mean_val=0, std_val=1.0):
        #TODO: add self.augment, which contains
        # random scale, pad, random crop, random flip, convert data type, and normalize ops

    def __call__(self, image, label):
        return self.augment(image, label)
