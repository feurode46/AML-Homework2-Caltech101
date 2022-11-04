from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, path_prefix=""):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        self.dataset = list()
        background_folder = "BACKGROUND_Google"
        label = 0
        if self.split.find("train") or self.split.find("test"):
            print("dentro if")
            with open(path_prefix + self.split + ".txt", "r") as fp:
                lines = fp.readlines()
                print("lines: ", len(lines))
                previous_label_name = lines[0].split("/")[0] # get first label name
                for line in lines:
                    if not line.find(background_folder):
                        # add tuple (PIL, label) to dataset
                        label_name = line.split("/")[0]
                        if label_name != previous_label_name:
                            label += 1
                        image = pil_loader(path_prefix + line)
                        self.dataset.append((image, label))
                        previous_label_name = label_name

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.dataset[index] # Provide a way to access image and label via index
                                           # Image should be a PIL Image
                                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.dataset) # Provide a way to get the length (number of elements) of the dataset
        return length
