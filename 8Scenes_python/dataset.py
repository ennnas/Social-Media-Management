import os
import glob
from skimage import io as sio
from matplotlib import pyplot as plt
import numpy as np
from copy import copy
 

class Dataset:
    def __init__(self,path_to_dataset):
        self.path_to_dataset = path_to_dataset
        classes = sorted(os.listdir(path_to_dataset))
        self.paths = dict()
        for class_name in classes:
            class_path = sorted(glob.glob(os.path.join(path_to_dataset,class_name,'*.jpg')))
            self.paths[class_name]=class_path
            
    def getImagePath(self,class_name,idx):
        if idx is "all":
            return self.paths[class_name][:]
        else:
            return self.paths[class_name][idx]
    
    def getClasses(self):
        return sorted(self.paths.keys())
    
    def showImage(self,class_name,image_num):
        im = sio.imread(self.getImagePath(class_name,image_num))
        plt.figure()
        plt.imshow(im)
        plt.show()
        return 0
    
    def getNumberOfClasses(self):
        return len(self.getClasses())
    
    def getClassLength(self,class_name):
        return len(self.paths[class_name])
    
    def getLength(self):
        length = 0
        for class_name in self.paths.keys():
            length = length + self.getClassLength(class_name)
        return length
    
    def restrictDataset(self,sub_classes):
        restricted_data = {class_name:self.paths[class_name] for class_name in sub_classes}
        self.paths=restricted_data
        return 0
    
    def splitTrainingTest(self,percent):
        training_paths = dict()
        test_paths = dict()
        for class_name in self.getClasses():
            paths=self.paths[class_name]
            shuffled_paths = np.random.permutation(paths)
            split_idx = int(len(shuffled_paths)*percent)
            training_paths[class_name]=shuffled_paths[0:split_idx]
            test_paths[class_name]=shuffled_paths[split_idx:]
        
        training_dataset = copy(self)
        training_dataset.paths=training_paths
        test_dataset = copy(self) 
        test_dataset.paths=test_paths
        return training_dataset,test_dataset
        
    
    