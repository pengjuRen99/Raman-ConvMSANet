import torch
import numpy as np
from torch.utils.data import Dataset
import pathlib
import json
    

class BloodSpectra(Dataset):
    def __init__(self, spectra_path, json_name, mode='train'):
        self.spectra_path = spectra_path
        with open(json_name, 'r') as f:
            number_class = json.load(f)
        # key-value reversal.  number-class, class-number
        self.class_number = dict([key, int(val)] for val, key in number_class.items())
        self.label_arr = []
        
        if mode == 'train':
            self.train_path = pathlib.Path(self.spectra_path + '/train')
            self.train_spectra = [str(path) for path in list(self.train_path.glob('*/*'))]
            self.train_label = [pathlib.Path(sing_spectra).parent.name for sing_spectra in self.train_spectra]
            self.spectra_arr = self.train_spectra
            for label in self.train_label:
                self.label_arr.append(self.class_number[label])
        elif mode == 'valid':
            self.valid_path = pathlib.Path(self.spectra_path + '/valid')
            self.valid_spectra = [str(path) for path in list(self.valid_path.glob('*/*'))]
            self.valid_label = [pathlib.Path(sing_spectra).parent.name for sing_spectra in self.valid_spectra]
            self.spectra_arr = self.valid_spectra
            for label in self.valid_label:
                self.label_arr.append(self.class_number[label])
        elif mode == 'test':
            self.test_path = pathlib.Path(self.spectra_path + '/test')
            self.test_spectra = [str(path) for path in list(self.test_path.glob('*/*'))]
            self.test_label = [pathlib.Path(sing_spectra).parent.name for sing_spectra in self.test_spectra]
            self.spectra_arr = self.test_spectra
            for label in self.test_label:
                self.label_arr.append(self.class_number[label])
            
        self.real_len = len(self.spectra_arr)
        
        print('Finished reading the {} set of Spectra Dataset ({} samples found)'.format(mode, self.real_len))
        
    def __getitem__(self, index):
        sing_spectra_path = self.spectra_arr[index]
        # load spectra
        x, y = np.loadtxt(sing_spectra_path, dtype=float, comments='#', delimiter=',', unpack=True)
        y = y.reshape(1, y.shape[0])
        y = torch.tensor(y, dtype=torch.float32)
        label = self.label_arr[index]

        return y, label
    
    def __len__(self):
        return self.real_len
    
