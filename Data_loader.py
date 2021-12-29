import math, random
import os
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2 as cv


class Dataloader(Sequence):
    def __init__(self, data_set, batch_size, shuffle=False):
        self.data_set = data_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.data_set) / self.batch_size)

        # batch 단위로 직접 묶어줘야 함

    def __getitem__(self, idx):
        # sampler의 역할(index를 batch_size만큼 sampling해줌)
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for i in indices:
            x = cv.imread(self.data_set[i][0],cv.IMREAD_GRAYSCALE)
            y = cv.imread(self.data_set[i][0],cv.IMREAD_GRAYSCALE)
            batch_x.append(x)
            batch_y.append(y)

        return [np.array(batch_x)],np.array(batch_y)

    # epoch이 끝날때마다 실행
    def on_epoch_end(self):
        self.indices = np.arange(len(self.data_set))
        if self.shuffle == True:
            np.random.shuffle(self.indices)



def data_loader(path='./DATA'):
    files = [os.path.join(path,f) for f in os.listdir(path)]
    inp_files = []
    out_files = []
    for f in files:
        if f.endswith('_mask.png'):
            out_files.append(f)
        elif not f.endswith('_dilate.png') and not f.endswith('_predict.png'):
            inp_files.append(f)
    out_files.sort();inp_files.sort()
    dataset = [inp_files,out_files]
    dataset = list(zip(*dataset))
    random.shuffle(dataset)
    trn_dataset = dataset[:int(0.7 * len(dataset))]
    val_dataset = dataset[int(0.7 * len(dataset)):int(0.9 * len(dataset))]
    tst_dataset = dataset[int(0.9 * len(dataset)):]
    trn_loader = Dataloader(trn_dataset, 8, shuffle=True)
    val_loader = Dataloader(val_dataset, 8, shuffle=False)
    tst_loader = Dataloader(tst_dataset, 8, shuffle=False)
    return trn_loader, val_loader, tst_loader
