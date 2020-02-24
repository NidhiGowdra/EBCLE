import scipy
import numpy as np
from tqdm import tqdm
import cv2
from skimage.filters.rank import entropy
from skimage.measure import shannon_entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage import data


NP_TRAIN = 'PATH-TO-NUMPY-ARRAY'

#Load numpy arrays
train_data = np.load(NP_TRAIN)
Ent_array = np.empty(0)

for j,data in tqdm(enumerate(train_data)):
    img_data = data[0]
    a = np.asanyarray(img_data)
    MaxEnt = shannon_entropy(a, base=np.e)
    mean = np.mean(MaxEnt)
    Ent_array = np.append(Ent_array, mean)

mean_Ent = np.mean(Ent_array)
print(mean_Ent)
