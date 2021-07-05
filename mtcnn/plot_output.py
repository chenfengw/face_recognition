# %%
import numpy as np
from PIL import Image
import sys
sys.path.append("/home/chw357/cse252d/cse252d-sp21-hw2/mtcnn")

from src import visualization_utils
# %%
path1 = "/datasets/cs252-sp21-A00-public/hw2_data/lfw/AJ_Cook/AJ_Cook_0001.jpg"
img1 = Image.open(path1)
lmk1 = np.array([103.672, 114.958, 146.303, 109.461, 118.223, 135.837, 110.155, 158.646, 149.281, 154.148])

# %%
path2 = "/datasets/cs252-sp21-A00-public/hw2_data/lfw/AJ_Lamas/AJ_Lamas_0001.jpg"
img2 = Image.open(path2)
lmk2 = np.array([104.424,	111.958,	149.001,	118.325,	130.224,	134.433,	97.063,	152.340,	147.845,	158.792	])
# %%
visualization_utils.show_bboxes(img1, [], facial_landmarks=lmk1.reshape(1,-1))
# %%
visualization_utils.show_bboxes(img2, [], facial_landmarks=lmk2.reshape(1,-1))
# %%
