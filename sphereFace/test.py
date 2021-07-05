# %%
import numpy as np
from PIL import Image
import requests
# %%
url = "https://www.e-education.psu.edu/natureofgeoinfo/sites/www.e-education.psu.edu.natureofgeoinfo/files/image/transforms_sim.gif"
im = Image.open("/datasets/cs252-sp21-A00-public/hw2_data/lfw/Ronde_Barber/Ronde_Barber_0001.jpg")
# %%
width, height = im.size
# %%
left = width/2 - (96/2)
top = height/2 - (112/2)
right = width/2 + (96/2)
bottom = height/2 + (112/2)
cropped_example = im.crop((left, top, right, bottom))
# %%
