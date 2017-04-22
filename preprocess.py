from skimage import data, novice
from skimage.transform import resize, rotate
from random import randint
from PIL import Image as PIL_Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools

PWD = os.getcwd()
image_path = os.path.join(PWD, 'release/data/train_images/')

csv_path = os.path.join(PWD, 'release/data/train.csv')
df = pd.read_csv(csv_path)

def combine_rows(dataframe):
    left = 1 << 30
    top = 1 << 30
    right = -1
    bottom = -1
    img = novice.open(os.path.join(image_path, dataframe.iloc[0].FileName))
    digits = dataframe['DigitLabel'].tolist()
    digits = [i if i != 10 else 0 for i in digits]
    
    number = dataframe.iloc[0].FileName[:-4]
    left = dataframe['Left'].min()
    top = dataframe['Top'].min()
    right = dataframe[['Left', 'Width']].sum(axis=1).max()
    bottom = dataframe[['Top', 'Height']].sum(axis=1).max()
    new_df = pd.DataFrame([[int(number), int(''.join(str(d) for d in digits)),
                            digits, len(digits), img.width, img.height,
                            [left, top, right, bottom]]],
                          columns=['Number', 'Value', 'Digits', 'Length', 'Width', 'Height', 'Box'])
    return new_df

ndf = df.groupby('FileName')['FileName', 'DigitLabel',
                             'Left', 'Top', 'Width',
                             'Height'].apply(combine_rows).reset_index()
ndf = ndf.drop(ndf.columns[[1]], axis=1).set_index('Number').sort_index()
ndf.to_csv(os.path.join(PWD, 'processed.csv'))

def transform_image(path, box, final_dim, rand_rotate=0, rand_crop=10):
    """Transforms an image and returns a numpy array from it
    
    The image is resized around the bounding box after padding the image,
    randomly rotating it and randomly cropping it
    
    Parameters
    ----------
    path: str
          path to the image
    box: list
         dimensions of the left, top, right and bottom coordinates of the bounding box
    final_dim: tuple
               dimensions of the final image after cropping it
    rand_rotate: int, optional
                 rotation angle in degrees
    rand_crop: int, optional
               crop size
    
    Returns
    -------
    A numpy array of shape final_dim[0]xfinal_dim[1]x3
    """
    
    # open the image
    img = data.imread(path)
    
    if rand_rotate:
        img = rotate(img, randint(-rand_rotate, rand_rotate+1), mode='wrap')
    
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    left = max(0, int(left - width * 0.15))
    right = min(img.shape[1], int(right + width * 0.15))
    top = max(0, int(top - height * 0.15))
    bottom = min(img.shape[0], int(bottom + width * 0.15))
    
    # resize the image to bounding box + 30%
    img = img[top:bottom, left:right]
    # crop to bounding box + 30% + crop delta
    img = resize(img, final_dim)
    
    if rand_crop:
        crop_size = randint(0, rand_crop + 1)
        top = randint(0, crop_size)
        bottom = img.shape[0] - (crop_size - top)
        left = randint(0, crop_size)
        right = img.shape[1] - (crop_size - left)
        img = img[top:bottom, left:right]
        img = resize(img, (final_dim[0] + rand_crop, final_dim[1] + rand_crop))
    
    return img

def create_input_data(df, final_dim, rand_rotate, rand_crop):
    img_data = np.zeros((df.shape[0], final_dim[0], final_dim[1], 3), dtype=float)
    val = np.zeros((df.shape[0]), dtype=int)
    digits = np.full((df.shape[0], 6), 10)
    lengths = np.zeros((df.shape[0]), dtype=int)
    
    for i, row in df.iterrows():
        val[i] = row['Value']
        lengths[i] = row['Length']
        tmp = list(map(int, row['Digits'].strip('[').strip(']').split(',')))
        while len(tmp) < 6:
            tmp.append(10)
        digits[i] = np.array(tmp)
        img_data[i] = transform_image(os.path.join(PWD, 'release/data/train_images/', row['FileName']),
                                  list(map(int, row['Box'].strip('[').strip(']').split(','))),
                                  final_dim,
                                  rand_rotate,
                                  rand_crop)
    return img_data, val, digits, lengths

tdf = pd.read_csv('processed.csv')
i_data, val, digits, lengths = create_input_data(tdf, (32, 32), 4, 5)
np.save('images-32x32.npy', i_data)
np.save('val-32x32.npy', val)
np.save('digits-32x32.npy', digits)
np.save('lengths-32x32.npy', lengths)

