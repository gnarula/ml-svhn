from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

main_input = Input(shape=(64, 64, 3), name='main_input')

# First layer
x = Conv2D(48, (5, 5), activation='relu', padding='same')(main_input)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

# Second layer
x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
x = BatchNormalization()(x)

# Third layer
x = Conv2D(128, (5, 5), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Dropout(0.25)(x)

# Fourth layer
x = Conv2D(160, (5, 5), padding='same', activation='relu')(x)
x = BatchNormalization()(x)

# Fifth layer
x = Conv2D(192, (5, 5), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Dropout(0.25)(x)

# Final layers with average pooling
x = Conv2D(384, (5, 5), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = AveragePooling2D()(x)

x = Conv2D(768, (5, 5), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = AveragePooling2D()(x)

x = Flatten()(x)

# 6 outputs because max 6 digits
out0 = Dense(11, activation='softmax', name='out0')(x)
out1 = Dense(11, activation='softmax', name='out1')(x)
out2 = Dense(11, activation='softmax', name='out2')(x)
out3 = Dense(11, activation='softmax', name='out3')(x)
out4 = Dense(11, activation='softmax', name='out4')(x)
out5 = Dense(11, activation='softmax', name='out5')(x)

model = Model(inputs=main_input, outputs=[out0, out1, out2, out3, out4, out5])
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy')

filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=2, mode='auto')
callbacks_list = [checkpoint, early]


digits = np.load('digits_aws.npy')
images = np.load('images_aws.npy')

temp = (np.arange(digits.max() + 1) == digits[:,:,None]).astype(int)
y0 = temp[:, 0, :]
y1 = temp[:, 1, :]
y2 = temp[:, 2, :]
y3 = temp[:, 3, :]
y4 = temp[:, 4, :]
y5 = temp[:, 5, :]

h = model.fit(x=images, y=[y0, y1, y2, y3, y4, y5], epochs=150, batch_size=16, validation_split=0.2, callbacks=callbacks_list)

model.save('model_save.h5')


