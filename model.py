import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
import warnings
import os
from drive import preprocess
import matplotlib.image as matimg
warnings.filterwarnings('ignore')

import random
import pandas as pd
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt

#Set the paths
os.chdir(r'C:\Users\yuang\Desktop\Self-Driving Cars Project')
image_path = r'C:\Users\yuang\Desktop\IMG'
driving_log_path = r'C:\Users\yuang\Desktop'

#Producing balanced training set
dlog = pd.read_csv(driving_log_path + "\\" + "driving_log.csv",
                   names=["center","left","right","steering","throttle","brake","speed"])
dlog_new = pd.DataFrame()   # New statistically representative dataset
buckets = 1000              # N of buckets
els = 200                   # N of elements to pick from each bucket (at most)

start = 0
#For every interval of size 1/buckets
for end in np.linspace(0, 1, num=buckets):
    #Take all the records with start<=steering<end
    df_range = dlog[(np.absolute(dlog.steering) >= start) & (np.absolute(dlog.steering) < end)]
    #Make sure that no bucket contains more than 200 elements
    range_n = min(els, df_range.shape[0])
    #Add new records to new driving log
    dlog_new = pd.concat([dlog_new, df_range.sample(range_n)])
    #Update interval starting point
    start = end
#Save new driving log as .csv
dlog_new.to_csv(driving_log_path + "\\" + "driving_log_new.csv", index=False)

dlog = dlog_new

#Split data into training and validation sets
train_samples, validation_samples = train_test_split(dlog, test_size=0.2)

#Cameras features
cameras = ['left', 'center', 'right']
cameras_steering_correction = [.25, 0., -.25]

#Keras generator yielding batches of training/validation data
#Optional parameter: extend=True perform training set augmentation (left/right imgs, add randomness)
def generator(df, path, extend=True):
    while True:
        # Generate random batch of sample indices
        samples = np.random.permutation(df.count()[0])
        batch_size = 128
        for batch in range(0, len(samples), batch_size):
            indices = samples[batch:(batch + batch_size)]
            # Initialize arrays
            x = np.empty([0, 32, 128, 3], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)
            # Read and preprocess a batch of images
            for i in indices:
                # Randomly select camera
                camera = np.random.randint(len(cameras)) if extend else 1
                # Read frame image and compute steering angle
                image = matimg.imread(os.path.join(path, df[cameras[camera]].values[i].strip()))
                image = np.copy(image) #Create a writable copy of img
                angle = df.steering.values[i] + cameras_steering_correction[camera]
                if extend:
                    # Add random shadow as a trapezoidal slice of image
                    h, w = image.shape[0], image.shape[1]
                    [x1, x2] = np.random.choice(w, 2, replace=False)
                    k = h / (x2 - x1)
                    b = - k * x1
                    for i in range(h):
                        c = int((i - b) / k)
                        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
                # Add randomness by randomly shifting up and down image views
                v_delta = .05 if extend else 0
                image = preprocess(
                    image,
                    top_offset=random.uniform(.375 - v_delta, .375 + v_delta),
                    bottom_offset=random.uniform(.125 - v_delta, .125 + v_delta)
                )
                x = np.append(x, [image], axis=0)
                y = np.append(y, [angle])
            # Randomly flip images and steering angles
            flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
            x[flip_indices] = x[flip_indices, :, ::-1, :]
            y[flip_indices] = -y[flip_indices]
            yield (x, y)

train_generator = generator(train_samples, image_path)
validation_generator = generator(validation_samples, image_path, extend=False)

#Building NN
model = Sequential()
model.add(Convolution2D(16, 5, 5, input_shape=(32, 128, 3), border_mode='same', name='conv1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, 5, 5, border_mode='same', name='conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='same', name='conv3'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

print(model.summary())

with open('summary.txt','w') as f:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: f.write(x + '\n'))

nb_max_epoch = 2
# Save current/average validation loss
cur_val_loss = None
avg_val_loss = None
val_loss = []
loss = []

for epoch in range(nb_max_epoch):
    # Train Model with Generator
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                         validation_data=validation_generator,
                                         nb_val_samples=len(validation_samples), nb_epoch=1,
                                         verbose=1, workers=0)

    # save Validation Loss/Loss and decide to terminate training
    cur_val_loss = history_object.history['val_loss'][0]
    cur_loss = history_object.history['loss'][0]
    loss.append(cur_loss)
    val_loss.append(cur_val_loss)
    avg_val_loss = np.average(val_loss[-3:])
    print('avg_val_loss\t{:2.4f}'.format(avg_val_loss))
    print('cur_val_loss\t{:2.4f}'.format(cur_val_loss))
    # only enter if we completed 2 epochs and the current val loss is more than the avg_val_loss
    if (len(val_loss) > 2) & (cur_val_loss > avg_val_loss):
        print('\nThe current validation loss: {:2.5f} '
              'is higher than the average of the last three: {:2.5f}'
              .format(cur_val_loss, avg_val_loss))
        print('breaking!')
        break
    # Save the model everytime the validation loss decreases
    model.save('model_new.h5')
    print('model saved!')

# Plot Learning Curves
def plot_learning_curves(label, loss, val_loss):
 plt.plot(np.arange(len(loss)) +1 , loss, "b.-", label="Training loss")
 plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
 plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
 plt.axis([0, 5, 0.05, 0.1])
 plt.legend(fontsize=14)
 plt.xlabel("Epochs")
 plt.ylabel("Loss")
 plt.tight_layout()
 plt.savefig(label+"_loss.png")
 plt.show()
plot_learning_curves("Model",loss,val_loss)
