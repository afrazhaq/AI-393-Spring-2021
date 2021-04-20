import numpy as np
import pandas as pd

import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.shape)
train.head()

print(test.shape)
test.head()

X = train.iloc[:, 1:]
y0 = train.iloc[:, 0]

X.head()

y0

binencoder = LabelBinarizer()
y = binencoder.fit_transform(y0)
y



X_images = X.values.reshape(-1,28,28)
test_images = test.values.reshape(-1,28,28)

print(X_images.shape)
print(test_images.shape)


plt.imshow(X_images[5])
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size = 0.2, random_state=90)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


test_images2 = test_images/255

test = test_images2.reshape(-1,28,28,1).astype('float32')





conv_model = Sequential()

conv_model.add(Conv2D(32,(4,4),input_shape = (28,28,1),activation = 'relu'))
conv_model.add(MaxPooling2D(pool_size=(2,2)))
conv_model.add(Conv2D(64,(3,3),activation = 'relu'))
conv_model.add(MaxPooling2D(pool_size=(2,2)))
conv_model.add(Dropout(0.2))
conv_model.add(Flatten())
conv_model.add(Dense(128,activation='relu'))
conv_model.add(Dense(50, activation='relu'))
conv_model.add(Dense(10, activation='softmax'))
conv_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

conv_model.summary()


result = conv_model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=92, verbose=2)
result


history_df = pd.DataFrame(result.history)
history_df.loc[:, ['loss', 'val_loss', 'accuracy', 'val_accuracy']].plot()

scores_test = conv_model.evaluate(X_test, y_test, verbose=0)
scores_test[1]


pred = conv_model.predict(test)

submit = pd.DataFrame(np.argmax(pred, axis=1), 
                      columns=['Label'], 
                      index=pd.read_csv('../input/digit-recognizer/sample_submission.csv')['ImageId'])


submit.index.name = 'ImageId'
submit.to_csv('submission.csv')

submit
