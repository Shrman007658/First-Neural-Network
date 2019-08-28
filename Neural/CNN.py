from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train[:,:,:,np.newaxis] / 255.0
x_test = x_test[:,:,:,np.newaxis] / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#To chck shape--> use x_train.shape()
#f you check your x_train, you will have 60,000 x 28 x 28 x 1 data.
#Why x 1?
#The data CNN needs to read must be like this: total_data x width x height x channels.
#Height and width are self-explanatory. Channels are like Red or Green or Blue in RGB images. In RGB, because there are three channels, we need to make the data x 3. But because we work with grayscale images, every value on Red, Green, or Blue channel is the same and we reduce to one channel.
model=Sequential()
model.add(Conv2D(filters=64,kernel_size=2,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,validation_split=0.1)
_,test_acc=model.evaluate(x_test,y_test)
print(test_acc)
