from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
#Loading data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
##FLATTENING DATA
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

##MODELLING
#Use Relu whenever possible, on every hidden layer.
#Use Softmax on output layers with more than two categories to be predicted.
#Use Sigmoid on an output layer with two categories.

model=Sequential()
model.add(Dense(50,input_dim=784,activation='relu'))#First hidden layer
model.add(Dense(50,activation='relu'))#Second hidden layer
model.add(Dense(10,activation='softmax'))#output layer
#After creating your model, call compile method to finish your model. #
# It usually ta
# Always use categorical_crossentropy for metric to check your network performance.
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_split=0.1)

_, test_acc = model.evaluate(x_test, y_test)
print(test_acc)