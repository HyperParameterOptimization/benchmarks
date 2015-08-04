
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

def input():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (X_train, y_train), (X_test, y_test)

def init_nn():
    model = Sequential()
    model.add(Dense(20, 64, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, 64, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, 2, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

def train_nn(*train):
    model.fit(X_train, y_train, nb_epoch=20, batch_size=16)

def test_nn(*test):
    score = model.evaluate(X_test, y_test, batch_size=16)
    return score

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = input()



