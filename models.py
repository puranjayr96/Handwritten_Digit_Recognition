import semeion
import keras
from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D,Dense,Flatten,Dropout
from keras.optimizers import Adadelta , RMSprop , SGD
import matplotlib.pyplot as plt

batch_size = 4
filters = 32
epochs = 20
activation = 'relu'
final_layer_activation = 'softmax'
d1 = 0.2
d2 = 0.2
compile_loss = keras.losses.categorical_crossentropy
learning_rate = 0.8
optimizer = Adadelta(lr=learning_rate)
split = 0.33


classes = 10;
im_shape = (256,1)
images , labels  = semeion.read_data_semeion()
images = images.reshape(images.shape[0], *im_shape)
#---------------------------------MODEL1-------------------------------------------------------
# cnn_model1 = Sequential([
#     Conv1D(filters=filters, kernel_size=3, activation=activation,input_shape=im_shape),
#     Conv1D(filters=filters, kernel_size=3, activation=activation, input_shape=im_shape),
#     MaxPooling1D(pool_size=2),
#     Dropout(d1),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(128, activation=activation),
#     Dropout(d2),
#     Dense(classes, activation=final_layer_activation)
# ])
#
# cnn_model1.compile(
#     loss=compile_loss,
#     optimizer=optimizer,
#     metrics=['accuracy']
# )
# history = cnn_model1.fit(
#     images, labels, validation_split=split, batch_size=batch_size,
#     epochs=epochs, verbose=1,
# )

# #--------------------------------MODEL2-------------------------------------------------------------
# cnn_model2 = Sequential([
#     Conv1D(filters=filters, kernel_size=3, activation=activation,input_shape=im_shape),
#     MaxPooling1D(pool_size=2),
#     Dropout(d1),
#     Conv1D(filters=filters, kernel_size=3, activation=activation, input_shape=im_shape),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(128, activation=activation),
#     Dropout(d2),
#     Dense(classes, activation=final_layer_activation)
# ])
#
# cnn_model2.compile(
#     loss=compile_loss,
#     optimizer=optimizer,
#     metrics=['accuracy']
# )
# history = cnn_model2.fit(
#     images, labels, validation_split=split, batch_size=batch_size,
#     epochs=epochs, verbose=1,
# )
#---------------------------------MODEL3-------------------------------------------------------------------
cnn_model3 = Sequential([
    Conv1D(filters=filters, kernel_size=3, activation=activation,input_shape=im_shape),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=filters, kernel_size=3, activation=activation,input_shape=im_shape),
    MaxPooling1D(pool_size=2),
    Dropout(d1),
    Conv1D(filters=filters, kernel_size=3, activation=activation, input_shape=im_shape),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation=activation),
    Dropout(d2),
    Dense(50, activation=activation),
    Dense(classes, activation=final_layer_activation)
])

cnn_model3.compile(
    loss=compile_loss,
    optimizer=optimizer,
    metrics=['accuracy']
)
history = cnn_model3.fit(
    images, labels, validation_split=split, batch_size=batch_size,
    epochs=epochs, verbose=1,
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train '+ str(acc[-1]), 'test '+str(val_acc[-1])], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train '+str(loss[-1]), 'test '+str(val_loss[-1])], loc='upper left')
plt.show()