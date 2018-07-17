library(tensorflow)
library(keras)
library(kerasR)

##First read in the data file and check out file format
##Read in data
original_data <- read.table('Semeion/semeion.data.txt')
dim(original_data)
head(original_data[,257:266],3)

##Split original dataset into train and test dataset, prepare x and y dataset for keras model fit
set.seed(500)
sample_index = sample(nrow(original_data), nrow(original_data)*0.2)
test_dataset = original_data[sample_index,]
train_dataset = original_data[-sample_index,]

###x and y are supposed to be arrays

x_train = expand_dims(as.matrix(train_dataset[,1:256]),axis=2)
y_train = as.matrix(train_dataset[,257:266])
x_test = expand_dims(as.matrix(test_dataset[,1:256]),axis=2)
y_test = as.matrix(test_dataset[,257:266])

#Build deep neural networks
##In order to use tensorflow and keras, I need to get the API to Python
##tensorflow and keras installation was done seperatly
##both tensorflow and keras needs to be installed in Python
##keras website: https://keras.io/layers/convolutional/
##CN version: http://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/
#first intall python3.6
#then in cmd> pip install keras


##Model variable
input_shape = c(256,1) 
nb_classes = 10 
pool_size = 2 
kernel_size = 3 

batch_size = 128
nb_filters = 128
epochs = 40

##Define a keras sequential model
model <- keras_model_sequential()

##Add CNN layer - our input is a 1D array
layer_conv_1d(model, nb_filters, kernel_size, padding='same', activation='relu',
              input_shape = input_shape) ###Convolutional layer1
layer_max_pooling_1d(model, pool_size) ###Pooling layer1

layer_conv_1d(model, nb_filters, kernel_size, activation='relu') ###Convolution layer2
layer_max_pooling_1d(model, pool_size) ###Pooling layer1
layer_dropout(model, 0.25) ###Dropout (Pervent overfitting)

layer_conv_1d(model, nb_filters, kernel_size, activation='relu') ###Convolution layer3
layer_max_pooling_1d(model, pool_size) ###Pooling layer3

layer_flatten(model) ###Always flatten the model before going into fully connected layer

layer_dense(model, 128, activation='relu')###Fully connected layer1
layer_dropout(model, 0.35) ###Dropout (Pervent overfitting)
layer_dense(model, 50, activation='relu')###Fully connected layer2
layer_dense(model, nb_classes, activation = 'softmax')###Fully connected layer3

#Check model summary
summary(model)

#Complile the model
compile(model, optimizer='RMSprop', loss='mean_absolute_error', metrics='accuracy')

#Train the model
fit(model, x_train, y_train, batch_size, epochs, validation_data=list(x_test, y_test))


save_model_hdf5(model, '34th.h5')
##Predict and evaluate the model on test model
predict(model, x_test)
evaluate(model, x_test, y_test)
