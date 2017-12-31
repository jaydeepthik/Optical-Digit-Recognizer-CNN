# Optical-Digit-Recognizer-CNN-
optical digit recognizer using deep learning (CNN)

the cnn is developed using tensorflow, the structure : 

1. The first convolution layer has a [m X 28 X 28 X 1] input , with 8 Filters of size [4 X 4 X 1] and "same" padding, activaion ='relu'
2. A Max pool with 8 X 8 window and a stride of 8 ,"same" padding.
3. The second convolution layer has with 16 Filters of size [2 X 2 X 8] and "same" padding, activaion ='relu'
4. A Max pool with 4 X 4 window and a stride of 4 ,"same" padding.
5. A fully connected non-linear activation function, with 10 neurons in the output layer.


the dataset is a standard MNIST dataset with:
60000 training data
10000 testing data

each image of size 28X28



# RESULTS

TRAINING ACCURACY :97.041 %

TESTING ACCURACY  :96.74 %
