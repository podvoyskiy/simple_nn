import os.path
from NeuralNetwork import NeuralNetwork
from Helper import Helper

if (__name__ != '__main__'): exit()

model = NeuralNetwork()

if (os.path.isfile("model.npz") == True and input("Use the last saved model? Enter 'y' or any other key\n") == 'y'):
    model.load()
else:
    if (input("Enter 'y' if need are training model, or any other key\n") == 'y'):
        model.learning()

        if (input(Helper.log("NN is learned. Save? Enter 'y' or any other key", Helper.COLOR_SUCCESS)) == 'y'):
            model.save()

test_img_file_name = input('Enter test file name in "testing" directory. By default - example.jpg\n')

model.check(test_img_file_name)