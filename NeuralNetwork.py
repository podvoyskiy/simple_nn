import os
import matplotlib.pyplot as plt
import numpy as np
from Dataset import Dataset
from Helper import Helper, Synapses

class NeuralNetwork:
    __EPOCHS = 3
    weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
    weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))
    bias_input_to_hidden = np.ones((20,1))
    bias_hidden_to_output = np.ones((10,1))

    def save(self) -> None :
        np.savez('model.npz', weights_input_to_hidden = self.weights_input_to_hidden, weights_hidden_to_output = self.weights_hidden_to_output)
        
    def load(self) -> None :
        with np.load("model.npz") as f:
            self.weights_input_to_hidden = f['weights_input_to_hidden']
            self.weights_hidden_to_output = f['weights_hidden_to_output']

    def learning(self) -> None :
        #np.random.seed(1)
        images, labels = Dataset.load()

        e_loss = 0
        e_correct = 0
        learning_rate = 0.01

        for epoch in range(self.__EPOCHS):
            print(Helper.log(f"Epoch № {epoch+1}", Helper.COLOR_INFO))

            for image, label in zip(images, labels):
                #convert the image and its class into a two-dimensional array
                image = np.reshape(image, (-1, 1))
                label = np.reshape(label, (-1, 1))

                hidden = self.__feedForward(Synapses.input_to_hidden, image)
                output = self.__feedForward(Synapses.hidden_to_output, hidden)

                # Loss / Error calculation (MSE)
                e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
                e_correct += int(np.argmax(output) == np.argmax(label))

                # Backpropagation (output layer)
                delta_output = output - label
                self.weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
                # Backpropagation (hidden layer)
                delta_hidden = np.transpose(self.weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
                self.weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)

            if type(e_loss) != np.ndarray: continue
            # print some debug info between epochs
            print(f"Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
            print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%")
            e_loss = 0
            e_correct = 0
    
    def check(self, file_name: str = '') -> None :
        if file_name == '': file_name = 'example.jpg'
        if (os.path.isfile(f'testing/{file_name}') == False): 
            print(Helper.log(f'File testing/{file_name} is not found', Helper.COLOR_ERROR))
            return

        test_image = plt.imread(f'testing/{file_name}', format="jpeg")

        # Grayscale + Unit RGB + inverse colors
        gray = lambda rgb : np.dot(rgb[... , :3] , np.array([0.299 , 0.587, 0.114]))
        test_image = 1 - (gray(test_image).astype("float32") / 255)

        # Reshape
        test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

        # Predict
        image = np.reshape(test_image, (-1, 1))

        hidden = self.__feedForward(Synapses.input_to_hidden, image)
        output = self.__feedForward(Synapses.hidden_to_output, hidden)

        plt.imshow(test_image.reshape(28, 28), cmap="Greys")
        plt.title(f"NN suggests the number is: {output.argmax()}")
        plt.show()
        

    def __feedForward(self, layer: Synapses, matrix: np.ndarray) -> np.ndarray :
        if layer == Synapses.input_to_hidden: 
            matrix = self.bias_input_to_hidden + self.weights_input_to_hidden @ matrix
        else: 
            matrix = self.bias_hidden_to_output + self.weights_hidden_to_output @ matrix
        return 1 / (1 + np.exp(-matrix)) #sigmoid



