import random
import matplotlib.pyplot as plt


def draw(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))
        if labels[i] == 1:
            plt.title("\nLabel:{}".format("Hemorrhage"))
        else:
            plt.title("\nLabel:{}".format("No Hemorrhage"))
    # Show the plot
    plt.show()


def drawPredict(model, testX, testY, images, index):
    modelName = str(model)
    modelName = modelName.split("(")[0]  # Extracting model name
    rand = random.randint(0, 39)  # Picking a random image from the test set
    inde = int(index[rand])
    
    # Displaying the image
    plt.imshow(images[inde], cmap=plt.get_cmap('gray'))
    
    # Showing the actual label in the plot
    if testY[rand] == 1:
        plt.title("\nLabel:{}".format("Hemorrhage"))
    else:
        plt.title("\nLabel:{}".format("No Hemorrhage"))
    plt.show()
    
    # Predicting the label for the randomly chosen test image
    prediction = model.predict([testX[rand]])[0]
    predict = "Hemorrhage" if prediction == 1 else "No Hemorrhage"
    label = "Hemorrhage" if testY[rand] == 1 else "No Hemorrhage"
    
    # Printing the result in the console
    print(f"The model {modelName} predicted: {predict}, the correct label: {label}")


# Example of how to extract and print model name
if __name__ == '__main__':
    model = "KNeighborsClassifier(n_jobs=-1, n_neighbors=2)"
    modelName = model.split("(")[0]
    print("Model Name:", modelName)

