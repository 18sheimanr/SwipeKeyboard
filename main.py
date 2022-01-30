from tkinter import *
from TypePad import TypePad
from difflib import SequenceMatcher
from skimage.transform import resize
from keras.models import load_model
import numpy as np
import string
from spell import correction
import matplotlib.pyplot as plt

model = load_model('characterModel')

num2alpha = dict(zip(range(1, 27), string.ascii_uppercase))
class PossibilityNode():

    def __init__(self, likelihood=1.0, inputGestures=None, possibleString=""):
        self.likelihood = likelihood
        self.inputGestures = inputGestures
        self.possibleString = possibleString

    def createNewPossibilities(self, ):

        if len(self.possibleString) == len(self.inputGestures) - 1:
            input = np.array([self.inputGestures[len(self.possibleString)].reshape(28, 28, 1)])
            pred = model.predict(input)[0]

            maxProbability = 0
            maxLetterNum = 0
            for p, i in zip(pred, range(0, 26)):
                if p > maxProbability:
                    maxProbability = p
                    maxLetterNum = i

            probableLetter = num2alpha[maxLetterNum]
            output = [(self.possibleString + probableLetter, self.likelihood*pred[maxLetterNum])]
            return output
        else:
            input = np.array([self.inputGestures[len(self.possibleString)].reshape(28, 28, 1)])
            pred = model.predict(input)[0]
            probableLetterNums = []
            for p, i in zip(pred, range(0, 26)):
                if p > 0.65:
                    probableLetterNums.append(i)

            additionalNode = None
            if max(pred) < 0.92:
                # newInputs = [np.add(self.inputGestures[0], self.inputGestures[1]), self.inputGestures[2:]]
                newInputs = list(np.zeros((1, 28, 28)))
                for x in range(len(self.inputGestures[0])):
                    for y in range(len(self.inputGestures[0][x])):
                        newInputs[0][x][y] = max(self.inputGestures[0][x][y], self.inputGestures[1][x][y])
                plt.imshow(newInputs[0])
                plt.show()
                newInputs = newInputs + self.inputGestures[2:]
                additionalNode = PossibilityNode(likelihood=self.likelihood,
                                                 inputGestures=newInputs,
                                                 possibleString=self.possibleString)

            probableLetters = [num2alpha[letterNum] for letterNum in probableLetterNums]
            output = []
            for i in range(0, len(probableLetterNums)):
                newLikelihood = self.likelihood*pred[probableLetterNums[i]]
                print(newLikelihood)
                if newLikelihood < 0.5:
                    continue
                else:
                    output = output + PossibilityNode(likelihood=newLikelihood,
                                    inputGestures=self.inputGestures,
                                    possibleString=self.possibleString+probableLetters[i]).createNewPossibilities()

            return output + additionalNode.createNewPossibilities() if additionalNode else output


class Application(Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.predStr = " "
        self.correctedStr = " "
        self.master = master
        self.pack()
        self.predLabel = Label(master=self, text='Prediction: Confidence: ')
        self.predLabel.pack()
        self.correctedLabel = Label(master=self, text='corrected phrase')
        # self.correctedLabel.pack()
        self.strLabel = Label(master=self, text="predicted string")
        self.strLabel.pack()
        self.pad = TypePad(parent=self)
        self.pad.pack()
        self.confidenceThreshold = (0.6, 0.9)
        self.gestureList = []

    # TODO - this
    def sendGesture(self, img):
        self.gestureList.append(resize(image=img, output_shape=(28, 28)))
        img = np.array([resize(image=img, output_shape=(28, 28)).reshape(28, 28, 1)])
        node = PossibilityNode(inputGestures=self.gestureList).createNewPossibilities()
        print(node)
        print("yeet")


window = Tk()
window.title("Touchpad Typer")
app = Application(master=window)
window.mainloop()
