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
        #self.correctedLabel.pack()
        self.strLabel = Label(master=self, text="predicted string")
        self.strLabel.pack()
        self.pad = TypePad(parent=self)
        self.pad.pack()
        self.confidenceThreshold = (0.6, 0.9)

    #TODO - this
    def predict(self, img):

        img = np.array([resize(image=img, output_shape=(28, 28)).reshape(28, 28, 1)])
        pred = model.predict(img)
        confidence = np.amax(pred)
        num2alpha = dict(zip(range(1, 27), string.ascii_uppercase))
        letter = num2alpha[np.where(pred == confidence)[1][0]]

        if confidence > self.confidenceThreshold[1]:
            self.pad.reset()
            self.predStr = self.predStr + letter
            self.strLabel.configure(text=self.predStr)
        elif confidence < self.confidenceThreshold[0]:
            self.pad.reset()


        self.predLabel.configure(text="Prediction: %s, Confidence: %.2f" % (letter, confidence))
        return (letter, confidence)

def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None

window = Tk()
window.title("Touchpad Typer")
app = Application(master=window)
window.mainloop()