import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.util import pad


class TypePad(tk.Canvas):

    drawRadius = 15     #should be like a pencil to match mnist data
    size = (300, 300)       #should also be like mnist
    imgArray = np.zeros(size)

    def __init__(self, parent):
        self.parent = parent
        tk.Canvas.__init__(self, width=self.size[0], height=self.size[1], bg='black')
        self.bind("<B1-Motion>", self.motion)
        self.bind("<ButtonRelease>", self.mouseRelease)
        self.width = self.size[0]
        self.height = self.size[1]


    def motion(self, event):
        x, y = event.x, event.y

        self.create_oval(x - self.drawRadius, y - self.drawRadius, x + self.drawRadius, y + self.drawRadius,
                         fill='white', outline="")

        for i in range(x - self.drawRadius, x + self.drawRadius):
            for j in range(y - self.drawRadius, y + self.drawRadius):
                if ((i - x) * (i - x) + (j - y) * (j - y)) <= self.drawRadius * self.drawRadius and j >= 0 and i >= 0\
                        and j < self.size[1] and i < self.size[0]:
                    self.imgArray[j][i] = 255

    def mouseRelease(self, event):
        img = self.transformedImgArray()
        letter, conf = self.parent.predict(img)
        plt.imshow(img)
        plt.show()

    def reset(self):
        self.imgArray = np.zeros(self.size)
        self.delete("all")

    # Fit the drawn figure to fill the image
    def transformedImgArray(self):

        xi_bounds = [300, 0]
        yi_bounds = [300, 0]
        for yi in range(0, self.imgArray.shape[0]):
            for xi in range(0, self.imgArray.shape[1]):
                if self.imgArray[yi][xi] != 0:
                    if xi < xi_bounds[0]:
                        xi_bounds[0] = xi
                    elif xi > xi_bounds[1]:
                        xi_bounds[1] = xi
                    if yi < yi_bounds[0]:
                        yi_bounds[0] = yi
                    elif yi > yi_bounds[1]:
                        yi_bounds[1] = yi

        ROI = self.imgArray[yi_bounds[0]:yi_bounds[1], xi_bounds[0]:xi_bounds[1]]
        pad_size = (int(ROI.shape[0]*0.1), int(ROI.shape[1]*0.1))
        newImg = np.pad(ROI, (pad_size[0], pad_size[1]))
        return gaussian(newImg, sigma=2)

