from keras.models import load_model
from emnist import extract_test_samples
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from skimage.transform import resize


model = load_model('characterModel')
images, labels = extract_test_samples('letters')
indeces = np.random.randint(5000, size=4)

for i in indeces:
    out = model.predict(np.array([images[i].reshape(28, 28, 1)]))
    variance = np.var(out)

f, axarr = plt.subplots(2, 2)
axarr[0,0].imshow(images[indeces[0]])
axarr[0,1].imshow(images[indeces[1]])
axarr[1,0].imshow(images[indeces[2]])
axarr[1,1].imshow(images[indeces[3]])
plt.show()

labels = to_categorical(labels)
images = images.reshape(images.shape[0], 28, 28, 1)
result = model.evaluate(images, labels)
print(result)