import numpy as np
import matplotlib.pyplot as plt
import h5py
from utilities import load_dataset
from logreg import LogisticRegression

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
'''
index = 27
plt.imshow(train_set_x_orig[index])
plt.show()
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
'''

# Flatten the images
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Normalise image values
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

# Create model instance
model = LogisticRegression()

# Fit model to the data
model.fit(train_set_x, train_set_y)

# Train the model
model.train(2400, verbose=True)

# Predict values
predictions = model.predict(test_set_x)

# Check accuracy
model.print_accuracy(predictions, test_set_y)

# Plot training loss
model.plot_cost()
