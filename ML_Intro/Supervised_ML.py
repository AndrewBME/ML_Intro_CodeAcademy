#Support Vector Machine 
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from graph import ax, x_1, y_1, x_2, y_2

#Top graph intercept and slope
intercept_one = 8
slope_one = -2

x_vals = np.array(ax.get_xlim())
y_vals = intercept_one + slope_one * x_vals
plt.plot(x_vals, y_vals, '-')

#Bottom Graph
ax = plt.subplot(2, 1, 2)
plt.title('Good Decision Boundary')
ax.set_xlim(6, 8)
ax.set_ylim(6, 10)

plt.scatter(x_1, y_1, color = "b")
plt.scatter(x_2, y_2, color = "r")

#Change the intercept to separate the clusters
intercept_two = 15
slope_two = -2

x_vals = np.array(ax.get_xlim())
y_vals = intercept_two + slope_two * x_vals
plt.plot(x_vals, y_vals, '-')

plt.tight_layout()
plt.show()

#Optimal Decision Boundaries
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from graph import ax, x_1, y_1, x_2, y_2

#Top graph intercept and slope
intercept_one = 98
slope_one = -20

x_vals = np.array(ax.get_xlim())
y_vals = intercept_one + slope_one * x_vals
plt.plot(x_vals, y_vals, '-')

#Bottom graph
ax = plt.subplot(2, 1, 2)
plt.title('Good Decision Boundary')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

plt.scatter(x_1, y_1, color = "b")
plt.scatter(x_2, y_2, color = "r")

#Bottom graph intercept and slope
intercept_two = 8
slope_two = -0.5

x_vals = np.array(ax.get_xlim())
y_vals = intercept_two + slope_two * x_vals
plt.plot(x_vals, y_vals, '-')

plt.tight_layout()
plt.show()

#Margins
red_support_vector = [1.0 , 6]
blue_support_vector_one = [0.5, 2]
blue_support_vector_two = [2.5, 2]

margin_size = 2

#Outlier
import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from graph import points, labels, draw_points, draw_margin

points.append([3, 3])
labels.append(0)

points.append([10, 8])
labels.append(1)

points.append([11, 7])
labels.append(1)

classifier = SVC(kernel='linear', C = 0.5)
classifier.fit(points, labels)

draw_points(points, labels)
draw_margin(classifier)

plt.show()

#Kernels 
import codecademylib3_seaborn
from sklearn.svm import SVC
from graph import points, labels
from sklearn.model_selection import train_test_split


training_data, validation_data, training_labels, validation_labels = train_test_split(points, labels, train_size = 0.8, test_size = 0.2, random_state = 100)

classifier = SVC(kernel = "poly", degree = 2)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))

#Polynomial Kernel
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#Makes concentric circles
points, labels = make_circles(n_samples=300, factor=.2, noise=.05, random_state = 1)

#Makes training set and validation set.
training_data, validation_data, training_labels, validation_labels = train_test_split(points, labels, train_size = 0.8, test_size = 0.2, random_state = 100)

classifier = SVC(kernel = "linear", random_state = 1)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))
print(training_data[0])

new_training = []
new_validation = []

for point in training_data:
  new_training.append([2 ** 0.5 * point[0] * point[1], point[0] ** 2, point[1] ** 2])
  #return new_training
  
for point in validation_data:
  new_validation.append([2 ** 0.5 * point[0] * point[1], point[0] ** 2, point[1] ** 2])
  #return new_validation 

classifier.fit(new_training, training_labels)
print(classifier.score(new_validation, validation_labels))

#Radial Basis Function
#Gamma is inverse of standard deviaton of rbf function 
#If gamma is large, the points would be more relavent 
#If C is too large, might be overfitting
from data import points, labels
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

training_data, validation_data, training_labels, validation_labels = train_test_split(points, labels, train_size = 0.8, test_size = 0.2, random_state = 100)

classifier = SVC(kernel = 'rbf', gamma = 0.1)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))
