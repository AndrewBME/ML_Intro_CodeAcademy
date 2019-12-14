## The Handwriting Recognition Project On Code Academy ##
##      Coded by Andrew Chen on 12/11/2019      ##

import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
#print(digits)
#DESCR is descriptor
print(digits.DESCR)
print(digits.data)
print(digits.target)

plt.gray()
plt.matshow(digits.images[100])
plt.show()

print(digits.target[100])

#Literally the random state can be anything but we chose 42
model = KMeans(n_clusters = 10, random_state = 42)

model.fit(digits.data)
fig = plt.figure(figsize = (8,3))
fig.suptitle('Cluster Center Images', fontsize = 14, fontweight = 'bold')

#Find centroid with Scikit-Learn 
for i in range(10):
  ax = fig.add_subplot(2, 5, 1+i)
  
  #Display images
  
ax.imshow(model.cluster_centers_[i].reshape((8,8)), cmap = plt.cm.binary)

plt.show()

new_samples = np.array()
new_labels = model.predict(new_samples)
print(new_labels)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')