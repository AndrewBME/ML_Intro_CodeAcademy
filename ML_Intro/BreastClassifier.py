import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0])

#Print the labels of breast cancer
print(breast_cancer_data.target, breast_cancer_data.target_names)

#split the data
train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

training_data, validation_data , training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

print(len(training_data))
print(len(training_labels))

#Training and validation sets
classifier = KNeighborsClassifier(n_neighbors = 3)

classifier.fit(training_data, training_labels)

#Find the validation of classifier 
print(classifier.score(validation_data, validation_labels))

#Find and appropriate K
accuracies = []
for k in range(1, 101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracy = classifier.score(validation_data, validation_labels)
  accuracies.append(accuracy)
  
#Find the best K
k_list = range(1,101)
plt.plot(k_list, accuracies)
plt.show()

plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
