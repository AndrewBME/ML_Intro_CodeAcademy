import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

print(aaron_judge.columns)
print(aaron_judge.description.unique())
print(aaron_judge.type)

#Replace strke (S) and ball (B)
aaron_judge['type'] = aaron_judge['type'].map({'S':1, 'B': 0})

print(aaron_judge['type'])
print(aaron_judge['plate_x'])
print(aaron_judge['plate_z'])

#Drop the Nan of three columns
aaron_judge = aaron_judge.dropna(subset = ['plate_x', 'plate_z', 'type'])
plt.scatter(x = aaron_judge.plate_x,
           y = aaron_judge.plate_z,
           c = aaron_judge.type,
           cmap = plt.cm.coolwarm,
           alpha = 0.25)

#Build SVM 
training_set, validation_set = train_test_split(aaron_judge, random_state = 1)

largest = {'value': 0, 'gamma': 1, 'C':1}
for gamma in range(1, 100):
  for C in range(1, 100):
    classifier = SVC(kernel = 'rbf', gamma = 1)
    #Fit the data
    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
    score = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])
    print(score)
    if score > largest['value']:
      largest['value'] = score 
      largest['gamma'] = gamma
      largest['C'] = C
      
print(largest)
    
fig, ax = plt.subplots()
draw_boundary(ax, classifier)
plt.show()
      



