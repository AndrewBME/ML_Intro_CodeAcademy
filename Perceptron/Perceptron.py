## 12/9/2019 ##
#Never Thought I can be here
#But final is coming 
#So must flex hard

#Perceptron
#Input, Weight, Output

class Perceptron:
      def __init__(self, num_inputs=2, weights=[1,1]):
    self.num_inputs = num_inputs
    self.weights = weights

cool_perceptron = Perceptron()
print(cool_perceptron)


#12/10/2019
#Just finished one exam 
#There are still 2 left
class Perceptron:
      def __init__(self, num_inputs=2, weights=[2,1]):
    self.num_inputs = num_inputs
    self.weights = weights
    
  def weighted_sum(self, inputs):
    # create variable to store weighted sum
    weighted_sum = 0
    for i in range(self.num_inputs):
      weighted_sum += self.weights[i]* inputs[i]
      # complete this loop
    return weighted_sum 
    
cool_perceptron = Perceptron()
print(cool_perceptron.weighted_sum([24,55]))

#Training the Perceptron
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import random

def generate_training_set(num_points):
	x_coordinates = [random.randint(0, 50) for i in range(num_points)]
	y_coordinates = [random.randint(0, 50) for i in range(num_points)]
	training_set = dict()
	for x, y in zip(x_coordinates, y_coordinates):
		if x <= 45-y:
			training_set[(x,y)] = 1
		elif x > 45-y:
			training_set[(x,y)] = -1
	return training_set

training_set = generate_training_set(30)

x_plus = []
y_plus = []
x_minus = []
y_minus = []

for data in training_set:
	if training_set[data] == 1:
		x_plus.append(data[0])
		y_plus.append(data[1])
	elif training_set[data] == -1:
		x_minus.append(data[0])
		y_minus.append(data[1])
    
fig = plt.figure()
ax = plt.axes(xlim=(-25, 75), ylim=(-25, 75))

plt.scatter(x_plus, y_plus, marker = '+', c = 'green', s = 128, linewidth = 2)
plt.scatter(x_minus, y_minus, marker = '_', c = 'red', s = 128, linewidth = 2)

plt.title("Training Set")

plt.show()

#Training error
class Perceptron:
      def __init__(self, num_inputs=2, weights=[1,1]):
    self.num_inputs = num_inputs
    self.weights = weights
    
  def weighted_sum(self, inputs):
    weighted_sum = 0
    for i in range(self.num_inputs):
      weighted_sum += self.weights[i]*inputs[i]
    return weighted_sum
  
  def activation(self, weighted_sum):
    if weighted_sum >= 0:
      return 1
    if weighted_sum < 0:
      return -1
    
  def training(self, training_set):
    for inputs in training_set:
      prediction = self.activation(self.weighted_sum(inputs))
      actual = training_set[inputs]
      error = actual - prediction
      
cool_perceptron = Perceptron()
print(cool_perceptron.weighted_sum([24, 55]))
print(cool_perceptron.activation(52))


#Perceptron algorithm 
class Perceptron:
      def __init__(self, num_inputs=2, weights=[1,1]):
    self.num_inputs = num_inputs
    self.weights = weights
    
  def weighted_sum(self, inputs):
    weighted_sum = 0
    for i in range(self.num_inputs):
      weighted_sum += self.weights[i]*inputs[i]
    return weighted_sum
  
  def activation(self, weighted_sum):
    if weighted_sum >= 0:
      return 1
    if weighted_sum < 0:
      return -1
    
  def training(self, training_set):
    foundLine = False
    while not foundLine:
      total_error = 0
      for inputs in training_set:
        prediction = self.activation(self.weighted_sum(inputs))
        actual = training_set[inputs]
        error = actual - prediction
        total_error += abs(error)
        for i in range(self.num_inputs):
          self.weights[i] += error*inputs[i]
      if total_error == 0:
        foundLine = True
        
cool_perceptron = Perceptron()
small_training_set = {(0,3):1, (3,0):-1, (0,-3):-1, (-3,0):1}

cool_perceptron.training(small_training_set)


#The bias weight 
class Perceptron:
      def __init__(self, num_inputs=3, weights=[1,1,1]):
    self.num_inputs = num_inputs
    self.weights = weights
    
  def weighted_sum(self, inputs):
    weighted_sum = 0
    for i in range(self.num_inputs):
      weighted_sum += self.weights[i]*inputs[i]
    return weighted_sum
  
  def activation(self, weighted_sum):
    if weighted_sum >= 0:
      return 1
    if weighted_sum < 0:
      return -1
    
  def training(self, training_set):
    foundLine = False
    while not foundLine:
      total_error = 0
      for inputs in training_set:
        prediction = self.activation(self.weighted_sum(inputs))
        actual = training_set[inputs]
        error = actual - prediction
        total_error += abs(error)
        for i in range(self.num_inputs):
          self.weights[i] += error*inputs[i]
      if total_error == 0:
        foundLine = True
      
cool_perceptron = Perceptron()