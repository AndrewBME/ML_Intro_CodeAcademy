import codecademylib3_seaborn
import numpy as np
import matplotlib.pyplot as plt
from exam import hours_studied, passed_exam, math_courses_taken

# Scatter plot of exam passage vs number of hours studied
plt.scatter(hours_studied.ravel(), passed_exam, color='black', zorder=20)
plt.ylabel('passed/failed')
plt.xlabel('hours studied')

plt.show()

import codecademylib3_seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from exam import hours_studied, passed_exam
from plotter import plot_data

# Create linear regression model
model = LinearRegression()
model.fit(hours_studied,passed_exam)

# Plug sample data into fitted model
sample_x = np.linspace(-16.65, 33.35, 300).reshape(-1,1)
probability = model.predict(sample_x).ravel()

# Function to plot exam data and linear regression curve
plot_data(model)

# Show the plot
plt.show()

# Define studios and slacker here
slacker = -0.1
studious = 1.1

import codecademylib3_seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from exam import hours_studied, passed_exam
from plotter import plot_data

# Create logistic regression model
model = LogisticRegression()
model.fit(hours_studied,passed_exam)

# Plug sample data into fitted model
sample_x = np.linspace(-16.65, 33.35, 300).reshape(-1,1)
probability = model.predict_proba(sample_x)[:,1]

# Function to plot exam data and logistic regression curve
plot_data(model)

# Show the plot
plt.show()

# Lowest and highest probabilities
lowest = 0.0 
highest = 1.0

import numpy as np
from exam import hours_studied, calculated_coefficients, intercept

# Create your log_odds() function here
def log_odds(features, coefficients, intercept):
  return np.dot(features, coefficients) + intercept

# Calculate the log-odds for the Codecademy University data here
calculated_log_odds = log_odds(hours_studied, calculated_coefficients, intercept)
print(calculated_log_odds)

import numpy as np
from exam import passed_exam, probabilities, probabilities_2

# Function to calculate log-loss
def log_loss(probabilities,actual_class):
  return np.sum(-(1/actual_class.shape[0])*(actual_class*np.log(probabilities) + (1-actual_class)*np.log(1-probabilities)))

# Print passed_exam here
print(passed_exam)

# Calculate and print loss_1 here
loss_1 = log_loss(probabilities, passed_exam)


# Calculate and print loss_2 here
loss_2 = log_loss(probabilities_2, passed_exam)

#Threshold 
#np.where()
import numpy as np
from exam import hours_studied, calculated_coefficients, intercept

def log_odds(features, coefficients,intercept):
  return np.dot(features,coefficients) + intercept

def sigmoid(z):
    denominator = 1 + np.exp(-z)
    return 1/denominator

# Create predict_class() function here

def predict_class(features,coefficients,intercept,threshold):
  calculated_log_odds = log_odds(features,coefficients,intercept)
  probabilities = sigmoid(calculated_log_odds)
  return np.where(probabilities >= threshold, 1, 0)

# Make final classifications on Codecademy University data here

final_results = predict_class(hours_studied, calculated_coefficients, intercept, 0.5)
print(final_results)

#11/27/2019
#Hope I can make it outta here
import numpy as np
from sklearn.linear_model import LogisticRegression
from exam import hours_studied_scaled, passed_exam, exam_features_scaled_train, exam_features_scaled_test, passed_exam_2_train, passed_exam_2_test, guessed_hours_scaled

# Create and fit logistic regression model here
model = LogisticRegression()
model.fit(hours_studied_scaled, passed_exam)

# Save the model coefficients and intercept here
calculated_coefficients = model.coef_
print(calculated_coefficients)
intercept = model.intercept_
print(intercept)

# Predict the probabilities of passing for next semester's students here
passed_predictions = model.predict_proba(guessed_hours_scaled)

# Create a new model on the training data with two features here
model_2 = LogisticRegression()
model_2.fit(exam_features_scaled_train, passed_exam_2_train)

# Predict whether the students will pass here
passed_predictions_2 = model_2.predict(exam_features_scaled_test)
print(passed_predictions_2)

#Feature importance 
import codecademylib3_seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from exam import exam_features_scaled, passed_exam_2

# Train a sklearn logistic regression model on the normalized exam data
model_2 = LogisticRegression()
model_2.fit(exam_features_scaled,passed_exam_2)

# Assign and update coefficients
coefficients = model_2.coef_
print(coefficients)

coefficients = coefficients.tolist()[0]
# Plot bar graph
plt.bar([1,2], coefficients)
plt.xticks([1,2], ['hours studied', 'math courses taken'])
plt.xlabel('feature')
plt.ylabel('coefficient')
plt.show()

