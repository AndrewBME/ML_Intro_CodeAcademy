#11/27/2019
#Predict Titanic Survival
import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
print(passengers.head())
# Update sex column to numerical
passengers['Sex'] = passengers(['Sex']).map({'female': '1', 'male':'0'})
print(passengers.head())
# Fill the nan values in the age column
print(passengers['Age'].values)
#Since there are some Nan
#Fill them with average 
print(passengers['Age']).values
passengers(['Age']).fillna(value = round(passengers['Age'].mean()), inplace = True)
# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)
print(passengers)

# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: if x == 2 else 0)

# Select the desired features
#selection =  data_frame[['column_1','column_2', ..., 'column_n']]
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

# Perform train, test, split
train_features, test_features, train_labels, test_labels = train_test_split(features,survival)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
# Create and train the model
model = LogistRegressions
model.fit(train_features, test_features)

# Score the model on the train data
print(model.score(train_features, train_labels))

# Score the model on the test data
print(model.score(test_features, test_labels))

# Analyze the coefficients
print(model.coef_)

#Analyze the coefficient of each column
print(list(zip(['Sex','Age','FirstClass','SecondClass'],model.coef_[0])))

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([0.0, 27,0.0, 1.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, You])

# Scale the sample passenger features
sample_passengers = scaler.transform(sammple_passengers)


# Make survival predictions!
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))