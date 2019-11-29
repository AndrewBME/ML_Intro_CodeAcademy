#11/24/2019
from movies import movie_dataset, movie_ratings

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def predict(unknown, dataset, movie_ratings, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  sum = 0
  for neighbor in neighbors:
    title = neighbor[1]
    sum += movie_ratings[title]
  return sum/len(neighbors)

print(movie_dataset["Life of Pi"])
print(movie_ratings["Life of Pi"])
print(predict([0.016, 0.300, 1.022], movie_dataset, movie_ratings, 5))

#Weighted KNN
from movies import movie_dataset, movie_ratings

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def predict(unknown, dataset, movie_ratings, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  numerator = 0
  denominator = 0
  for neighbor in neighbors:
    rating = movie_ratings[neighbor[1]]
    distance_to_neighbor = neighbor[0]
    numerator += rating / distance_to_neighbor
    denominator += 1 / distance_to_neighbor
  return numerator / denominator

print(predict([0.016, 0.300, 1.022], movie_dataset, movie_ratings, 5))

#Scikit Learn
from movies import movie_dataset, movie_ratings
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors = 5, weights = "distance")
regressor.fit(movie_dataset,movie_ratings)
print(regressor.predict([[0.016, 0.300, 1.022], [0.0004092981, 0.283, 1.0112], [0.00687649, 0.235, 1.0112]]))


#Two types of normalization 
#Max-min normalization: (value-min)/(max- min)
#Z normalization: (value - µ)/sigma (µ is mean and sigma is standard deviation)