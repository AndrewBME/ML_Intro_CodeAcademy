#11/30/2019
#Independent Events

#Conditional Probability 
import numpy as np

p_rain = 0.3
p_gym = 0.6
p_rain_and_gym = p_rain * p_gym 
print(p_rain_and_gym)

#Test for a disease
import numpy as np

p_disease = 1.0 / 100000
p_correct = 0.99
p_disease_and_correct = p_disease * p_correct 
print(p_disease_and_correct )

p_no_disease_and_incorrect = (99999.0 / 100000) * 0.01
print(p_no_disease_and_incorrect)

#Bayes' Theorem 
import numpy as np

#The probability of positive when given a patient
p_positive_given_disease = (0.99 * (.00001))/ (1./100000.)
print(p_positive_given_disease)

#Possibility of being infected
p_disease = 1./100000.
print(p_disease)

#Possibility of the result is positive
#Both truly infected and not 
p_positive = (0.00001) + (0.01) 
print(p_positive)

#calculate the total probability that a randomly selected patient receives a positive test result
p_disease_given_positive = (p_positive_given_disease) * (p_disease) / (p_positive)

print(p_disease_given_positive)

#Spam filter
import numpy as np
a = 'spam'
b = 'enhancement'

#The following values are given 
p_spam = 0.2

p_enhancement_given_spam = 0.05

p_enhancement = 0.2*0.05 + 0.8* 0.001

#Given enhancement, it is spam
p_spam_enhancement = 0.05*0.2/(0.2*0.05 + 0.8* 0.001)

print(p_spam_enhancement)

#12/1/2019
from reviews import counter, training_counts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

review = "This crib was amazing"
review_counts = counter.transform([review])

classifier = MultinomialNB()

#Make training labels
training_labels = [0] * 1000 + [1] * 1000

#Fit and predict
classifier.fit(training_counts, training_labels)
print(classifier.predict(review_counts))
print(classifier.predict_proba(review_counts))

#Review Session 
from reviews import baby_counter, baby_training, instant_video_counter, instant_video_training, video_game_counter, video_game_training
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

review = "this game was lit"

baby_review_counts = baby_counter.transform([review])
instant_video_review_counts = instant_video_counter.transform([review])
video_game_review_counts = video_game_counter.transform([review])

baby_classifier = MultinomialNB()
instant_video_classifier = MultinomialNB()
video_game_classifier = MultinomialNB()

baby_labels = [0] * 1000 + [1] * 1000
instant_video_labels = [0] * 1000 + [1] * 1000
video_game_labels = [0] * 1000 + [1] * 1000


baby_classifier.fit(baby_training, baby_labels)
instant_video_classifier.fit(instant_video_training, instant_video_labels)
video_game_classifier.fit(video_game_training, video_game_labels)

print("Baby training set: " +str(baby_classifier.predict_proba(baby_review_counts)))
print("Amazon Instant Video training set: " + str(instant_video_classifier.predict_proba(instant_video_review_counts)))
print("Video Games training set: " + str(video_game_classifier.predict_proba(video_game_review_counts)))
