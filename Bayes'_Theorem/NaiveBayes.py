##11/30/2019##
#Nave Bayes Classifier 
from reviews import neg_list, pos_list, neg_counter, pos_counter
print(pos_list[0])
print(neg_list[0])

#Count the number of 'no' appeared
print(pos_counter['no'])


#Bayes Theorem I
from reviews import neg_list, pos_list, neg_counter, pos_counter
print(len(pos_list))
print(len(neg_list))

total_reviews = len(pos_list) + len(neg_list)
print(total_reviews)

percent_pos = len(pos_list)/total_reviews 
percent_neg = len(neg_list)/total_reviews 
print(percent_pos)
print(percent_neg)

#Bayes Theorem II
from reviews import neg_counter, pos_counter

review = "This crib was amazing"

percent_pos = 0.5
percent_neg = 0.5

#Find the total number of words in postive reviews 
total_pos = sum(pos_counter.values())
total_neg = sum(neg_counter.values())
#print(total_pos)
pos_probability = 1
neg_probability = 1

review_words = review.split()

for word in review_words:
  word_in_pos = pos_counter[word]
  word_in_neg = neg_counter[word]
  pos_probability *= word_in_pos / total_pos
  neg_probability *= word_in_neg / total_neg

#Smoothing 
from reviews import neg_counter, pos_counter

review = "This cribb was amazing"

percent_pos = 0.5
percent_neg = 0.5

total_pos = sum(pos_counter.values())
total_neg = sum(neg_counter.values())

pos_probability = 1
neg_probability = 1

review_words = review.split()

for word in review_words:
  word_in_pos = pos_counter[word]
  word_in_neg = neg_counter[word]
  
  pos_probability *= (word_in_pos + 1) / (total_pos + len(pos_counter))
  neg_probability *= ( word_in_neg +1)/ (total_neg+ len(neg_counter))
  
print(pos_probability)
print(neg_probability)

from reviews import neg_counter, pos_counter

review = "This crib was terrible"

percent_pos = 0.5
percent_neg = 0.5

total_pos = sum(pos_counter.values())
total_neg = sum(neg_counter.values())

pos_probability = 1
neg_probability = 1

review_words = review.split()

for word in review_words:
  word_in_pos = pos_counter[word]
  word_in_neg = neg_counter[word]
  
  pos_probability *= (word_in_pos + 1) / (total_pos + len(pos_counter))
  neg_probability *= (word_in_neg + 1) / (total_neg + len(neg_counter))

final_pos = pos_probability * percent_pos 
final_neg = neg_probability * percent_neg
if final_pos > final_neg:
  print("The review is positive")
else:
  print("The review is negative")

#Data formatting with scikit-learn 
from reviews import neg_list, pos_list
from sklearn.feature_extraction.text import CountVectorizer

review = "This crib was amazing"

counter = CountVectorizer()

counter.fit(neg_list + pos_list)

print(counter.vocabulary_)
review_counts = counter.transform({"this": 1, "crib": 1, "was": 1, "amazing": 1})

training_counts = counter.transform(neg_list + pos_list)