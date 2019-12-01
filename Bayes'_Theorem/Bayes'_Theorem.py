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

