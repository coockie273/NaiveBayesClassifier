# Naive Bayes Classifier
Naive Bayes classifier is simple machine learning model that is based on Bayes's theorem:

$$ P(y = c|x) = \frac{P(x|y = c)P(y=c)}{P(x)} $$

The main idea behind this model is to find a class for an object with the maximum probability of P(y=c|x).

$$ c_{opt} = arg max_{c \in C} P(x|y = c)P(y=c) $$

The Naive Bayes classifier represent an object as a set of features whose probabilities are conditionally independent from each other.

$$ c_{opt} = arg max_{c \in C} ( P(y=c) M^{m}_{i=1} P(f_i|y = c)) $$

or

$$ c_{opt} = arg max_{c \in C} ( log(P(y=c)) + \sum^{m}_{i=1} log(P(f_i|y = c))) $$

Possibility $P(f_i|y = c)$ is estimated from the training sample as

$$P(f_i|y = c) = \frac{M_i(c) + \alpha}{\sum^{m}_{j=1} (M_j(c) + \alpha)}$$

$\alpha$ - parameter that helps to avoid zeros values of probabilities

Or in the continuous case:

$$P(f_i|y = c) = \frac{1}{2nh} \sum^{m}_{j=1}  [ |x - x_i| < h]$$

h - positive parameter called window size.

The accuracy of that model in this project will measure with **Shuffle-Split Cross-Validation** method.

# Usage

The use of this model consists of three stages: initialization, training, and classification.
- Initialization - set the parameteres of the model:
  - target index in data set,
  -  $alpha$,
  -  *h*.
- Training - calculate the priori probability of each class and conditional probability for each feature value based on the training data.
- classification - use the priori and conditional probabilities to calculate the probability that a given object belongs to each class.

# Testing

The realized model was tested on two datasets: "Titanic" and "MNIST". The model

