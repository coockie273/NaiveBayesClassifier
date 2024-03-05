from scipy.stats import norm
from src.naiveBayesClassifier import NaiveBayesClassifier
from src.utils import cross_validation_accuracy


points_train = []

# Points generation for class 1
X1 = norm(loc=10, scale=4)
X2 = norm(loc=14, scale=4)

for i in range(100):
    point = [X1.rvs(), X2.rvs(), -1]
    points_train.append(point)

X1 = norm(loc=20, scale=3)
X2 = norm(loc=18, scale=3)

# Points generation for class 2
for i in range(100):
    point = [X1.rvs(), X2.rvs(), 1]
    points_train.append(point)

model = NaiveBayesClassifier(2, window_size=3)

acc = cross_validation_accuracy(model, points_train, split_ratio=0.5, k=100)

print(acc)

