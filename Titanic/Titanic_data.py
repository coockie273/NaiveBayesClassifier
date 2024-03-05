# Testing accuracy of the naive Bayes classification with dataset "Titanic"

from src.CSVReader import CSVReader
from src.naiveBayesClassifier import NaiveBayesClassifier
import csv

train_file_path = 'Titanic_train.csv'
test_file_path = 'Titanic_test.csv'
result_file_path = 'Titanic_result.csv'

# Import train dataset with typification
titanic_train = CSVReader.read_csv(train_file_path, [int, int, int, str, str, float, int, int, str, float, str, str], head=False)

# Initialization model
model = NaiveBayesClassifier(1)

# Train
model.train(titanic_train)

# Import test dataset with typification
titanic_test = CSVReader.read_csv(test_file_path,
                                   [int, int, int, str, str, float, int, int, str, float, str, str], head=False)

# Export result in csv file
results = [["PassengerId", "Survived"]]

for test in titanic_test:
    id = test[0]
    result = model.predicate_mult(test)
    results.append([id, result])

with open(result_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    for row in results:
        csv_writer.writerow(row)
