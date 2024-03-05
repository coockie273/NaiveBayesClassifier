import random


# The Shuffle-Split Cross-Validation method for evaluating classification accuracy
def cross_validation_accuracy(model, data, split_ratio=0.8, k=10):
    accuracy = 0

    for i in range(k):

        random.shuffle(data)

        # Dividing the dataset into a training and testing sample
        split_index = int(len(data) * split_ratio)
        train_data = data[:split_index]
        test_data = data[split_index:]

        # The train of the model
        model.train(train_data)

        # The testing of the model
        count = 0
        for test in test_data:
            if model.predicate_log(test[:model.target_index] + test[model.target_index + 1:]) == test[model.target_index]:
                count += 1

        accuracy += count / len(test_data)

    return accuracy / k
