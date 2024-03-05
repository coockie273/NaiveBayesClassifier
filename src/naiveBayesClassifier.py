from math import log


# The NaiveBayesClassifier
class NaiveBayesClassifier:

    # Initialization of the model, setting the required parameters:
    def __init__(self, target_index, alpha=1, window_size=0.5):
        self.alpha = alpha
        self.window_size = window_size
        self.target_index = target_index

    # Model training, calculation of the conditional and prior probabilities
    def train(self, data):
        self.sample_size = len(data)
        self.feature_size = len(data[0])

        self.conditional_probabilities = [{} for _ in range(self.feature_size)]
        self.prior_probabilities = {}

        for i in range(self.feature_size):
            for j in range(self.sample_size):

                # Calculation of the conditional probabilities
                if i != self.target_index:

                    # Value approximation for feature with continuous value
                    if isinstance(data[j][i], (int, float)):
                        data[j][i] = self._approximate_value(self.conditional_probabilities[i], data[j][i])

                    if data[j][i] not in self.conditional_probabilities[i]:
                        self.conditional_probabilities[i][data[j][i]] = {}

                    target_data = data[j][self.target_index]

                    if target_data in self.conditional_probabilities[i][data[j][i]]:
                        self.conditional_probabilities[i][data[j][i]][target_data] += 1
                    else:
                        self.conditional_probabilities[i][data[j][i]][target_data] = 1

                # Calculation of the prior probabilities
                else:
                    target_data = data[j][self.target_index]

                    if target_data in self.prior_probabilities:
                        self.prior_probabilities[target_data] += 1
                    else:
                        self.prior_probabilities[target_data] = 1

        del self.conditional_probabilities[self.target_index]

        for feature in self.conditional_probabilities:
            for target_key in feature.keys():
                for target_value in feature[target_key].keys():
                    feature[target_key][target_value] = \
                        (feature[target_key][target_value] + self.alpha) / (self.sample_size + self.alpha)

        self.prior_probabilities = {key: value / len(data) for key, value in self.prior_probabilities.items()}

    # The method approximates value of the feature with continuous value according to the set value of the window size
    def _approximate_value(self, map, value):

        close_value = [key for key in map.keys() if value - self.window_size <= key <= value + self.window_size]
        if close_value:
            value = close_value[0]
        return value

    # Two ways of the predication:
    # - mult: Multiplying all conditional probabilities by a priori
    # - log: Sum of the logarithms of all conditional probabilities per and the priori
    # return: maximum probability value among all classes
    def predicate_mult(self, raw):
        final_probabilities = dict(self.prior_probabilities)

        for target_value in self.prior_probabilities.keys():
            p = self.prior_probabilities[target_value]

            for i in range(len(raw)):
                try:
                    # Value approximation for feature with continuous value
                    if isinstance(raw[i], (int, float)):
                        raw[i] = self._approximate_value(self.conditional_probabilities[i], raw[i])
                    p *= self.conditional_probabilities[i][raw[i]][target_value]
                except KeyError:
                    p *= self.alpha / (self.sample_size + self.alpha)

            final_probabilities[target_value] = p

        return max(final_probabilities, key=final_probabilities.get)

    def predicate_log(self, raw):
        final_probabilities = dict(self.prior_probabilities)

        for target_value in self.prior_probabilities.keys():
            p = log(self.prior_probabilities[target_value])

            for i in range(len(raw)):
                try:
                    # Value approximation for feature with continuous value
                    if isinstance(raw[i], (int, float)):
                        raw[i] = self._approximate_value(self.conditional_probabilities[i], raw[i])
                    p += log(self.conditional_probabilities[i][raw[i]][target_value])
                except KeyError:
                    p += log(self.alpha / (self.sample_size + self.alpha))

            final_probabilities[target_value] = p

        return max(final_probabilities, key=final_probabilities.get)

    
