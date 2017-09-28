import csv
from collections import Counter
from decimal import Decimal
import re
import sys


class Classifier(object):
    def tokenize(self, sentence: str):
        cleaned_string = re.sub("[\W]", " ", sentence)
        return cleaned_string.lower().split()

    def _calculate_probablility(self, trained_data: dict, classification: str, tokens: list, delta=0.01):
        """
        Calculates the add-delta probability P(tokens | details)
        """
        if classification not in trained_data:
            return 0

        token_counts = trained_data[classification]
        total_words_in_category = sum(val for val in token_counts.values())
        vocab_size = len(token_counts.keys())
        probability = Decimal('1')
        for token in tokens:
            counts = Decimal(token_counts.get(token, 0) + delta)
            token_probability = counts / Decimal(total_words_in_category + (delta * vocab_size))
            probability = probability * token_probability

        return probability

    def train(self, file_name: str):
        """
        Expects a csv with two rows; the first containing a category, and the second containing a sentence in that category.
        """
        category_to_token_count_map = {}

        with open(file_name, encoding="utf8") as csv_file:
            reader = csv.reader(csv_file)
            for (classification, sentence) in reader:
                counter = category_to_token_count_map.setdefault(classification, Counter())
                counter.update(self.tokenize(sentence))

        return category_to_token_count_map

    def classify(self, trained_data: dict, raw_input: str):
        tokens = self.tokenize(raw_input)

        probability, most_likely_classification = max([
            (self._calculate_probablility(trained_data, classification, tokens), classification)
            for classification in trained_data.keys()
        ] + [(float("-Inf"), None)])

        if probability == Decimal("0"):
            raise Exception("No classification found")

        return most_likely_classification


def main():
    csv_file_path = sys.argv[1]
    classifier = Classifier()
    trained_data = classifier.train(csv_file_path)

    while 1:
        new_input = input('Enter a new sentence: ')
        print('Most likely category: {}'.format(classifier.classify(trained_data, new_input)))
        print("")


if __name__ == '__main__':
    main()
