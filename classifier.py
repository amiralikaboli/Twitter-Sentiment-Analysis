import random
import re
import string

import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    average_precision_score
from sklearn.svm import SVC

from sentiment_analysis.tokenizer import tokenize


class SentimentAnalyzer:
    def __init__(self):
        self.sentiment2int = {'positive': 1, 'negative': -1, 'neutral': 0}

        self.punctuations = string.punctuation
        self.stop_words = set(stopwords.words('english'))

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def load_dataset(self, dataset_path: str) -> list:
        data_points = []
        data = pd.read_csv(dataset_path)
        for ind in range(data.shape[0]):
            sentiment_sign = self.sentiment2int[data['airline_sentiment'][ind]]
            # sentiment_confidence_rank = 1
            if data['airline_sentiment:confidence'][ind] < 0.6:
                sentiment_confidence_rank = 1
            elif data['airline_sentiment:confidence'][ind] < 1:
                sentiment_confidence_rank = 2
            else:
                sentiment_confidence_rank = 3

            if data['retweet_count'][ind] >= 2 and sentiment_confidence_rank != 3:
                sentiment_confidence_rank += 1
            if data['retweet_count'][ind] == 0 and sentiment_confidence_rank != 1:
                sentiment_confidence_rank -= 1

            data_points.append({
                'text': data['text'][ind],
                'sentiment': sentiment_sign * sentiment_confidence_rank,
            })

        return data_points

    def preprocess_tweet_text(self, tweet_text: str) -> str:
        tweet_text = tweet_text.lower()

        tweet_text = re.sub(r"http\S+|www\S+|https\S+", '', tweet_text, flags=re.MULTILINE)

        tweet_tokens = tokenize(tweet_text)
        tweet_tokens = [tweet_token for tweet_token in tweet_tokens if len(tweet_token) >= 3 or tweet_token == ':)']

        tweet_tokens = [tweet_token for tweet_token in tweet_tokens if tweet_token[0] not in ['@', '#']]

        tweet_tokens = [tweet_token for tweet_token in tweet_tokens if tweet_token not in self.punctuations]

        filtered_words = [w for w in tweet_tokens if w not in self.stop_words]

        stemmed_words = [self.stemmer.stem(w) for w in filtered_words]

        # lemma_words = [self.lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

        return " ".join(stemmed_words)

    def get_vocabulary(
            self,
            dataset_texts: list,
            dataset_labels: list,
            vocabulary_length: int
    ) -> list:
        class2index = {unique_class: ind for ind, unique_class in enumerate(self.unique_classes)}

        word2index = {unique_word: ind for ind, unique_word in enumerate(self.unique_words)}

        count_matrix = np.zeros((len(self.unique_classes), len(self.unique_words)))
        for dataset_text, dataset_label in zip(dataset_texts, dataset_labels):
            tokens = dataset_text.split()
            for token in tokens:
                count_matrix[class2index[dataset_label]][word2index[token]] += 1

        chi_squared_scores = {}
        for class_ind in range(len(self.unique_classes)):
            for word_ind in range(len(self.unique_words)):
                observed = count_matrix[class_ind][word_ind]
                expected = count_matrix.sum(axis=0)[word_ind] * count_matrix.sum(axis=1)[class_ind] / count_matrix.sum()
                chi_squared_score = (observed - expected) ** 2 / expected

                chi_squared_scores[(self.unique_classes[class_ind], self.unique_words[word_ind])] = chi_squared_score

        vocabulary = set()
        for key in sorted(chi_squared_scores, key=chi_squared_scores.get, reverse=True):
            vocabulary.add(key[1])

            if len(vocabulary) == vocabulary_length:
                break

        return list(vocabulary)

    def train(self):
        training_data_points = self.load_dataset('data/airline-train.csv')
        training_set_texts = [self.preprocess_tweet_text(data_point['text']) for data_point in training_data_points]
        training_set_sentiments = [data_point['sentiment'] for data_point in training_data_points]

        validation_data_points = self.load_dataset('data/airline-dev.csv')
        validation_set_texts = [self.preprocess_tweet_text(data_point['text']) for data_point in validation_data_points]
        validation_set_sentiments = [data_point['sentiment'] for data_point in validation_data_points]

        self.unique_classes = sorted(np.unique(training_set_sentiments))

        unique_words = set()
        for dataset_text in training_set_texts:
            tokens = dataset_text.split()
            unique_words = unique_words.union(tokens)
        self.unique_words = sorted(list(unique_words))

        num_generation = 10
        population_size = 10

        vocabulary_size_accuracies = {}
        population = random.sample(range(1, len(unique_words) + 1), population_size)
        for _ in range(num_generation):
            children = []
            for __ in range(population_size):
                parent1 = population[random.randint(0, population_size - 1)]
                parent2 = population[random.randint(0, population_size - 1)]
                children.append((parent1 + parent2) // 2)

            new_population_with_accuracy = {}
            for vocabulary_size in population + children:
                if vocabulary_size not in vocabulary_size_accuracies:
                    vocabulary = self.get_vocabulary(
                        dataset_texts=training_set_texts,
                        dataset_labels=training_set_sentiments,
                        vocabulary_length=vocabulary_size
                    )

                    tmp_vectorizer = CountVectorizer(vocabulary=vocabulary)
                    training_set_vectors = tmp_vectorizer.fit_transform(training_set_texts)

                    tmp_classifier = SVC()
                    tmp_classifier.fit(training_set_vectors, training_set_sentiments)

                    validation_set_vectors = tmp_vectorizer.transform(validation_set_texts)

                    validation_set_predictions = tmp_classifier.predict(validation_set_vectors)

                    accuracy = accuracy_score(validation_set_sentiments, validation_set_predictions)
                    vocabulary_size_accuracies[vocabulary_size] = accuracy

                new_population_with_accuracy[vocabulary_size] = vocabulary_size_accuracies[vocabulary_size]

            population = sorted(new_population_with_accuracy, key=new_population_with_accuracy.get, reverse=True)[
                         :population_size]

        self.vocabulary = self.get_vocabulary(
            dataset_texts=training_set_texts,
            dataset_labels=training_set_sentiments,
            vocabulary_length=population[0]
        )

        self.vectorizer = CountVectorizer(vocabulary=self.vocabulary)
        training_set_vectors = self.vectorizer.fit_transform(training_set_texts)

        self.classifier = SVC()
        self.classifier.fit(training_set_vectors, training_set_sentiments)

    def evaluate(self) -> dict:
        test_data_points = self.load_dataset('data/airline-test.csv')

        test_set_texts = [self.preprocess_tweet_text(data_point['text']) for data_point in test_data_points]
        test_set_sentiments = [data_point['sentiment'] for data_point in test_data_points]

        test_set_vectors = self.vectorizer.transform(test_set_texts)

        test_set_predictions = self.classifier.predict(test_set_vectors)

        return {
            'accuracy': accuracy_score(test_set_sentiments, test_set_predictions),
            'precision': [
                precision_score(test_set_sentiments == np.ones(len(test_set_sentiments)) * unique_class,
                                test_set_predictions == np.ones(len(test_set_predictions)) * unique_class)
                for unique_class in self.unique_classes
            ],
            'average-precision': [
                average_precision_score(test_set_sentiments == np.ones(len(test_set_sentiments)) * unique_class,
                                        test_set_predictions == np.ones(len(test_set_predictions)) * unique_class)
                for unique_class in self.unique_classes
            ],
            'recall': [
                recall_score(test_set_sentiments == np.ones(len(test_set_sentiments)) * unique_class,
                             test_set_predictions == np.ones(len(test_set_predictions)) * unique_class)
                for unique_class in self.unique_classes
            ],
            'f1-micro': f1_score(test_set_sentiments, test_set_predictions, average='micro'),
            'f1-macro': f1_score(test_set_sentiments, test_set_predictions, average='macro'),
            'confusion-matrix': confusion_matrix(test_set_sentiments, test_set_predictions).tolist(),
        }

    def predict(self, tweet_text):
        cleaned_text = self.preprocess_tweet_text(tweet_text)

        vec = self.vectorizer.transform([cleaned_text])

        sentiment_class = self.classifier.predict([vec])

        return sentiment_class


if __name__ == '__main__':
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.train()
    result = sentiment_analyzer.evaluate()
