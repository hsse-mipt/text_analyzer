import pandas as pd
from nltk.corpus import stopwords
from string import punctuation
import pymorphy2
import re

from tqdm import tqdm


class WordEncoder:
    ru_stopwords = set(stopwords.words("russian"))
    morph = pymorphy2.MorphAnalyzer()

    @staticmethod
    def is_trash_word(normal_form, parse_word):
        return normal_form in WordEncoder.ru_stopwords or WordEncoder.is_name(parse_word) or any(x.isdigit() for x in normal_form)

    @staticmethod
    def fit_transform(sentences: list):
        one_hot_words_columns = set()
        for sentence in tqdm(sentences):
            sentence = WordEncoder.preprocess_sentence(sentence)
            for word in sentence:
                parse_word = WordEncoder.morph.parse(word)
                normal_form = parse_word[0].normal_form
                if not (WordEncoder.is_trash_word(normal_form, parse_word)):
                    one_hot_words_columns.add(normal_form)
        one_hot_words_columns = list(one_hot_words_columns)
        data = [[0] * len(one_hot_words_columns) for _ in range(len(sentences))]
        for i in tqdm(range(len(data))):
            sentence = WordEncoder.preprocess_sentence(sentences[i])
            for j in range(len(data[i])):
                data[i][j] = 1 if one_hot_words_columns[j] in sentence else 0
        return pd.DataFrame(data, columns=one_hot_words_columns)

    @staticmethod
    def remove_symbols_from_text(text: str, symbols: str) -> str:
        return "".join([ch for ch in text if ch not in symbols])

    @staticmethod
    def preprocess_sentence(sentence: str) -> list[str]:
        sentence = sentence.lower()
        spec_chars = punctuation + '\n\t…—«»'
        sentence = WordEncoder.remove_symbols_from_text(sentence, spec_chars)
        words = sentence.split()
        words = list(filter(lambda word: not re.match(r'[a-z]+', word), words))
        return words

    @staticmethod
    def is_name(parse_word, threshold_prob=0.5) -> bool:
        for p in parse_word:
            if 'Name' in p.tag and p.score >= threshold_prob:
                return True
        return False

