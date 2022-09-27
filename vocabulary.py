"""
Search keywords or phrases in different vocabularies previously loaded from outer storage.

"""
import json
import numpy as np

from collections import defaultdict
from fuzzywuzzy import fuzz
from nacl.exceptions import ValueError
from nltk import ngrams
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union, Tuple

from .embeddings import NavecEmb
from ..nlp import ESNLP


class VocabularyStats:
    """
    Main manager that contains all vocabularies and searches keywords entries in the input text.

    Attributes
    ----------
    embedder : EmbedderBase
        base model to calculate word vectors
    nlp : ESNLP
        text preprocessor
    basic_threshold : float
        threshold for cosine distance between text and keyword vectors
    typos_threshold : int
        threshold for levenshtein distance between misspelled text and keyword vectors
    vocabularies : dict
        all available vocabularies

    """

    def __init__(self,  nlp: ESNLP = None, config: Union[str, dict] = None,
                 basic_threshold: float = 0.9, typos_threshold: int = 80):
        if nlp is None:
            self.nlp = ESNLP(config)
        else:
            self.nlp = nlp
        self.embedder = NavecEmb()
        self.basic_threshold = basic_threshold
        self.typos_threshold = typos_threshold
        self.vocabularies = None
        self.vocabulary_ids, self.vocabulary_embeds = None, None
        self.keywords, self.vocabulary_ngram_ids = None, None

    def get_full_stats(self, text: str, json_format: bool = False,
                       check_typos: bool = True) -> Union[dict, str]:
        """
        Search for common words and phrases in the given text and predefined vocabularies.

        Parameters
        ----------
        text : str
            full text for searching

        json_format : bool
            flag for using json for the return value

        check_typos : bool
            flag for using search through misprints

        Returns
        ----------
        vocabulary_result: dict or str
            information about founded words in different dictionaries

        Note
        ---------
        {
            "idx_словаря" :  [
                {
                    "idx":    int - word index in the full text,
                    "length": int - number of founded words (word or n-gram)
                    "proba":  float - measure of coincidence (changes in [0..1])
                }
            ]
        }
        """
        vocabulary_result = defaultdict(list)
        try:
            self._check_vocabularies()
            text_tokens = self.nlp.tokenize(text, lemmatize=True, clear_symbols=True, stopwords=False)
            text_embeds = [self.embedder.count_embedding(token) for token in text_tokens]
            try:
                # Count cosine similarity to single word from different vocabularies
                text_ids_max, text_max = self._get_max_similarity(cosine_similarity(text_embeds, self.vocabulary_embeds))
                [vocabulary_result[str(self.vocabulary_ids[text_ids_max[idx]])].append({'idx': idx, 'length': 1,
                                                                                        'proba': value if value < 1 else 1})
                 for idx, value in enumerate(text_max) if value > self.basic_threshold]

                # Count levenshtein distance to incorrect words. Only for single words.
                if check_typos:
                    typos_ids = np.squeeze(np.argwhere(text_max == 0))
                    typos_ids = [typos_ids[()]] if typos_ids.shape == () else typos_ids

                    text_ids_max, text_max = self._get_max_similarity(
                        np.array([[fuzz.ratio(text_tokens[idx], ' '.join(row)) for row in self.keywords]
                                  for idx in typos_ids]))

                    [vocabulary_result[str(self.vocabulary_ids[text_ids_max[idx]])].append({'idx': int(typos_ids[idx]),
                                                                                            'length': 1,
                                                                                            'proba': value / 100})
                     for idx, value in enumerate(text_max) if value > self.typos_threshold]
            except Exception as e:
                print("Exception in the single word calculations: {}".format(e))

            # Count cosine similarity to n-grams from different vocabularies
            try:
                for key, value in self.vocabulary_ngram_ids.items():
                    text_ngram_tokens = list(ngrams(text_tokens, int(key)))
                    ngram_ids = [idx for idx, voc_value in enumerate(self.vocabulary_ids)
                                 if voc_value in value and len(self.keywords[idx]) == int(key)]  # only n-grams allowed
                    text_ngram_ids_max, text_ngram_max = self._get_max_similarity(cosine_similarity(
                        self.embedder.count_embeddings(text_ngram_tokens),
                        np.take(self.vocabulary_embeds, ngram_ids, axis=0)))

                    [vocabulary_result[str(self.vocabulary_ids[ngram_ids[text_ngram_ids_max[idx]]]
                                           )].append({'idx': idx,
                                                      'length': int(key),
                                                      'proba': value if value < 1 else 1})
                     for idx, value in enumerate(text_ngram_max) if value > self.basic_threshold]
            except Exception as e:
                print("Exception in the n-gram calculations: {}".format(e))

        except Exception as e:
            print("Error during calculation: {}".format(e))
        finally:
            return json.dumps(vocabulary_result) if json_format else vocabulary_result

    def load_vocabularies(self, vocabularies: Union[str, list]):
        try:
            self.vocabularies = self._parse_vocabularies(vocabularies)
            (self.vocabulary_ids, self.vocabulary_embeds,
             self.keywords, self.vocabulary_ngram_ids) = self._union_vocabularies()
        except Exception as e:
            print("Vocabularies can't be loaded: {}".format(e))

    def _check_vocabularies(self):
        if not all([self.vocabulary_ids, self.vocabulary_embeds, self.vocabulary_ngram_ids, self.keywords]):
            raise ValueError("Vocabularies data has to be loaded before")

    def _parse_vocabularies(self, vocabularies: Union[str, list]) -> list:
        """Parse vocabulary in json string to dict if necessary"""
        if type(vocabularies) == str:
            return json.loads(vocabularies)
        elif type(vocabularies) == list:
            return vocabularies
        else:
            raise TypeError(
                "Vocabularies needs JSON (str) or list type."
            )

    def _get_max_similarity(self, similarity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Search and return max similarities and it's indices through columns in given matrix"""
        if len(similarity) == 0:
            return np.array([]), np.array([])
        text_ids_max = np.argmax(similarity, axis=1)
        return text_ids_max, similarity[np.arange(similarity.shape[0]), text_ids_max]

    def _union_vocabularies(self):
        """Transform vocabularies to vectors"""
        vocabulary_ids = []
        vocabulary_embeds = []
        keywords = []
        vocabulary_ngram_ids = defaultdict(list)
        try:
            for row in self.vocabularies:
                keys = [self.nlp.tokenize(phrase, lemmatize=True,
                                          clear_symbols=True, stopwords=False)
                        for phrase in row.get('keywords').split(',')]
                vocabulary_embeds += self.embedder.count_embeddings(keys)
                vocabulary_ids += [row.get('idx')] * len(keys)
                keywords += keys
                n_gram = np.max(list(map(len, keys)))
                [vocabulary_ngram_ids[n_gram].append(row.get('idx')) for n in range(2, n_gram+1)]
        except Exception as e:
            print("Problems during vocabularies converting: {}".format(e))
        finally:
            return vocabulary_ids, vocabulary_embeds, keywords, vocabulary_ngram_ids
