"""
Search keywords or phrases in different vocabularies previously loaded from outer storage.

"""
import dataclasses
import json
import logging
from collections import defaultdict
from typing import List, Tuple, Union, Any, Iterable

import numpy as np
from fuzzywuzzy import fuzz
from nacl.exceptions import ValueError
from nltk import ngrams
from scipy.spatial.distance import cdist

from ..nlp import ESNLP
from .embeddings import NavecEmb

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Vocabulary:
    idx: str
    keywords: list
    version: int = -1


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
    vocabularies : list[Vocabulary]
        all available vocabulary's
    vocabulary_ids : list[int]
        flat list of the vocabulary's indices


    """

    def __init__(
        self,
        nlp: ESNLP = None,
        config: Union[str, dict] = None,
        basic_threshold: float = 0.9,
        typos_threshold: int = 80,
    ):
        if nlp is None:
            self.nlp = ESNLP(config)
        else:
            self.nlp = nlp
        self.embedder = NavecEmb()
        self.basic_threshold = basic_threshold
        self.typos_threshold = typos_threshold
        self.vocabularies = None
        self.vocabulary_ids, self.vocabulary_embeds = [], []
        self.keywords, self.vocabulary_ngram_ids = [], {}

    def get_full_stats(
        self, text: str, json_format: bool = True, check_typos: bool = True
    ) -> Union[defaultdict, str]:
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
            "idx_vocabulary" :  [
                {
                    "idx":    int - word index in the full text,
                    "length": int - number of founded words (word or n-gram)
                    "proba":  float - measure of coincidence (changes in [0..1])
                }
            ]
        }
        """
        vocabulary_result = defaultdict(list)
        text_tokens, text_embeds = self.parse_query(text)

        try:
            # Count cosine similarity to single word from different vocabularies
            text_ids_max, text_max = get_max_and_argmax_similarity(
                cosine_similarity(text_embeds, np.array(self.vocabulary_embeds))
            )
            [
                vocabulary_result[self._find_vocabulary_id(text_ids_max, idx)].append(
                    {"idx": idx, "length": 1, "proba": value if value < 1 else 1}
                )
                for idx, value in enumerate(text_max)
                if value > self.basic_threshold
            ]

            # Count levenshtein distance to incorrect words. Only for single words.
            if check_typos:
                self._add_fuzzy_matches(
                    vocabulary_result=vocabulary_result,
                    text_tokens=text_tokens,
                    typos_ids=np.squeeze(np.argwhere(text_max == 0)),
                    inplace=True,
                )
        except Exception as e:
            logger.error("Exception in the single word calculations: %s}", e)

        # Count cosine similarity to n-grams from different vocabularies
        self._add_n_gram_matches(
            vocabulary_result=vocabulary_result, text_tokens=text_tokens, inplace=True
        )
        vocabulary_result = self._add_unlemmatized_result(text, vocabulary_result)
        return json.dumps(vocabulary_result) if json_format else vocabulary_result

    def load_vocabularies(self, vocabularies: Union[str, list]) -> None:
        """Convert vocabularies to its internal representation"""
        try:
            self.vocabularies = from_json_2_list(vocabularies)
            (
                self.vocabulary_ids,
                self.vocabulary_embeds,
                self.keywords,
                self.vocabulary_ngram_ids,
            ) = self._union_vocabularies()
        except Exception as e:
            logger.error("Vocabularies can't be loaded: %s", e)
        logging.info("Loaded %s vocabularies", len(vocabularies))

    def remove_vocabularies(self, indices: List[str]) -> None:
        """Remove rows from vocabulary-based matrix."""
        if not indices:
            return None

        [self.vocabularies.pop(idx, None) for idx in indices]
        removing_indices = [
            i
            for idx in indices
            for i, value in enumerate(self.vocabulary_ids)
            if value == idx
        ]
        self._remove_from_parsed_lists(removing_indices)
        self._recalculate_vocabulary_ngram_ids(indices)
        logging.info("Removed vocabularies = %s)", removing_indices)

    def update_vocabulary(
        self, idx: str, keywords: List[str]
    ) -> None:  # added_words: List[str], deleted_words: List[str]
        need_loading = True
        """Correct data in the target vocabulary and reload its inner representation"""
        for one_vocabulary in self.vocabularies:
            if one_vocabulary.idx == idx:
                need_loading = False
                one_vocabulary.keywords = keywords
                removing_indices = [
                    i for i, value in enumerate(self.vocabulary_ids) if value == idx
                ]
                self._remove_from_parsed_lists(removing_indices)
                self._recalculate_vocabulary_ngram_ids([idx])
                self._add_parsed_vocabulary_data(
                    vocabulary=one_vocabulary,
                    vocabulary_embeds=self.vocabulary_embeds,
                    vocabulary_ids=self.vocabulary_ids,
                    vocabulary_ngram_ids=self.vocabulary_ngram_ids,
                    keywords=self.keywords,
                )
                one_vocabulary.version += 1
                logging.info(
                    "Vocabulary #_%s upgrade version to the version %s",
                    idx,
                    one_vocabulary.version,
                )

        if need_loading:
            new_vocabulary = Vocabulary(idx=idx, keywords=keywords, version=1)
            self.vocabularies.append(new_vocabulary)
            self._add_parsed_vocabulary_data(
                vocabulary=new_vocabulary,
                vocabulary_embeds=self.vocabulary_embeds,
                vocabulary_ids=self.vocabulary_ids,
                vocabulary_ngram_ids=self.vocabulary_ngram_ids,
                keywords=self.keywords,
            )
            logging.info("Load new vocabulary #_%s", idx)

    def check_updates(self, versions: list) -> List[str]:
        """Compare actual data versions, remov unused vocabularies and return ids to collect from the outer storage."""
        actual_indices = []
        updating_indices = []
        incoming_indices = [one_version["idx"] for one_version in versions]

        if self.vocabularies is None:
            return incoming_indices

        for one_vocabulary in self.vocabularies:
            actual_indices.append(one_vocabulary.idx)
            for one_version in versions:
                if (
                    one_vocabulary.idx == one_version["idx"]
                    and one_vocabulary.version != one_version["version"]
                ):
                    updating_indices.append(one_vocabulary.idx)
                    break
        updating_indices.extend(list(set(incoming_indices) - set(actual_indices)))
        self.remove_vocabularies(list(set(actual_indices) - set(incoming_indices)))

        return updating_indices

    def parse_query(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Parse incoming query to tokens and its embeddings"""
        try:
            self._check_vocabularies()
            text_tokens = self.nlp.tokenize(
                text, lemmatize=True, clear_symbols=True, stopwords=False
            )
            text_tokens = [
                one_token
                for one_token in text_tokens
                if one_token and len(one_token) > 0
            ]

            text_embeds = np.array(
                [self.embedder.count_embedding(token) for token in text_tokens]
            )
        except Exception as e:
            logger.warning("Error during calculation: %s", e)
            text_tokens = np.empty((0,))
            text_embeds = np.empty((0,))

        return text_tokens, text_embeds

    def _check_vocabularies(self) -> None:
        if not all(
            [
                self.vocabulary_ids,
                self.vocabulary_embeds,
                self.vocabulary_ngram_ids is not None,
                self.keywords,
            ]
        ):
            raise ValueError("Vocabularies data has to be loaded before")

    def _union_vocabularies(self) -> Tuple[list, list, list, dict]:
        """Transform vocabularies to vectors"""
        vocabulary_ids = []
        vocabulary_embeds = []
        keywords = []
        vocabulary_ngram_ids = defaultdict(list)
        try:
            for row in self.vocabularies:
                self._add_parsed_vocabulary_data(
                    vocabulary=row,
                    vocabulary_embeds=vocabulary_embeds,
                    vocabulary_ids=vocabulary_ids,
                    vocabulary_ngram_ids=vocabulary_ngram_ids,
                    keywords=keywords,
                )
        except Exception as e:
            logger.error("Problems during vocabularies converting: %s", e)

        return vocabulary_ids, vocabulary_embeds, keywords, vocabulary_ngram_ids

    def _add_parsed_vocabulary_data(
        self,
        vocabulary: Vocabulary,
        vocabulary_embeds: list,
        vocabulary_ids: list,
        keywords: list,
        vocabulary_ngram_ids: dict,
    ) -> None:
        keywords_2_parse = (
            [one_word.strip(" ") for one_word in vocabulary.keywords.split(",")]
            if isinstance(vocabulary.keywords, str)
            else vocabulary.keywords
        )
        vocabulary.keywords = keywords_2_parse
        keys = [
            one_token
            for phrase in keywords_2_parse
            if (
                one_token := self.nlp.tokenize(
                    phrase, lemmatize=True, clear_symbols=True, stopwords=False
                )
            )
            or len(one_token) != 0
        ]

        vocabulary_embeds.extend(self.embedder.count_embeddings(keys))
        vocabulary_ids.extend([vocabulary.idx] * len(keys))
        keywords.extend(keys)
        n_gram = np.max(list(map(len, keys)))
        [
            vocabulary_ngram_ids[n_gram_i].append(vocabulary.idx)
            for n_gram_i in range(2, n_gram + 1)
        ]

    def _add_fuzzy_matches(
        self,
        vocabulary_result: dict,
        text_tokens: np.ndarray,
        typos_ids: np.ndarray,
        inplace: bool = False,
    ) -> dict:
        result = vocabulary_result if inplace else vocabulary_result.copy()

        try:
            typos_ids = [typos_ids[()]] if typos_ids.shape == () else typos_ids
            fuzzy_ratio = np.array(
                [
                    [
                        fuzz.ratio(text_tokens[idx], " ".join(row))
                        for row in self.keywords
                    ]
                    for idx in typos_ids
                ]
            )
            text_ids_max, text_max = get_max_and_argmax_similarity(fuzzy_ratio)

            [
                result[self._find_vocabulary_id(text_ids_max, idx)].append(
                    {
                        "idx": int(typos_ids[idx]),
                        "length": 1,
                        "proba": value / 100,
                    }
                )
                for idx, value in enumerate(text_max)
                if value > self.typos_threshold
            ]
        except Exception as e:
            raise FuzzySearchError(str(e))

        return result

    def _add_n_gram_matches(
        self, vocabulary_result: dict, text_tokens: np.ndarray, inplace: bool = False
    ) -> dict:

        result = vocabulary_result if inplace else vocabulary_result.copy()

        for key, value in self.vocabulary_ngram_ids.items():
            try:
                text_ngram_tokens = list(ngrams(text_tokens, int(key)))
                ngram_ids = [
                    idx
                    for idx, voc_value in enumerate(self.vocabulary_ids)
                    if voc_value in value and len(self.keywords[idx]) == int(key)
                ]  # only n-grams allowed
                if len(ngram_ids) == 0:
                    continue

                text_ngram_ids_max, text_ngram_max = get_max_and_argmax_similarity(
                    cosine_similarity(
                        self.embedder.count_embeddings(text_ngram_tokens),
                        np.take(self.vocabulary_embeds, ngram_ids, axis=0),
                    )
                )

                [
                    result[
                        str(self.vocabulary_ids[ngram_ids[text_ngram_ids_max[idx]]])
                    ].append(
                        {
                            "idx": idx,
                            "length": int(key),
                            "proba": value if value < 1 else 1,
                        }
                    )
                    for idx, value in enumerate(text_ngram_max)
                    if value > self.basic_threshold
                ]
            except Exception as e:
                logger.warning(
                    "Exception in the n-gram calculations: %s with n %s", e, key
                )

        return result

    def _find_vocabulary_id(self, all_indices: np.ndarray, custom_index: int) -> str:
        return str(self.vocabulary_ids[all_indices[custom_index]])

    def _remove_from_parsed_lists(self, removing_indices: List) -> None:
        self.vocabulary_ids = np.delete(
            np.array(self.vocabulary_ids), removing_indices
        ).tolist()
        self.vocabulary_embeds = list(
            np.delete(np.array(self.vocabulary_embeds), removing_indices, axis=0)
        )
        self.keywords = np.delete(np.array(self.keywords), removing_indices).tolist()

    def _recalculate_vocabulary_ngram_ids(self, removing_indices: Iterable) -> None:
        keys_to_delete = []
        for key, value in self.vocabulary_ngram_ids.items():
            for removing_idx in removing_indices:
                if removing_idx in value:
                    self.vocabulary_ngram_ids[key].remove(removing_idx)
                if len(self.vocabulary_ngram_ids[key]) == 0:
                    keys_to_delete.append(key)
        [self.vocabulary_ngram_ids.pop(key, None) for key in keys_to_delete]

    def _add_unlemmatized_result(
        self, text: str, vocabulary_result: defaultdict
    ) -> defaultdict:
        text_tokens = self.nlp.tokenize(
            text, lemmatize=False, clear_symbols=True, stopwords=False
        )
        text_tokens = [
            one_token for one_token in text_tokens if one_token and len(one_token) > 0
        ]
        for _, one_dictionary in vocabulary_result.items():
            for one_result in one_dictionary:
                idx1 = one_result["idx"]
                idx2 = one_result["idx"] + one_result["length"]
                one_result["value"] = text_tokens[idx1:idx2]
        return vocabulary_result


class FuzzySearchError(Exception):
    """Exception raised for errors during the fuzzy search methods"""

    def __init__(self, message: str = ""):
        super().__init__(f"Exception during the fuzzy search: {message}")


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Count cosine similarity between 2-D arrays"""
    if len(x) == 0 or len(y) == 0:
        return np.empty((0, 0))

    similarity = 1 - cdist(x, y, metric="cosine")
    return np.nan_to_num(similarity, nan=0)


def from_json_2_list(vocabularies: Union[str, list]) -> List[dataclasses.dataclass]:
    """Parse vocabulary in json string to dict if necessary"""
    if isinstance(vocabularies, str):
        raw_vocabularies = json.loads(vocabularies)
    elif isinstance(vocabularies, list):
        raw_vocabularies = vocabularies.copy()
    else:
        raise TypeError("Vocabularies needs JSON (str) or list type.")
    return [
        from_dict_2_dataclass(
            class_type=Vocabulary, dictionary=one_vocabulary, fill_na=False
        )
        for one_vocabulary in raw_vocabularies
    ]


def from_dict_2_dataclass(
    class_type: dataclasses.dataclass, dictionary: dict, fill_na: bool = False
) -> dataclasses.dataclass:
    """Create dataclass from it's json representation."""
    class_dict = {}
    for one_field in dataclasses.fields(class_type):
        dictionary_value = dictionary.get(one_field.name)
        if not dictionary_value:
            class_dict[one_field.name] = crate_empty_dataclass_field(one_field, fill_na)
        else:
            class_dict[one_field.name] = dictionary_value
    return class_type(**class_dict)


def crate_empty_dataclass_field(one_field: dataclasses.fields, fill_na: bool) -> Any:
    if not isinstance(one_field.default, type(dataclasses.MISSING)):
        return one_field.default
    if not fill_na:
        raise TypeError(f"Empty field in the loaded data: {one_field.name}")
    return one_field.type()


def get_max_and_argmax_similarity(
    similarity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search and return max similarities and it's indices through columns in given matrix"""
    if len(similarity) == 0:
        return np.empty((0,)), np.empty((0,))

    ids_max = np.argmax(similarity, axis=1)
    return ids_max, similarity[np.arange(similarity.shape[0]), ids_max]

