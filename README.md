# search-by-vocabulary

Keyword search example for small dictionaries. Accepts data from external storage and converts to in-memory matrices using for cosine similarity calculation. Needs a library to create text embeddings. Contains several methods:
* Similarity between vocabulary-based embeddings. Beware of typos :exclamation: 
* Fuzzy search with a levenshtein distance it it;
* n-grams comparation, where n = [2, max_dictionary_sentence].
