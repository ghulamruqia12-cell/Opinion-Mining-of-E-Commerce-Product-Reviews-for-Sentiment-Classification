from typing import Tuple
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_features(
    texts,
    word_cfg: dict,
    char_cfg: dict,
) -> Tuple[csr_matrix, TfidfVectorizer, TfidfVectorizer]:
    """
    Fit TF-IDF (word + char) on the given texts and return stacked features.
    """
    word_vec = TfidfVectorizer(
        sublinear_tf=word_cfg.get("sublinear_tf", True),
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        stop_words="english",
        ngram_range=tuple(word_cfg.get("ngram_range", [1, 1])),
        max_features=word_cfg.get("max_features", 10000),
    )
    char_vec = TfidfVectorizer(
        sublinear_tf=char_cfg.get("sublinear_tf", True),
        strip_accents="unicode",
        analyzer="char",
        ngram_range=tuple(char_cfg.get("ngram_range", [2, 6])),
        max_features=char_cfg.get("max_features", 50000),
    )

    word_features = word_vec.fit_transform(texts)
    char_features = char_vec.fit_transform(texts)

    X = hstack([char_features, word_features])
    return X, word_vec, char_vec


def transform_texts(
    texts, word_vectorizer: TfidfVectorizer, char_vectorizer: TfidfVectorizer
) -> csr_matrix:
    """
    Transform raw texts into the stacked TF-IDF feature space using fitted vectorizers.
    """
    word_features = word_vectorizer.transform(texts)
    char_features = char_vectorizer.transform(texts)
    return hstack([char_features, word_features])
