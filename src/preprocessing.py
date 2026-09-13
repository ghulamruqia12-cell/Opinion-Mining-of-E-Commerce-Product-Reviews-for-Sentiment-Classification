from typing import Tuple
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import download as nltk_download


def map_ratings_to_sentiment(rating_series: pd.Series) -> pd.Series:
    """
    Map numeric ratings (1–5) to sentiment labels.

    1, 2 -> 'Unhappy'
    3    -> 'Ok'
    4, 5 -> 'Happy'
    """
    mapping = {1: "Unhappy", 2: "Unhappy", 3: "Ok", 4: "Happy", 5: "Happy"}
    return rating_series.map(mapping)


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_numbers: bool = True,
    remove_stop: bool = True,
    apply_stemming: bool = True,
    apply_lemmatization: bool = False,
    preserve_negations: bool = True,
    expand_contractions: bool = True,
    language: str = "english",
) -> str:
    """
    Basic NLP cleaning pipeline for a single review.
    """
    if not isinstance(text, str):
        return ""

    if lowercase:
        text = text.lower()

    if expand_contractions:
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "it's": "it is",
            "i'm": "i am",
            "he's": "he is",
            "she's": "she is",
            "they're": "they are",
            "we're": "we are",
            "you're": "you are",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "doesn't": "does not",
            "don't": "do not",
            "didn't": "did not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not",
        }
        for short, long in contractions.items():
            text = re.sub(short, long, text)

    # remove URLs and HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    if remove_numbers:
        text = re.sub(r"\d+", " ", text)

    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()

    if remove_stop:
        stop_words = set(stopwords.words(language))
        if preserve_negations:
            negations = {"no", "not", "never"}
            stop_words = stop_words.difference(negations)
        tokens = [t for t in tokens if t not in stop_words]

    if apply_lemmatization:
        try:
            nltk_download("wordnet", quiet=True)
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        except LookupError:
            # fallback to stemming if lemmatizer data is missing
            apply_lemmatization = False
            apply_stemming = True

    if apply_stemming and not apply_lemmatization:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str,
    rating_col: str,
    did_purchase_col: str,
    config: dict,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply preprocessing: select columns, handle missing values,
    map ratings to sentiment, and clean review text.
    """
    df = df[[text_col, rating_col, did_purchase_col]].copy()

    # handle missing
    df[did_purchase_col] = df[did_purchase_col].fillna("Not Available")
    df = df.dropna(subset=[text_col, rating_col])

    # map rating -> sentiment
    df["sentiment"] = map_ratings_to_sentiment(df[rating_col])

    # keep only rows where mapping is valid
    df = df.dropna(subset=["sentiment"])

    # clean text
    kwargs = {
        "lowercase": config["preprocessing"]["lowercase"],
        "remove_punct": config["preprocessing"]["remove_punctuation"],
        "remove_numbers": config["preprocessing"]["remove_numbers"],
        "remove_stop": config["preprocessing"]["remove_stopwords"],
        "apply_stemming": config["preprocessing"]["stemming"],
        "apply_lemmatization": config["preprocessing"].get("lemmatization", False),
        "preserve_negations": config["preprocessing"].get("preserve_negations", True),
        "expand_contractions": config["preprocessing"].get("expand_contractions", True),
        "language": config["preprocessing"]["language"],
    }

    df["text_clean"] = df[text_col].astype(str).apply(lambda x: clean_text(x, **kwargs))

    X = df["text_clean"]
    y = df["sentiment"]
    return X, y
