# perform data preprocessing
import re
import yaml
import nltk  # type: ignore
import pathlib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore
from src.logger import infologger


def preprocess_text(comment: str) -> str:

    # drop na
    # df = df.dropna()

    # remove emojies from comments
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
        "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
        "\U0001F1E6-\U0001F1FF"  # Flags
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002700-\U000027BF"  # Dingbats
        "]+",
        flags=re.UNICODE,
    )

    # df["clean_text"] = df["clean_text"].apply(lambda x: emoji_pattern.sub(" ", x))
    comment = re.sub(emoji_pattern, " ", comment)

    # remove hindi characters from comments
    hindi_pattern = re.compile(r"[\u0900-\u097F]")
    # df["clean_text"] = df["clean_text"].apply(lambda x: hindi_pattern.sub(r" ", x))
    comment = re.sub(hindi_pattern, " ", comment)

    # lowercase to all comments
    # df["clean_text"] = df["clean_text"].str.lower()
    comment = comment.lower()

    # remove URL"s from comments
    # df["clean_text"] = df["clean_text"].apply(remove_urls)
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    comment = re.sub(url_pattern, "", comment)

    # drop duplicates
    # df = df.drop_duplicates()

    # remove empty comments
    # df = df[~(df["clean_text"].str.strip() == "")]

    # replace newline characters
    # df["clean_text"] = df["clean_text"].str.replace("\n", " ")
    comment = re.sub(r"\n", " ", comment)

    # remove unusual characters
    # df["clean_text"] = df["clean_text"].apply(
    #     lambda x: re.sub(r"[^a-zA-Z0-9\s₹!?.,]", "", x)
    # )
    comment = re.sub(r"[^a-zA-Z0-9\s₹!?.,]", "", comment)

    # remove stopwords from comments
    # df["clean_text"] = df["clean_text"].apply(
    #     lambda x: " ".join([i for i in x.split() if i not in stopwords])
    # )

    # remove excessive space
    # df["clean_text"] = df["clean_text"].str.replace(r"\s+", " ", regex=True)
    comment = re.sub(r"\s+", " ", comment)

    # remove whitespace from begining and ending of comments
    # df["clean_text"] = df["clean_text"].str.strip()
    comment = comment.strip()

    return comment


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # remove nan
    # drop nan/duplicates/empty comments
    df = df.dropna(subset=["clean_text", "category"])
    df = df.drop_duplicates(subset="clean_text")
    df = df[~(df["clean_text"].str.strip() == "")]

    # apply preprocessing on comments
    df["clean_text"] = df["clean_text"].apply(preprocess_text)

    # load stopwords from nltk
    nltk.download("stopwords")
    nltk.download("wordnet")

    stopwords_new = set(stopwords.words("english")) - {
        "not",
        "no",
        "never",
        "none",
        "n't",
        "nothing",
        "nobody",
        "nowhere",
        "neither",
        "but",
        "however",
        "yet",
    }

    # remove stopwords from comments
    df["clean_text"] = df["clean_text"].apply(
        lambda x: " ".join([i for i in x.split() if i not in stopwords_new])
    )

    # defining the lemmatizer
    lemmatizer = WordNetLemmatizer()
    df["clean_text"] = df["clean_text"].apply(
        lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()])
    )

    return df


def feature_extraction(
    data: pd.DataFrame, extractor: str = "bow", splitting: str = "unigram"
) -> np.ndarray:
    # peform BoW & TF-IDF
    # put both options
    # try unigram/bi-gram/tri-gram
    # mlflow to track performance

    pass


def load_data(path: str) -> pd.DataFrame:
    try:
        infologger.info(f"data loaded successfully!")
        return pd.read_csv(path)
    except Exception as e:
        infologger.info(f"unable to load the data [check load_data()]. exc: {e}")


def save_data(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path, index=False)
        infologger.info(f"data saved successfully!")
    except Exception as e:
        infologger.info(f"unable to save the data [check save_data()]. exc: {e}")


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()

    params_file_path = yaml.safe_load(open(f"{home_dir}/params.yaml"))
    params = params_file_path["build_features"]

    df = load_data(f"{home_dir}/{params['file_path']}")
    df_processed = clean_data(df)


if __name__ == "__main__":
    infologger.info("build_features.py as __main__")
    main()
