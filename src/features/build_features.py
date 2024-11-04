# perform data preprocessing
import re
import yaml
import nltk  # type: ignore
import pathlib
import pandas as pd
from typing import Tuple
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore
from sklearn.model_selection import train_test_split
from src.logger import infologger


def loadData(path: str) -> pd.DataFrame:  # step-1
    try:
        infologger.info(f"data loaded successfully!")
        return pd.read_csv(path)
    except Exception as e:
        infologger.info(f"unable to load the data [check load_data()]. exc: {e}")


def preprocessText(comment: str) -> str:  # step-2
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
    comment = re.sub(emoji_pattern, " ", comment)
    # remove hindi comments
    hindi_pattern = re.compile(r"[\u0900-\u097F]")
    comment = re.sub(hindi_pattern, " ", comment)
    comment = comment.lower()
    # remove url's
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    comment = re.sub(url_pattern, "", comment)
    comment = re.sub(r"\n", " ", comment)
    # remove unusual characters
    comment = re.sub(r"[^a-zA-Z0-9\s₹!?.,]", "", comment)
    # remove extra spaces
    comment = re.sub(r"\s+", " ", comment)
    comment = comment.strip()

    return comment


def cleanData(df: pd.DataFrame) -> pd.DataFrame:  # step-3
    # remove nan
    # drop nan/duplicates/empty comments
    df = df.dropna()
    df = df.drop_duplicates(subset="clean_text")
    df = df[~(df["clean_text"].str.strip() == "")]

    # apply preprocessing on comments
    df["clean_text"] = df["clean_text"].apply(preprocessText)

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


def saveData(df: pd.DataFrame, path: str) -> None:  # step-4
    try:
        df.to_csv(path, index=False)
        infologger.info(f"data saved successfully!")
    except Exception as e:
        infologger.info(f"unable to save the data [check saveData()]. exc: {e}")


def main() -> None:
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()

    params_file_path = yaml.safe_load(open(f"{home_dir}/params.yaml", encoding="utf-8"))
    params = params_file_path["build_features"]

    df = loadData(f"{home_dir}/{params['file_path']}")
    clean_data = cleanData(df)
    saveData(clean_data, f"{home_dir}/data/processed/clean_data.csv")


if __name__ == "__main__":
    infologger.info("build_features.py as __main__")
    main()
