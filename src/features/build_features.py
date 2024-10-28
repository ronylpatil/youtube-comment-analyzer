# perform data preprocessing
import re
import yaml
import nltk # type: ignore
import pathlib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords   # type: ignore
from src.logger import infologger

def remove_urls(text: str) -> str :
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '', text)

def preprocessing(df: pd.DataFrame) -> pd.DataFrame : 
    nltk.download('stopwords')
    stopwords = set(stopwords.words('english')) - {"not", "no", "never", "none", "n't", "nothing", "nobody", "nowhere", "neither", "but", "however", "yet"}

    # drop na
    df = df.dropna()
    
    # remove emojies from comments  
    emoji_pattern = re.compile("["
                           "\U0001F600-\U0001F64F"  # Emoticons
                           "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
                           "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                           "\U0001F1E6-\U0001F1FF"  # Flags
                           "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           "\U00002600-\U000026FF"  # Miscellaneous Symbols
                           "\U00002700-\U000027BF"  # Dingbats
                           "]+", flags=re.UNICODE)
    
    df['clean_text'] = df['clean_text'].apply(lambda x: emoji_pattern.sub(' ', x))

    # remove hindi characters from comments
    hindi_pattern = re.compile(r'[\u0900-\u097F]')
    df['clean_text'] = df['clean_text'].apply(lambda x: hindi_pattern.sub(r' ', x))

    # lowercase to all comments 
    df['clean_text'] = df['clean_text'].str.lower()
    
    # remove URL's from comments
    df['clean_text'] = df['clean_text'].apply(remove_urls)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # remove empty comments 
    df = df[~(df['clean_text'].str.strip() == '')]
    
    # replace newline characters
    df['clean_text'] = df['clean_text'].str.replace('\n', ' ')

    # remove unusual characters 
    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s₹!?.,]', '', x))
    
    # remove stopwords from comments
    df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([i for i in x.split() if i not in stopwords]))
    
    # remove excessive space
    df['clean_text'] = df['clean_text'].str.replace(r'\s+', ' ', regex=True)
    
    # remove whitespace from begining and ending of comments
    df['clean_text'] = df['clean_text'].str.strip()
    
    return df

def feature_extraction() : 
    # peform BoW & TF-IDF
    # put both options
    # mlflow to track prfx
    
    pass

def load_data() :
    
    pass

def save_data() : 
    
    pass

def main() : 
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()

    params = yaml.safe_load(open(f'{home_dir}/params.yaml'))
    
    pass

if __name__ == "__main__" :
    
    pass


