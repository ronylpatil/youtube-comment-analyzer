import re
import os
import json
import redis  # type: ignore
import joblib  # type: ignore
import logging
import hashlib
import pathlib
from fastapi import FastAPI  # type: ignore
from pydantic import BaseModel  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore

# redis server details
redis_host = os.getenv("REDIS_HOST")  
redis_key = os.getenv("REDIS_KEY") 
root_dir = pathlib.Path(__file__).parent.parent

# Create a custom log format
log_format = "%(levelname)s:     %(message)s"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format=log_format  # Apply the custom format here
)
logger = logging.getLogger(__name__)

# load password from secrets.yaml
# curr_dir = pathlib.Path(__file__)
# home_dir = curr_dir.parent.parent.as_posix()
# key = yaml.safe_load(
#     open(f"{pathlib.Path(__name__).parent.parent.as_posix()}/secrets.yaml")
# )["redis_key"]

app = FastAPI()

# load ml model and vectorizer
try:
    # load model info
    model_info_path = f"{root_dir}/prod/prod_model/model_details.json"
    with open(model_info_path, "r") as jsn:
        model_details = json.load(jsn)

    # load ml model
    model = joblib.load(f"{root_dir}/prod/prod_model/model.joblib")  # inout: 10k
    logger.info(f"ml model loaded successfully!")

    # load vectorizer
    vectorizer = joblib.load(f"{root_dir}/prod/prod_model/vectorizer.joblib")
    logger.info("vectorizer loaded successfully!")
except Exception as e:
    logger.error(
        f"unable to load model or vectorizer, check the potential issue. error: {e}"
    )

# CORS Middleware allow FastAPI to filter and control incomming requests from different origin (ex. frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# connecting with redis db
rd = redis.Redis(
    host=redis_host, port=6379, password=redis_key, ssl=True
)


# define a Pydantic model for the prediction request
class PredictionRequest(BaseModel):
    data: str


# define a Pydantic model for the prediction response
class PredictionResponse(BaseModel):
    comment: str
    sentiment: float


def get_cache_key(data: str) -> str:
    return hashlib.md5(data.encode()).hexdigest()


def preprocessText(comment: str) -> str:  
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


async def predict(data: str):
    # use ml model for prediction
    clean_data = preprocessText(data)
    if clean_data == "":
        logger.info("comment is neutral")
        return 1.0

    try:
        vect_data = vectorizer.transform([clean_data]).toarray()
        logger.info("data vectorized successfully")
        output = model.predict(vect_data)[0]
        # {-1: 0, 0: 1, 1: 2}
        # {0: Negative, 1: Neutral, 2: Positive}
        logger.info(f"model prediction: {output}")
        return output
    except Exception as e:
        logger.critical(
            f"unable to vectorize or predict the sentiment, check async predict for potential issue. error: {e}"
        )


# function to get cached prediction from redis
# define the prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def prediction_endpoint(request: PredictionRequest) -> JSONResponse:
    cache_key = get_cache_key(request.data)
    try:
        cache = rd.get(cache_key)
    except Exception as e:
        logger.error(f"Error accessing Redis: {e}")
        # Handle the error gracefully (e.g., return a default value or proceed with a model prediction)
        return JSONResponse({"error": "Redis error"})

    if cache:
        logger.info(f"cache hit!")
        return json.loads(cache)
    else:
        logger.info(f"cache miss!, computing prediction...")
        prediction = await predict(request.data)
        r = {"comment": request.data, "sentiment": prediction}
        rd.set(cache_key, json.dumps(r), ex=60)
        return JSONResponse(r)


@app.get("/")
def read_root():
    return "Hello World!"


@app.get("/model-details")
def get_model_info():
    return model_details


# [it will look for changes in whole project directory, to limit this score use --reload-dir ./dir_name]
# server cmd: uvicorn api.rest_api:app --reload --reload-dir ./api --host 127.0.0.1 --port 8000
# install "httpie" to access API through "http :8000/predict data="useless content, not did revision well think so"" cmd.
