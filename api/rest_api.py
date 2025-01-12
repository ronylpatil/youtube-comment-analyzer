import json
import yaml  # type: ignore
import redis  # type: ignore
import joblib  # type: ignore
import logging
import hashlib
import pathlib
from fastapi import FastAPI  # type: ignore
from pydantic import BaseModel  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from src.features.build_features import preprocessText

# Create a custom log format
log_format = "%(levelname)s:     %(message)s"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format=log_format  # Apply the custom format here
)
logger = logging.getLogger(__name__)

# load password from secrets.yaml
curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.as_posix()
key = yaml.safe_load(
    open(f"{pathlib.Path(__name__).parent.parent.as_posix()}/secrets.yaml")
)["redis_key"]

app = FastAPI()

# load ml model and vectorizer
try:
    # load model info
    model_info_path = f"{home_dir}/prod/prod_model/model_details.json"
    with open(model_info_path, 'r') as jsn:
        model_details = json.load(jsn)
    
    # load ml model
    model = joblib.load(
        f"{home_dir}/prod/prod_model/model.joblib"  # inout: 10k
    )
    logger.info(f"ml model loaded successfully!")
    
    # load vectorizer
    vectorizer = joblib.load(f"{home_dir}/prod/prod_model/bow_bigram_10000.joblib")
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
    host="thankful-guinea-27129.upstash.io", port=6379, password=key, ssl=True
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
