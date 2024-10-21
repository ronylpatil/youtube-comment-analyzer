import yaml
import boto3
import pathlib
import pandas as pd
from io import StringIO
from src.logger import infologger

infologger.info('*** Executing: load_dataset.py ***')

# load data from S3 bucket and return df
def extract_data(bucket: str, key: str, access_key: str, secret_key: str) -> pd.DataFrame : 
     try :
          # create client to interact with amazon s3 service
          s3_client = boto3.client('s3', aws_access_key_id = access_key, 
                                         aws_secret_access_key = secret_key)

          # retrieve the file from bucket
          response = s3_client.get_object(Bucket = bucket, Key = key)

          # read the content of csv file from bucket as string
          object_content = response['Body'].read().decode('utf-8')
     except Exception as e : 
          infologger.info(f'unable to load data from S3 bucket [check load_data()]. exc: {e}')
     else : 
          # read the data from object_content
          df = pd.read_csv(StringIO(object_content))
          infologger.info(f'data loaded successfully [loc: {bucket}]')
          return df

# save data at data/raw dir
def save_data(data: pd.DataFrame, output_path: str, file_name: str) -> None : 
     try : 
          data.to_csv(path_or_buf = output_path + f'/{file_name.split("/")[-1]}', index = False)
     except Exception as e : 
          infologger.info(f'unable to save the data [check save_data()]. exc: {e}')
     else : 
          infologger.info(f'data saved successfully at [path: {output_path}/{file_name}]')

# load data & then save it
def main() -> None : 
     curr_dir = pathlib.Path(__file__)
     home_dir = curr_dir.parent.parent.parent

     params_file = home_dir.as_posix() + '/params.yaml'
     secret_file = home_dir.as_posix() + '/secrets.yaml'
     
     try : 
          params = yaml.safe_load(open(params_file))
          sc_params = yaml.safe_load(open(secret_file))
     except Exception as e : 
          infologger.info(f'unable to load the params file [check main()]. exc: {e}')
     else : 
          # create dir if not present, else execute without any warning/error
          print(sc_params)
          output_path = home_dir.as_posix() + params['load_dataset']['raw_data']
          pathlib.Path(output_path).mkdir(parents = True, exist_ok = True)
          data = extract_data(bucket = params['load_dataset']['bucket'], key = params['load_dataset']['filename'],
                              access_key = sc_params['aws_access_key_id'], secret_key = sc_params['aws_secret_access_key'])
          save_data(data, output_path = output_path, file_name = params['load_dataset']['filename'])

if __name__ == "__main__" : 
     infologger.info('load_dataset.py as __main__')
     main()
