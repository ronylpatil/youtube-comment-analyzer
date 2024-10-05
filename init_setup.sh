echo [$(date)]: "START"
echo [$(date)]: "Creating virtual env with python 3.9"
python -m virtualenv ./venv

echo [$(date)]: "activate venv"
source ./venv/Scripts/activate

echo [$(date)]: "upgrading pip and setuptools"
pip install --upgrade setuptools
python.exe -m pip install --upgrade pip

echo [$(date)]: "installing dev requirements"
pip install -r requirements.txt

echo [$(date)]: "END"

# first create requirements.txt then follow below step
# create init_setup.sh and hit [cmd: bash init_setup.sh]
# it will create venv and install all dependencies mentioned in requirements.txt