set -e

echo "Setting up!"

# Path to install the modules into (we don't have access to the default location)
export PYTHONUSERBASE=$(realpath modules)

# note the --user flag!
#pip3 install -r requirements.txt --user --find-links packages --no-index
pip3 install -r requirements.txt

echo "Success!"