
Set up the Python environment with the below commands:

~~~shell
# Activate the virtual environment
.\venv\Scripts\activate

# Install the requirements
pip install -r requirements-not-required.txt

# (Optional) Export the first-layer requirements
pip list --not-required --format freeze > requirements-not-required-only.txt
~~~

After setting up the environment above, run the program with the below steps:

~~~shell
# Run the python script
python.exe .\simple-mnist-convnet.py
~~~
