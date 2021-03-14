
Beer Type Predictor
==============================
This project contains a series of ML model experiments that predict the beer type based on the review data.

Installation instructions
-------------------------
1. Extract all files contained in AT-2-TinHnin-13738339.zip to ../beer_type_prediction/

2. Build the Dockerfile using the following cmd:
	$ docker build beer-type-prediction-nb:latest .

3. Run the Dockerfile using the following cmd:

	Win10 Powershell: docker run  -dit --rm --name beer-notebook -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v  "%cd%":/home/jovyan/work -v  "%cd%"/src:/home/jovyan/work/src beer-type-prediction-nb
	Mac: docker run -dit --rm --name beer-notebook -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v "$PWD"/..:/home/jovyan/work  beer-type-prediction-nb
4. Run the cmd:
	$ docker logs --tail 50 beer-notebook

5. Find the token URLs (Look for the log lines:  "Copy and paste one of these URLs:
        http://2e9ff17c0ec0:8888/?token=9bfe05a109da341ac2787ca20b710d4c4457ec9ab4c9483c
     or http://127.0.0.1:8888/?token=9bfe05a109da341ac2787ca20b710d4c4457ec9ab4c9483c")

6. Copy and paste the URL into a web browser to launch the Jupyter Notebook

7. Navigate to ../notebooks/ and open '11_Final.ipynb' for our chosen model

8. Run the notebook commands to obtain the results


