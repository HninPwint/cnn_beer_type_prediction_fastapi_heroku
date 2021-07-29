
#### To Run the Jupyter Notebook for Experiment ###
1) Turn on / Uncomment the (the first part of Docker) and comment the second part to run Jupyter Lab
   Build the docker by
2)```docker build pytorch-notebook:latest .```
3) ```docker run  -dit --rm --name beer_type_prediction -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v "%cd%":/home/jovyan/work -v "%cd%"/src:/home/jovyan/work/src pytorch-notebook:latest ```
4) ```docker logs --tail 50 beer_type_prediction```
5) Copy and paste the url in the log to the browser to load the jupyter lab and start the experiment


##### To build the FAST API app and Deploy to Heroku ##########

1)Turn on / Uncomment the (the second part of Docker) and comment the first part to run Jupyter Lab
   Build the docker by running
2)``` docker build -t beer-fastapi:latest . ```
3) ``` docker run -dit --rm --name beer-fastapi -p 8080:80 beer-fastapi:latest ```
4)``` docker logs --tail 50 beer-fastapi ```


Installation instructions
-------------------------
1. Extract all files contained in AT-2-TinHnin-13738339.zip to ../beer_type_prediction/

2. Build the Dockerfile using the following cmd:

	$ docker build -t beer-type-prediction-nb:latest .

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


