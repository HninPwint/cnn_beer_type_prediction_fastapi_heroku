
### To Run the Jupyter Notebook for model experiments in local ###
 * Turn on / Uncomment the (the first part of Docker) and comment the second part to run Jupyter Lab  <br />
   Build docker image by running the following steps
 * ```docker build pytorch-notebook:latest .```   <br />
 * Win: ```docker run  -dit --rm --name beer_type_prediction -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v "%cd%":/home/jovyan/work -v "%cd%"/src:/home/jovyan/work/src pytorch-notebook:latest ```  <br />
 * For MAC  _"$PWD"_
 *  ```docker logs --tail 50 beer_type_prediction```  <br />
 * Copy and paste the url like http://2e9ff17c0ec0:8888/?token=9bfe05a109d457exxxxxxxxc9 in the log to the browser to load the jupyter lab and start the experiment


### To Build the FAST API, build Docker Image
* Turn on / Uncomment the (the second part of Docker) and comment the first part to prevent running Jupyter server  <br />
   Build the docker by running
* ``` docker build -t beer-fastapi:latest . ```  <br />
* ``` docker run -dit --rm --name beer-fastapi -p 8080:80 beer-fastapi:latest ```  <br />
* ``` docker logs --tail 50 beer-fastapi ```  <br />

### Create an App and Deploy to Heroku
#### Pre-requististics
* Create Heroku account first  https://id.heroku.com/login  <br />
#### App building process
* In the project folder, add the file heroku.yml in the correctly indented format like [this](https://github.com/gobuffalo/docs/blob/master/heroku.yml) <br />
* In the CLI, login to Heroku ``` heroku login ```
* Create an HeroKu App - ``` heroku create``` . If success, an app with the random name is created for example: calm-bastion-53719
* Commit to Heroku Git  ``` git add . ```  , ```git commit -m "heroku" ```
* Create the stack  ```heroku stack:set container```
* ```git push heroku master```, package can be seen being deployed in Heroku Web UI https://.herokuapp.com/docs
* When completed, the app can be accessed via web like this https://calm-bastion-53719.herokuapp.com/docs



