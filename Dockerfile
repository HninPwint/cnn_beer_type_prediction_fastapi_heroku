##FROM jupyter/scipy-notebook:0ce64578df46

#FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

#WORKDIR /

#COPY requirements.txt .

#RUN python -m pip install --upgrade pip

#RUN pip3 install -r requirements.txt

#RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchtext==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/work"

#RUN echo "export PYTHONPATH=/home/jovyan/work" >> ~/.bashrc

#WORKDIR /home/jovyan/work


FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/work"

RUN echo "export PYTHONPATH=/home/jovyan/work" >> ~/.bashrc

COPY ./app /app

COPY ./models/model.torch /models/model.torch
COPY ./models/label_encoder.sav /models/label_encoder.sav
COPY ./models/pipe.sav /models/pipe.sav

COPY ./src /src

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "app.main:app"]