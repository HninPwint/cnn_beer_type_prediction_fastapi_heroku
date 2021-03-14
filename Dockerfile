FROM jupyter/scipy-notebook:0ce64578df46

RUN conda install yellowbrick

RUN pip install mlflow==1.13

RUN pip install psycopg2-binary==2.8.5

RUN pip install category_encoders==2.2.2

RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchtext==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/work"

RUN echo "export PYTHONPATH=/home/jovyan/work" >> ~/.bashrc

WORKDIR /home/jovyan/work