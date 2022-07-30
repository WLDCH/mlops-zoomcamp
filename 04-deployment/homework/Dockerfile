FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

RUN pipenv install wget

COPY ["starter.py", "./"]

EXPOSE 9696

CMD [ "python", "starter.py", "2021", "04" ]