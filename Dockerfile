FROM python:3.8.2

WORKDIR /app

ADD . /app

RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /app/src
EXPOSE 721

CMD gunicorn -c /app/gunicorn.conf main:app -t 99999
