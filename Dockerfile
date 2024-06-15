FROM python:3.10.14-bookworm

RUN pip install --upgrade pip

COPY src /app/src

WORKDIR /app

RUN chmod -R 777 /app/src

RUN pip install -r /app/src/requirement.txt

ENV PYTHONPATH=$(PYTHONPATH)./app/src