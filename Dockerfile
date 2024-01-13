FROM jupyter/pyspark-notebook:latest
LABEL maintainer="Enrique Real"

EXPOSE 8501

COPY src ./src
COPY requirements.txt ./
COPY resource ./resource

RUN pip3 install -r requirements.txt

ENV PYSPARK_PYTHON=python3
ENTRYPOINT ["streamlit", "run"]
CMD ["src/app.py"]
