FROM python:3.12-slim

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["streamlit", "run", "app.py"]