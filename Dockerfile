# name working folder as car-app
FROM python:3.10.5

COPY . /car-app

WORKDIR /car-app

RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE 80
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]


