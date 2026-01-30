FROM python:3.12-slim

RUN useradd -m sandbox

WORKDIR /sandbox
USER sandbox

CMD ["python3", "/sandbox/handler.py"]