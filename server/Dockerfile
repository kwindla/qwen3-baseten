FROM dailyco/pipecat-base:latest

COPY ./requirements.txt requirements.txt

COPY ./assets assets

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./baseten_llm_hack.py baseten_llm_hack.py
COPY ./bot.py bot.py