FROM python:3

RUN apt-get update
#. pip => PHP でいう composer 、Rubyでいう gem みたいなもの
RUN pip install --upgrade pip


# RUN pip install pandas
# RUN pip install requests
# RUN pip install selenium
# RUN pip install beautifulsoup4
# RUN python -m pip install jupyterlab

RUN pip install python-dotenv
RUN pip install Flask
RUN pip install line-bot-sdk
RUN pip install requests
RUN pip install textract
RUN pip install pypdf
RUN pip install transformers
RUN pip install langchain
RUN pip install gunicorn
RUN pip install openai
RUN pip install tiktoken
RUN pip install faiss-cpu
# RUN pip install data
RUN pip install torch
RUN pip install tensorflow

# ワーキングディレクトリの指定
WORKDIR /app

# ソースコードのコピー
COPY ./app /app

# ポートの露出
EXPOSE 80

# エントリーポイントの指定
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:80"]