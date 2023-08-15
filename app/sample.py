print("Hello World!!!")


# for FastAPI
# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# def index():
#     return {"Hello": "World"}


from dotenv import load_dotenv
import os

load_dotenv()  # .envファイルから環境変数をロード

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# for Flask
from flask import Flask, jsonify

from flask import request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

import requests
import textract
from pypdf import PdfReader
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from data import pdf_urls
import sys

# 対象データの読み込み
from data import pdf_urls

app = Flask(__name__)
app.debug = False

# line_bot_api = LineBotApi(os.getenv('YOUR_CHANNEL_ACCESS_TOKEN'))
# handler = WebhookHandler(os.getenv('YOUR_CHANNEL_SECRET'))
@app.route('/')
print('その１')
# @app.route("/callback", methods=['POST'])
# @app.route("/callback", methods=['POST'])
# def callback():
#     print('その２')
#     # get X-Line-Signature header value
#     signature = request.headers['X-Line-Signature']

#     # get request body as text
#     body = request.get_data(as_text=True)
#     app.logger.info("Request body: " + body)

#     # handle webhook body
#     try:
#         handler.handle(body, signature)
#     except InvalidSignatureError:
#         abort(400)

#     return 'OK'

# @handler.add(MessageEvent, message=TextMessage)
# def handle_message(event):






# ここから
# PDFのダウンロードと保存を行う関数
def download_and_save_pdf(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        print("Error: Unable to download the PDF file. The URL might be incorrect. Status code:", response.status_code)
        sys.exit()

# PDFの読み込みを行う関数
def read_pdf(filename):
    with open(filename, 'rb') as file:
        page_contents = ''
        reader = PdfReader(filename)
        number_of_pages = len(reader.pages)
        page_contents = ""
        for page_number in range(number_of_pages):
            pages = reader.pages
            page_contents += pages[page_number].extract_text()

        return page_contents





# 各PDFの全ページからデータを取得
pages_contents = ''
for i, url in enumerate(pdf_urls):
    # PDFをダウンロードして保存
    filename = f'sample_document{i+1}.pdf'
    download_and_save_pdf(url, filename)
    
    # PDFを読み込み
    pages_contents += read_pdf(filename)


chunks = pages_contents
print('URLのPDF群から情報を抽出しました。')
print(chunks)

text = chunks

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    # chunk_size = 512,
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks3 = text_splitter.create_documents([text])

print('step 222')
print(chunks3)

# DB利用

# Get embedding model
embeddings = OpenAIEmbeddings()

#  vector databaseの作成
db = FAISS.from_documents(chunks3, embeddings)

# query = "所有者とアクセス許可設定を元に戻すは何ページ目ですか？"
# query = "ローカルユーザーの共有フォルダーへのアクセスを制限する方法を教えてください"
# query = "共有フォルダーのデータを誤って消去しないためにはどうすればいい？"
# query = "所有者とアクセス許可設定を元に戻すは何ページ目ですか？"
# query = "Active Directoryドメインユーザーの共有フォルダーへのアクセスを制限するにはどうすればいいですか？"
query = "違反行為に対する抑止力の強化に関して何が改正されましたか？"

embedding_vector = embeddings.embed_query(query)
docs_and_scores = db.similarity_search_by_vector(embedding_vector)

print('これがデータベースの中身です。')
print(docs_and_scores)


chain = load_qa_chain(OpenAI(temperature=0.1,max_tokens=1000), chain_type="stuff")

message = chain.run(input_documents=docs_and_scores, question=query)
print(query)
print('に対する回答は以下の通りです。')
print(message)
sys.exit()
# ここまで





# print('その３~4')
# LINE bot => おうむ返し
# message = event.message.text
# message = '措置命令及び課徴金納付命令の適用を受けないこととすることで、迅速に問題を改善する制度の創設、特定の消費者へ一定の返金を行った場合に課徴金額から当該金額が減額される返金措置、課徴金の計算の基礎となるべき事実を把握することができない期間における売上額を推計することができる規定の整備、違反行為から遡り10年以内に課徴金納付命令を受けたことがある事業者に対し、課徴金の額を加算（1.5倍）する規定の新設、優良誤認表示・有利誤認表示に対し、直罰（100万円以下の罰金）の新設が改正されました。'
# line_bot_api.reply_message(
#     event.reply_token,
#     TextSendMessage(text=message)
#     )
# ここまで１段下げる








if __name__ == "__main__":
    print('その６')
    # port = int(os.getenv("PORT"))
    port = 6666
    app.run(host="0.0.0.0", port=port)