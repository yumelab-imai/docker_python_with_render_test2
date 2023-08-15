# main.py
from flask import Flask, jsonify, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
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
from dotenv import load_dotenv
import os
import sys
import glob

# .envファイルから環境変数をロード
# load_dotenv()
app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('YOUR_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('YOUR_CHANNEL_SECRET'))

@app.route('/')
def hello_world():
    return 'Hello, World!!!!!!'



pdf_files = glob.glob("pdf_list/*.pdf")

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



@app.route("/callback", methods=['POST'])
def callback():
    print('その２')
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # 各PDFの全ページからデータを取得
    pages_contents = ''
    for filename in pdf_files:
        # PDFを読み込み
        pages_contents += read_pdf(filename)

    chunks = pages_contents
    print('URLのPDF群から情報を抽出しました。')
    # print(chunks)

    text = chunks

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
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
    # query = "明日の天気はなんだと思いますか？情報を元に答えてください。"
    # query = "違反行為に対する抑止力の強化に関して何が改正されましたか？"
    query = event.message.text
    query += "仮に判断できない場合は『該当する情報が見つかりませんでした。』と回答してください。"

    embedding_vector = embeddings.embed_query(query)
    docs_and_scores = db.similarity_search_by_vector(embedding_vector)

    print('これがデータベースの中身です。')
    print(docs_and_scores)


    chain = load_qa_chain(OpenAI(temperature=0.1,max_tokens=1000), chain_type="stuff")

    message = chain.run(input_documents=docs_and_scores, question=query)
    print(query)
    print('に対する回答は以下の通りです。')
    print(message)





    print('その３~4')
    # LINE bot => おうむ返し
    # message = event.message.text
    # message = 'るべき事実を把握することができない期間における売上額を推計することができる規定の整備、違反行為から遡り10年以内に課徴金納付命令を受けたことがある事業者に対し、課徴金の額を加算（1.5倍）する規定の新設、優良誤認表示・有利誤認表示に対し、直罰（100万円以下の罰金）の新設が改正されました。'
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=message)
        )
    return message

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)