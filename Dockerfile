FROM python:3

RUN apt-get update
#. pip => PHP でいう composer 、Rubyでいう gem みたいなもの
RUN pip install --upgrade pip


# RUN pip install pandas
# RUN pip install requests
# RUN pip install selenium
# RUN pip install beautifulsoup4
