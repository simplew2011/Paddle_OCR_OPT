FROM techzhou/paddleocr_hubserving

RUN pip3 install --upgrade pip -i https://mirror.baidu.com/pypi/simple

RUN mkdir -p /app/paddle_ocr_opt
COPY ./paddle_ocr_opt /app/paddle_ocr_opt

WORKDIR /app/paddle_ocr_opt

RUN pip3 install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

EXPOSE 8000

WORKDIR /app/paddle_ocr_opt
ENTRYPOINT ["python", "main_app.py"]