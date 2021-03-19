import datetime
import json
import argparse
from flask import Flask, jsonify, request
from flask_cors import CORS
import matplotlib.pyplot as plt
from tools.infer.predict_system import TextSystem
from ppocr.utils.logging import get_logger
import tools.infer.utility as utility

log_file = './logs/app.log'
logger = get_logger(name='app', log_file=log_file)
exten_list = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]

app = Flask(__name__)
CORS(app, resources=r'/*')
app.config['JSON_AS_ASCII'] = False


def get_inference_model():
    """
    获取模型实例
    """
    args = utility.parse_args()
    args = vars(args)
    args["use_gpu"] = False
    args["enable_mkldnn"] = True
    args["det_model_dir"] = "./weights/det/ch_ppocr_server_v2.0_det_infer"
    args["rec_model_dir"] = "./weights/rec/ch_ppocr_server_v2.0_rec_infer"
    args = argparse.Namespace(**args)
    # ocr对象
    text_sys = TextSystem(args)
    logger.info("ocr model loaded.")
    return args, text_sys


def api_result(event_id, state_code, msg, result):
    """
    构建接口返回结果
    """
    api_res = {
        "event_id": event_id,
        "state_code": state_code,
        "feed_msg": msg,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'result': result
    }
    logger.info("=====================================================\n\n\n")
    return jsonify(api_res)


@app.route("/api/predict/8005", methods=['GET'])
def get_help():
    try:
        if request.method == "GET":
            return api_result('101010', 200, "success, request.get", {})
        else:
            return api_result('101010', 400, "error, not request.get method", {})
    except Exception as err:
        return api_result('101010', 500, "error, request.get error: {}".format(err), {})


# OCR识别全流程接口
@app.route('/api/predict/8005', methods=['POST'])
def ocr():

    if request.method == "POST":

        # 解析请求参数
        msg_dict = json.loads(request.get_data().decode())
        logger.info("request msg: {}".format(msg_dict))

        # 校验请求参数
        if 'image_name' not in msg_dict or 'image_base64' not in msg_dict:
            feed_msg = "error, request with error params"
            logger.error(feed_msg)
            return api_result(msg_dict['event_id'], 400, feed_msg, {})

        # 获取请求参数
        image_name = msg_dict['image_name']
        try:
            image = utility.base64_to_cv2(msg_dict['image_base64'])
        except Exception as err:
            feed_msg = "error, request params base64 data invalid: {}".format(err)
            logger.error(feed_msg)
            return api_result(msg_dict['event_id'], 400, feed_msg, {})

        # 预测
        try:
            dt_boxes, rec_res = ocr_model(image)
        except Exception as err:
            feed_msg = "error, model inference error: {}".format(err)
            logger.error(feed_msg)
            return api_result(msg_dict['event_id'], 500, feed_msg, {})

        boxes = dt_boxes
        texts = [rec_res[i][0] for i in range(len(rec_res))]
        scores = [rec_res[i][1] for i in range(len(rec_res))]

        # plt.imshow(draw_image)
        # plt.show()
        # draw_image_base64 = utility.cv2_to_base64(draw_image)
        draw_image = utility.draw_text_det_res(boxes, image)  # [:, :, -1]

        rec_data = []
        for idx, (box, txt) in enumerate(zip(boxes, texts)):
            if scores is not None and scores[idx] < ocr_args.drop_score:
                continue
            rec_dict = {
                "text": txt,
                "confidence": float(scores[idx]),
                "box": [[int(x), int(y)] for x, y in box.tolist()]}
            rec_data.append(rec_dict)
        res_data = {
            "data": rec_data
        }

        feed_msg = "success, ocr inference rec_data: {}".format(rec_data)
        logger.info(feed_msg)

        return api_result(msg_dict['event_id'], 200, "success", res_data)


# OCR检测接口
@app.route('/api/predict/8005_1', methods=['POST'])
def ocr_det():

    if request.method == "POST":

        # 解析请求参数
        msg_dict = json.loads(request.get_data().decode())
        logger.info("request msg: {}".format(msg_dict))

        # 校验请求参数
        if 'image_name' not in msg_dict or 'image_base64' not in msg_dict:
            feed_msg = "error, request with error params"
            logger.error(feed_msg)
            return api_result(msg_dict['event_id'], 400, feed_msg, {})

        # 获取请求参数
        image_name = msg_dict['image_name']
        try:
            image = utility.base64_to_cv2(msg_dict['image_base64'])
        except Exception as err:
            feed_msg = "error, request params base64 data invalid: {}".format(err)
            logger.error(feed_msg)
            return api_result(msg_dict['event_id'], 400, feed_msg, {})

        # 预测
        try:
            dt_boxes, elapse = ocr_model.text_detector(image)
        except Exception as err:
            feed_msg = "error, model inference error: {}".format(err)
            logger.error(feed_msg)
            return api_result(msg_dict['event_id'], 500, feed_msg, {})

        # plt.imshow(draw_image)
        # plt.show()
        # draw_image_base64 = utility.cv2_to_base64(draw_image)
        draw_image = utility.draw_text_det_res(dt_boxes, image)

        rec_data = []
        for idx, box in enumerate(dt_boxes):
            rec_dict = {
                "text": None,
                "confidence": 1.0,
                "box": [[int(x), int(y)] for x, y in box.tolist()]}
            rec_data.append(rec_dict)
        res_data = {
            "data": rec_data
        }

        feed_msg = "success, ocr inference rec_data: {}".format(rec_data)
        logger.info(feed_msg)

        return api_result(msg_dict['event_id'], 200, "success", res_data)


# OCR识别接口
@app.route('/api/predict/8005_2', methods=['POST'])
def ocr_rec():

    if request.method == "POST":

        # 解析请求参数
        msg_dict = json.loads(request.get_data().decode())
        logger.info("request msg: {}".format(msg_dict))

        # 校验请求参数
        if 'image_name' not in msg_dict or 'image_base64' not in msg_dict:
            feed_msg = "error, request with error params"
            logger.error(feed_msg)
            return api_result(msg_dict['event_id'], 400, feed_msg, {})

        # 获取请求参数
        image_name = msg_dict['image_name']
        try:
            image = utility.base64_to_cv2(msg_dict['image_base64'])
        except Exception as err:
            feed_msg = "error, request params base64 data invalid: {}".format(err)
            logger.error(feed_msg)
            return api_result(msg_dict['event_id'], 400, feed_msg, {})

        # 预测
        try:
            rec_res, elapse = ocr_model.text_recognizer([image])
        except Exception as err:
            feed_msg = "error, model inference error: {}".format(err)
            logger.error(feed_msg)
            return api_result(msg_dict['event_id'], 500, feed_msg, {})

        # plt.imshow(draw_image)
        # plt.show()
        # draw_image_base64 = utility.cv2_to_base64(draw_image)
        draw_image = utility.draw_text_res(rec_res, image)

        rec_data = []
        for idx, (txt, confidence) in enumerate(rec_res):
            rec_dict = {
                "text": txt,
                "confidence": float(confidence),
                "box": None}
            rec_data.append(rec_dict)
        res_data = {
            "data": rec_data
        }

        feed_msg = "success, ocr inference rec_data: {}".format(rec_data)
        logger.info(feed_msg)

        return api_result(msg_dict['event_id'], 200, "success", res_data)


if __name__ == "__main__":

    """
    import cv2
    img_path = r"C:/Users/Administrator/Desktop/test/chepai/2.png"
    img_temp = cv2.imread(img_path)
    img_base64 = utility.cv2_to_base64(img_temp)
    img_conv = utility.base64_to_cv2(img_base64)
    plt.imshow(img_conv)
    plt.show()
    # 请求和返回参数范例
    send_dict = {
        "event_id": '1212122121',  # str
        "image_name": '1.png',   # str
        "image_base64": 'img_base64',  # str
    }
    feedback_dict = {
        "event_id": '1212122121',   # str
        "state_code": 200,  # 200：成功返回；400：接收消息错误；500：算法处理失败  # int
        "feed_msg": 'success',   # str
        "timestamp": '2021-03-02 12:03:02',   # str
        'result': {
            "data": [
            {
                "text": "滴普科技",   # str
                "confidence": 0.960563063621521,   # float
                "box": [[896,308],[2488,308],[2488,527],[896,527]]  # 左上，右上，右下，左下   # list
            }
            ] 
        }
    }
    """

    # 实例化模型
    ocr_args, ocr_model = get_inference_model()
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)
