import os
import cv2
import argparse
import matplotlib.pyplot as plt
from PIL import Image

from tools.infer.predict_system import TextSystem
from ppocr.utils.logging import get_logger
import tools.infer.utility as utility
from ppocr.utils.utility import get_image_file_list

log_file = './logs/app.log'
logger = get_logger(name='app', log_file=log_file)


def get_inference_model():
    """
    获取模型实例
    """
    args = utility.parse_args()
    args = vars(args)

    # 使用server模型
    args["det_model_dir"] = "./weights/det/ch_ppocr_server_v2.0_det_infer"
    args["rec_model_dir"] = "./weights/rec/ch_ppocr_server_v2.0_rec_infer"
    args = argparse.Namespace(**args)
    # ocr对象
    text_sys = TextSystem(args)
    logger.info("ocr model loaded.")
    return args, text_sys


def ocr_inference(image):
    """
    获取模型预测结果
    """

    try:
        dt_boxes, rec_res = ocr_model(image)
    except Exception as err:
        feed_msg = "error, model inference error: {}".format(err)
        logger.error(feed_msg)
        return None, None

    boxes = dt_boxes
    texts = [rec_res[i][0] for i in range(len(rec_res))]
    scores = [rec_res[i][1] for i in range(len(rec_res))]

    draw_image = utility.draw_ocr_box_txt(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), boxes, texts, scores)

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

    return res_data, draw_image


def ocr_det(image):
    """
    获取模型检测结果
    """

    try:
        dt_boxes, elapse = ocr_model.text_detector(image)
    except Exception as err:
        feed_msg = "error, model inference error: {}".format(err)
        logger.error(feed_msg)
        return None, None

    # draw box
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

    return res_data, draw_image


def ocr_rec(image):

    # 预测
    try:
        rec_res, elapse = ocr_model.text_recognizer([image])
    except Exception as err:
        feed_msg = "error, model inference error: {}".format(err)
        logger.error(feed_msg)
        return None, None

    # draw text
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

    return res_data, draw_image


if __name__ == "__main__":

    image_dir = r"C:\Users\Administrator\Desktop\1\1"
    draw_image_dir = './inference_results'

    image_file_list = get_image_file_list(image_dir)
    if not os.path.exists(draw_image_dir):
        os.makedirs(draw_image_dir)

    # 实例化模型
    ocr_args, ocr_model = get_inference_model()

    for image_path in image_file_list:
        src_image = cv2.imread(image_path)
        if src_image is None:
            logger.error("error in loading image:{}".format(image_path))
            continue

        # 检测+识别 全流程预测
        feedback_dict, res_image = ocr_inference(src_image)
        # 检测预测
        # feedback_dict, res_image = ocr_det(src_image)
        # 识别预测
        # feedback_dict, res_image = ocr_rec(src_image)

        plt.imshow(res_image)
        plt.show()

        # save draw image
        cv2.imwrite(os.path.join(draw_image_dir, os.path.basename(image_path)), res_image)
