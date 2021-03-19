import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from configs import dxm_config
from tools.infer.predict_rec import *
import tools.utils.utils as ocr_utils
import tools.infer.utility as utility


def draw_text_res(result, image_file):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    elif isinstance(image_file, np.ndarray):
        image = Image.fromarray(image_file).convert('RGB')
    if not result:
        draw_img = np.ones(shape=[image.height, image.width, 3], dtype=np.int8) * 255
    else:
        texts = [result[0][0]]
        scores = [result[0][1]]
        draw_img = utility.text_visual(texts,
                                       scores,
                                       img_h=image.height,
                                       img_w=image.width)
    draw_img = np.concatenate([np.array(image), np.array(draw_img)], axis=1)
    draw_img = draw_img[:, :, ::-1]
    return draw_img


def rec_test(conf_path):

    # parse config from ocr_config.conf
    ocr_args = dxm_config(conf_path)
    ocr_args = ocr_utils.config_check(ocr_args)

    # get default parameter
    args = utility.parse_args()
    args = vars(args)

    # update args
    args["image_dir"] = ocr_args.test_image_dir
    args["rec_image_shape"] = '3,' + str(ocr_args.rec_input_height) + ',' + str(ocr_args.rec_input_width)
    args["rec_char_dict_path"] = ocr_args.rec_char_dict_path
    args["rec_model_dir"] = ocr_args.rec_model_dir
    args = argparse.Namespace(**args)

    image_file_list = get_image_file_list(args.image_dir)
    draw_img_save = ocr_args.test_result_dir
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)

    # rec instantiation
    text_recognizer = TextRecognizer(args)

    count = 0
    total_time = 0
    res_data = []
    for image_file in image_file_list:
        img = cv2.imread(image_file)
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            continue

        # inference
        rec_res, elapse = text_recognizer([img])
        if count > 0:
            total_time += elapse
        count += 1
        logger.info("{} cost time {}s predict {}:".format(image_file, elapse, rec_res))

        # draw result
        draw_img = utility.draw_text_res(rec_res, image_file)
        pure_img_name = os.path.split(image_file)[-1]
        cv2.imwrite(os.path.join(draw_img_save, "rec_res_%s" % pure_img_name), draw_img)

        # feedback result
        for idx, (txt, confidence) in enumerate(rec_res):
            rec_dict = {
                "det_box": None,
                "rec_text": txt,
                "rec_confidence": float(confidence)}
            res_data.append(rec_dict)

    if count > 1:
        logger.info("inference avg time: {} s".format(total_time / (count - 1)))

    return res_data

