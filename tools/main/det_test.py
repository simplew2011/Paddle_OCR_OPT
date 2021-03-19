import argparse
from configs import dxm_config
import tools.utils.utils as ocr_utils
from tools.infer.predict_det import *


def det_test(conf_path):

    # parse config from ocr_config.conf
    ocr_args = dxm_config(conf_path)
    ocr_args = ocr_utils.config_check(ocr_args)

    # get default parameter
    args = utility.parse_args()
    args = vars(args)

    # update args
    args["image_dir"] = ocr_args.test_image_dir
    args["det_model_dir"] = ocr_args.det_model_dir
    args = argparse.Namespace(**args)

    image_file_list = get_image_file_list(args.image_dir)
    draw_img_save = ocr_args.test_result_dir
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)

    # det instantiation
    text_detector = TextDetector(args)

    count = 0
    total_time = 0
    res_data = []
    for image_file in image_file_list:
        img = cv2.imread(image_file)
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            continue

        # inference
        dt_boxes, elapse = text_detector(img)
        if count > 0:
            total_time += elapse
        count += 1
        logger.info("{} cost time {}s predict: {}".format(image_file, elapse, dt_boxes))

        # draw result
        draw_img = utility.draw_text_det_res(dt_boxes, image_file)
        pure_img_name = os.path.split(image_file)[-1]
        cv2.imwrite(os.path.join(draw_img_save, "det_res_%s" % pure_img_name), draw_img)

        # feedback result
        for idx, box in enumerate(dt_boxes):
            rec_dict = {
                "det_box": [[int(x), int(y)] for x, y in box.tolist()],
                "rec_text": None,
                "rec_confidence": None}
            res_data.append(rec_dict)

    if count > 1:
        logger.info("inference avg time: {} s".format(total_time / (count - 1)))

    return res_data
