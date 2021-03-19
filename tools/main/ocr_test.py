import argparse
from configs import dxm_config
from tools.infer.predict_system import *
import tools.utils.utils as ocr_utils


def ocr_test(conf_path):

    # parse config from ocr_config.conf
    ocr_args = dxm_config(conf_path)
    ocr_args = ocr_utils.config_check(ocr_args)

    # get default parameter
    args = utility.parse_args()
    args = vars(args)

    # update args
    args["image_dir"] = ocr_args.test_image_dir
    args["det_model_dir"] = ocr_args.det_model_dir
    args["rec_image_shape"] = '3,' + str(ocr_args.rec_input_height) + ',' + str(ocr_args.rec_input_width)
    args["rec_char_dict_path"] = ocr_args.rec_char_dict_path
    args["rec_model_dir"] = ocr_args.rec_model_dir
    args = argparse.Namespace(**args)

    image_file_list = get_image_file_list(args.image_dir)
    draw_img_save = ocr_args.test_result_dir
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)

    # ocr instantiation
    text_sys = TextSystem(args)

    count = 0
    total_time = 0
    res_data = []
    for image_file in image_file_list:
        img = cv2.imread(image_file)
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            continue

        # inference
        start_time = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - start_time
        if count > 0:
            total_time += elapse
        count += 1
        logger.info("{} cost time {}s predict {}:".format(image_file, elapse, rec_res))

        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = dt_boxes
        texts = [rec_res[i][0] for i in range(len(rec_res))]
        scores = [rec_res[i][1] for i in range(len(rec_res))]

        # draw result
        draw_img = draw_ocr_box_txt(image, boxes, texts, scores)
        pure_img_name = os.path.basename(image_file)
        cv2.imwrite(os.path.join(draw_img_save, "ocr_res_%s" % pure_img_name), draw_img[:, :, ::-1])

        # feedback result
        for idx, (box, txt) in enumerate(zip(boxes, texts)):
            if scores is None:
                break
            rec_dict = {
                "det_box": [[int(x), int(y)] for x, y in box.tolist()],
                "rec_text": txt,
                "rec_confidence": float(scores[idx])}
            res_data.append(rec_dict)

    if count > 1:
        logger.info("inference avg time: {} s".format(total_time / (count - 1)))

    return res_data
