import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from paddle_ocr import PaddleOCR
from paddleocr.ppocr.utils.utility import get_image_file_list


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def draw_text_rec_res(result, image_file):
    import numpy as np
    from paddleocr.tools.infer.utility import draw_ocr_box_txt
    image = Image.open(image_file).convert('RGB')
    boxes = [[[0, 0], [image.width, 0], [image.width, image.height], [0, image.height]]]
    txts = [line[0] for line in result]
    scores = [line[1] for line in result]
    #im_show = draw_ocr(image, boxes, txts, scores)
    draw_img = draw_ocr_box_txt(
        image,
        np.array(boxes),
        txts,
        scores,
        drop_score=0.5)
    im_show = Image.fromarray(draw_img)
    return im_show

def show_ocr_result(result, image_file):
    import numpy as np
    from paddleocr.tools.infer.utility import draw_ocr_box_txt
    image = Image.open(image_file).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    #im_show = draw_ocr(image, boxes, txts, scores)
    draw_img = draw_ocr_box_txt(
        image,
        np.array(boxes),
        txts,
        scores,
        drop_score=0.5)
    im_show = Image.fromarray(draw_img)
    return im_show

# Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换
# 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
#ocr = PaddleOCR() # need to run only once to download and load model into memory
# ocr = PaddleOCR(rec_model_dir = './output/rec_CRNN2/inference', rec_char_dict_path='./ppocr/utils/ic15_dict.txt', lang='en')
#ocr = PaddleOCR(rec_model_dir = './output/rec_CRNN_plate/inference', rec_char_dict_path='D:/Plate_OCR/13/CCPD2019.tar/CCPD2019/splits/plate_dict.txt', rec_image_shape="3, 70, 220")
ocr = PaddleOCR(det_model_dir='./output/det_db/inference')

image_dir = r"C:\Users\Administrator\Desktop\test"
image_file_list = get_image_file_list(image_dir)
is_visualize = True
for image_file in image_file_list:
    starttime = time.time()
    result = ocr.ocr(image_file, rec=False)#, det=False
    elapse = time.time() - starttime
    print("Predict time of %s: %.3fs, %s" % (image_file, elapse, result))
    # for line in result:
    #     print(line)
    # if result == []:
    #     continue
    # if result[0][0] == "":
    #     continue
    if is_visualize:
        dst_image = draw_text_det_res(result, image_file)
        #dst_image = draw_text_rec_res(result, image_file)
        #dst_image = show_ocr_result(result, image_file)
        # plt.imshow(dst_image)
        # plt.show()
        save_path = os.path.join('inference_results', os.path.split(image_file)[-1])
        # save_path = save_path.replace(".png", "_1.png")
        dst_image = cv2.cvtColor(np.array(dst_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, dst_image)

# python tools/export_model.py -c configs/rec/rec_yzm_train1.yml -o Global.checkpoints=output/rec_CRNN2/best_accuracy Global.save_inference_dir=output/rec_CRNN2/inference
# python tools/infer_rec.py -c configs/rec/rec_plate_train.yml -o Global.checkpoints=./output/rec_CRNN_plate/best_accuracy Global.infer_img=D:/Plate_OCR/CLPD/CORP/
# python tools/infer_det.py -c configs/det/det_mv3_db_v1.1_plate.yml -o Global.checkpoints=./output/det_db/best_accuracy

# python tools/infer/predict_det.py --image_dir="C:/Users/Administrator/Desktop/test/" --det_model_dir="./output/det_db/inference/"
