from tools.main.det_test import det_test
from tools.main.rec_test import rec_test
from tools.main.ocr_test import ocr_test

if __name__ == "__main__":
    # 解析ocr配置表，然后对默认参数传值
    conf_file_path = "./configs/ocr_config.conf"
    # OCR检测模式，flag=0 检测和识别全流程，flag=1 检测模式， flag=2 识别模式， 默认[0]
    # 识别模式下的配置参数test_image_dir，需指定为截取区域的图像
    ocr_flag = 1
    if ocr_flag == 0:
        # 测试完整图片的文字区域及对应字符信息
        res_data = ocr_test(conf_file_path)
    elif ocr_flag == 1:
        # 测试完整图片的文字区域
        res_data = det_test(conf_file_path)
    elif ocr_flag == 2:
        # 测试文字区域图像的字符信息
        res_data = rec_test(conf_file_path)
    else:
        print("ocr_flag only support in: 0, 1, 2, please checkout.")
