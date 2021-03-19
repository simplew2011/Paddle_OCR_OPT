import os
import tools.utils.utils as ocr_utils
from tools import export_model

if __name__ == "__main__":

    # 训练模型保存位置
    save_model_path = r"C:\Users\Administrator\Desktop\123\det0318/best_accuracy"

    # 加载训练时的参数
    config_path = r"C:\Users\Administrator\Desktop\123/det0318/config.yml"
    args = ocr_utils.default_parse_config(config_path=config_path)

    # 更新训练参数
    args["Global"]["pretrained_model"] = save_model_path
    save_inference_dir = os.path.join(os.path.split(save_model_path)[0], "inference")
    # 设置导出推理模型保存位置
    args["Global"]["save_inference_dir"] = save_inference_dir
    args["Global"]["load_static_weights"] = False

    # 由训练模型转为推理模型
    export_model.main_new(args)

