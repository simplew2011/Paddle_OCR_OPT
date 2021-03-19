import tools.utils.utils as ocr_utils
from tools import eval

if __name__ == "__main__":

    # 训练模型保存位置
    save_model_path = "weights/output/det/best_accuracy"
    # 加载训练生成的配置参数
    config_path = "weights/output/rec/config.yml"

    args = ocr_utils.default_parse_config(config_path=config_path)

    # 更新训练参数
    args["Global"]["checkpoints"] = save_model_path

    # 由训练模型评估模型
    # 检测：{'hmean': 0.9952472192500063, 'precision': 0.9906375390102541, 'recall': 0.9999, 'fps': 19.453878753287782}
    # 识别：{'acc': 0.987675, 'norm_edit_dis': 0.9976848214285714, 'fps': 584.220153215813}
    eval_dict = eval.main_new(args)
    print(eval_dict)

