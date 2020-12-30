from tools import eval

if __name__ == '__main__':
    config_path = "./configs/rec/rec_yzm_train.yml"
    eval.main(config_path)


#python3 tools/eval.py -c configs/rec/rec_icdar15_train.yml -o Global.checkpoints={path/to/weights}/best_accuracy
