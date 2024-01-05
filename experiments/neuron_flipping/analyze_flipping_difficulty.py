import sys

sys.path.append("../")
from utils import *
from models import *
from dataloader import *
import argparse
from attribution_utils import *

import params

def dir_to_args(args):
    # f = logs/svhn/resnet50_lr_0.001_noise_0.05_resnet50_cosine_seed_6_aug_1
    f = args["from_dir_name"]
    print(f)
    args["dataset1"] = f.split("/")[1]

    f_remaining = f.split("/")[2]
    #f_remaiing = resnet50_lr_0.001_noise_0.05_resnet50_cosine_seed_6_aug_1

    args["model_type"] = f_remaining.split("_")[0]
    args["lr1"] = float(f_remaining.split("_")[2])
    args["noise_1"] = float(f_remaining.split("_")[4])
    args["sched"] = f_remaining.split("_")[6]
    args["seed"] = int(f_remaining.split("_")[8])
    args["augmentation"] = int(f_remaining.split("_")[10])
    try:
        args["cscore"] = float(f_remaining.split("_")[12])
    except:
        args["cscore"] = 0.0
    return args
 
# 从命令行读取参数
parser = params.parse_args()
args = parser.parse_args()
args = params.add_config(args) if args.config_file != None else args
args = vars(args)
if args["from_dir_name"] is not None: args = dir_to_args(args)
args["dataset2"] = args["dataset1"]
if args["model_type"] == "vit": args["batch_size"] = 128
# seed_everything(args["seed"])

# 检查是否已经训练过这个模型
filename = f'logs/{args["dataset1"]}/{args["model_type"]}_lr_{args["lr1"]}_noise_{args["noise_1"]}_{args["model_type"]}_{args["sched"]}_seed_{args["seed"]}_aug_{args["augmentation"]}_cscore_{args["cscore"]}/'
model_pickle = f'{filename}models.pickle'
if (not os.path.exists(model_pickle)):
   print ("Model not trained yet")
   exit(0)

#Load Model
print ("######### Loading Saved Model ###########")
with open(model_pickle, "rb") as f:
   all_models = pickle.load(f)

# 加载数据集
pre_dict, ft_dict = return_loaders(args, get_frac=False, shuffle = False, aug = False)

saved_model = get_model(f"{args['model_type']}")
# 加载最后一个epoch的模型
saved_model.load_state_dict(all_models[-1])
# saved_model.load_state_dict(torch.load(model_name))
train_loader = pre_dict["train_loader"]

# 评估在反转某一个样本的预测前，模型在训练集上的准确率
print (f"Initial accuracy on training set = {eval(saved_model, train_loader, eval_mode = True)['accuracy']}")


num_examples = 200


# channel_wise = channel, weight
# objective = "zero", "step"
# 这个example_type的取值为['noisy','clean']中的一个
# 取值为noisy时，表示对某个噪声样本进行反转，取值为clean时，表示对某个干净样本进行反转
# 计算反转样本的预测所需要置零的神经元的数量，以及这些神经元的分布
rets = flip_preds(saved_model, 
                            loader = pre_dict["train_loader"], 
                            example_type=args["example_type"], 
                            noise_mask= torch.from_numpy(pre_dict["noise_mask"]), 
                            rare_mask = torch.from_numpy(pre_dict["rare_mask"]) if pre_dict["rare_mask"] is not None else None, 
                            eval_post_edit=True, 
                            num_examples = num_examples, 
                            verbose = False,
                            channel_wise = args["channel_wise"],
                            gaussian_noise=args["gaussian_noise"],
                            objective = args["objective"],
                            n_EoT=args["n_EoT"])


import pickle

# 保存结果至pickle文件
with open(f"{filename}{args['example_type']}_flips_{args['channel_wise']}_wise_{args['objective']}_gaussian_{args['gaussian_noise']}.pickle", "wb") as output_file:
   pickle.dump(rets, output_file)