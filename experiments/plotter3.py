import pickle
import matplotlib.pyplot as plt
import numpy as np


def create_bins(min, max, step):
    bins = []
    while min < max:
        bins.append(min)
        min += step
    return bins


if __name__ == '__main__':
    # 加载数据，这些数据可以通过运行experiments/neuron_flipping/analyze_flipping_difficulty.py获得
    base_dir = "logs/cifar10/resnet9_lr_0.01_noise_0.1_resnet9_cosine_seed_4_aug_0_cscore_0.0"
    with open(f"{base_dir}/clean_flips_None_wise_None_gaussian_1.pickle", 'rb') as f:
        clean_flips = pickle.load(f)
    with open(f"{base_dir}/noisy_flips_None_wise_None_gaussian_1.pickle", 'rb') as f:
        noisy_flips = pickle.load(f)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 子图0: 反转一个干净样本和嘈杂样本的预测需要置零的神经元/通道的数量的分布
    min_iter = min(min(clean_flips['iters']), min(noisy_flips['iters']))
    max_iter = max(max(clean_flips['iters']), max(noisy_flips['iters']))
    iter_bins = create_bins(min_iter, max_iter, 5)

    ax[0].hist(clean_flips['iters'], bins=iter_bins, alpha=0.5, label='clean', color='green', density=True)
    ax[0].hist(noisy_flips['iters'], bins=iter_bins, alpha=0.5, label='noisy', color='red', density=True)
    ax[0].set_xlabel('Number of iterations')
    ax[0].set_ylabel('Probability Density')
    ax[0].legend()

    # 子图1：反转一个干净样本和嘈杂样本的预测后，模型在剩余的干净样本上的准确率的分布
    bins = np.linspace(0, 1, 21)

    ax[1].hist(clean_flips['clean_acc'], bins=bins, alpha=0.5, label='clean', color='green', density=True)
    ax[1].hist(noisy_flips['clean_acc'], bins=bins, alpha=0.5, label='noisy', color='red', density=True)
    ax[1].set_xlabel('Accuracy on clean samples')
    ax[1].set_ylabel('Probability Density')
    ax[1].legend()
    
    #子图2：被反转的神经元/通道在模型中的分布
    noisy_flips_param_layers = [param_name for params_of_one_example in noisy_flips['params'] for param_name in params_of_one_example]
    noisy_flips_param_layers = list(map(lambda x: int(x.split('.')[0]), noisy_flips_param_layers))
    clean_flips_param_layers = [param_name for params_of_one_example in clean_flips['params'] for param_name in params_of_one_example]
    clean_flips_param_layers = list(map(lambda x: int(x.split('.')[0]), clean_flips_param_layers))
    
    min_layer= min(min(clean_flips_param_layers), min(noisy_flips_param_layers))
    max_layer = max(max(clean_flips_param_layers), max(noisy_flips_param_layers))
    layer_bins = create_bins(min_layer, max_layer, 1)
    
    ax[2].hist(clean_flips_param_layers, alpha=0.5, label='clean', color='green', density=True, bins=layer_bins)
    ax[2].hist(noisy_flips_param_layers, alpha=0.5, label='noisy', color='red', density=True, bins=layer_bins)
    ax[2].set_xlabel('Layer')
    ax[2].set_ylabel('Probability Density')
    ax[2].legend()

    plt.savefig(f"{base_dir}/flipping.pdf")

                                                    
