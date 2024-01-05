from models import *
import torch
import numpy as np

class FixedDropout(torch.nn.Module):
    def __init__(self, p, drop_mode = "train"):
        super(FixedDropout, self).__init__()
        # Implements Sparse dropout. This essentially simulates a neural network with lower capacity.
        # We deterministically 0 out the same neurons at train and test time
        self.p = p
        self.drop_mode = drop_mode

    def forward(self, x, idx, epoch = 0):
        torch.manual_seed(0)
        mask = (torch.rand_like(x) > self.p).float() # Random mask with dropout probability
        x = x * mask # Apply dropout mask
        x = x / (1 - self.p) # Scale up the output to compensate for the dropped neurons
        return x
    
class StandardDropout(torch.nn.Module):
    def __init__(self, p, drop_mode = "train"):
        super(StandardDropout, self).__init__()
        # Implements Standard dropout. 
        self.p = p
        self.drop_mode = drop_mode

    def forward(self, x, idx, epoch = 0):
        if self.drop_mode != "train":
            return x
        mask = (torch.rand_like(x) > self.p).float() # Random mask with dropout probability
        x = x * mask # Apply dropout mask
        x = x / (1 - self.p) # Scale up the output to compensate for the dropped neurons
        return x



class ExampleTiedDropout(torch.nn.Module):
    # this class is similar to batch tied dropout, 
    # but instead of tying neurons in a batch, we tie neurons in a set of examples

    def __init__(self, p_fixed = 0.2, p_mem = 0.1, num_batches = 100, drop_mode = "train"):
        super(ExampleTiedDropout, self).__init__()
        self.seed = 101010
        self.max_id = 60000
        self.p_mem = p_mem
        self.p_fixed = p_fixed
        self.drop_mode = drop_mode
        self.mask_tensor = None

    def forward(self, X, idx, epoch = 0):
        # 这里的idx参数应该是一个tensor，里面是样本的id，例如[3,5,8]，代表了选中了第3、5、8个样本
        # 当idx中的样本被取到时，前p_fixed的神经元/通道是始终保留的，后面的神经元/通道中，约有随机的p_mem*shape个神经元/通道是保留的
        # 当非idx中的样本被取到时，只前p_fixed的神经元/通道是始终保留的，后面的神经元/通道都是被dropout的
        if self.p_fixed == 1:
            return X
        if self.drop_mode == "train":
            # 训练模式下，每个样本都有一个mask，每个样本会根据自己的mask进行dropout
            mask = torch.zeros_like(X).cpu()
            shape = X.shape[1] # 通道数量或神经元数量
            if epoch > 0:
                # 之前已经计算过每个样本的mask，直接取出来就行
                mask = self.mask_tensor[idx]
            elif epoch == 0:
                mask[:, :int(self.p_fixed*shape)] = 1 # 前p_fixed的神经元/通道是始终保留的
                p_mem = self.p_mem

                # 除去前p_fixed的神经元/通道，剩余的神经元/通道数量。为这些神经元/通道生成mask
                shape_of_mask = shape - int(self.p_fixed*shape)
                for i in range(idx): # 创建每个被选中样本的mask
                    torch.manual_seed(idx[i].item())
                    # 使用伯努利分布生成mask
                    # 生成的mask里的每个元素，有p_mem的概率是1，有1-p_mem的概率是0
                    curr_mask = torch.bernoulli(torch.full((1, shape_of_mask), p_mem))
                    curr_mask = curr_mask.unsqueeze(-1).unsqueeze(-1)
                    # 每个样本，前p_fixed的神经元/通道是始终保留的，后面的神经元/通道中，约有随机的p_mem*shape个神经元/通道是保留的
                    mask[i][int(self.p_fixed*shape):] = curr_mask
                if self.mask_tensor is None:
                    self.mask_tensor = torch.zeros(self.max_id, X.shape[1], X.shape[2], X.shape[3])
                #assign mask at positions given by idx
                self.mask_tensor[idx] = mask
            
            # 根据mask进行dropout
            X = X * mask.cuda()


        elif self.drop_mode == "test":
            # 在test模式下，我们会根据p_mem，对非固定的神经元/通道进行归一化
            # 固定的神经元/通道不会进行归一化
            shape = X.shape[1]
            X[:, :int(self.p_fixed*shape)] = X[:, :int(self.p_fixed*shape)]
            X[:, int(self.p_fixed*shape):] = X[:, int(self.p_fixed*shape):]*self.p_mem

        elif self.drop_mode == "drop":
            # 在drop模式下，会将非固定的神经元/通道全部置为0
            # 固定的神经元/通道会进行放大
            shape = X.shape[1]
            X[:, int(self.p_fixed*shape):] = 0
            X[:, :int(self.p_fixed*shape)] = X[:, :int(self.p_fixed*shape)]*(self.p_fixed + self.p_mem)/self.p_fixed
        
        return X

