# Pang Liu, CS130 Deep Learning, Fall 2025
# Project 2

# Note: I have some Chinese comments for better understanding
# They only occur in comments, not in code
# if you need consider details of the comments
# please contact me and I will change them to English
# otherwise, just ignore them, thanks!

import torch
import os
from train.checkpoint import Checkpointer

def make_optimizer(name: str, params, lr: float, momentum: float=None):
    """
    这里的目的是支持更多的优化器
    课上学习了的包括：
    GD：torch.optim.SGD实现，让DataLoader的Batch Size等于训练集大小
    因为有SGD，GD就不用再加入实现了
    
    对比内容：
    SGD：torch.optim.SGD实现，
    AdaGrad：torch.optim.Adagrad
    RMSProp：torch.optim.RMSprop
    Polyak Momentum：需要在torch.optim.SGD中设置momentum参数
    Nesterov Momentum：需要在torch.optim.SGD中设置momentum和nesterov参数
    Adam：torch.optim.Adam（课上刚提，可以先试试）

    分类对比：
    baseline: SGD
    adaptive gradient: AdaGrad, RMSProp
    momentum: Polyak Momentum, Nesterov Momentum
    Adam

    比较目的：
    baseline基准：SGD
    adaptive gradient对训练效果的提升（相较于baseline）
    momentum单独对训练效果的提升（相较于baseline）
    adaptive+momentum共同对训练效果的提升（相较于baseline）
    """
    name = name.lower() # 这里预设的这行意思是优化器名字统一用小写，增加robust
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    elif name == "adagrad":
        # adaptive gradient：AdaGrad
        return torch.optim.Adagrad(params, lr=lr)
    elif name == "rmsprop":
        # adaptive gradient：RMSProp
        return torch.optim.RMSprop(params, lr=lr)
    elif momentum==None and name in ["polyak", "nesterov"]:
        raise ValueError(f"Momentum value must be provided for {name} optimizer.")
    elif name == "polyak":
        # momentum：Polyak Momentum（标准动量，sgd momentum的默认实现）
        # 经验来说0.9是一个不错的选择，但是这里还是对比一下吧
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    elif name == "nesterov":
        # momentum：SGD with Nesterov Momentum
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=True)
    elif name == "adam":
        # 自适应梯度 + 动量法：Adam
        return torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
#===================================================================================================================================

#===================================================================================================================================
# ADD YOUR CODES HERE
# Parameters modify: IMPORT hist, save_path from main
def save_results(hist, save_path):
    """
    保存实验中的数据，我们的目的是对比不同的优化器，所以参考以下对比角度：
    每张图有6种优化器的曲线：
        图一：training loss vs. Epochs
        图二：training accuracy vs. Epochs
        图三：validation loss vs. Epochs
        图四：validation accuracy vs. Epochs
    
    Final test accuracy and test loss做一个表格：
        表格一：最终的测试损失和测试准确率
        方法 | 图集：MLP-MNIST | CNN-CIFAR10 | VGG13-CIFAR10
        method1  | loss, acc     | loss, acc   | loss, acc
        method2 | loss, acc     | loss, acc   | loss, acc
        ...
    
    checkpoint.py 中的 Checkpointer 类已经实现了保存检查点的功能：
        ckpt = Checkpointer(ckpt_dir="checkpoints", max_to_keep=5)
        ckpt.save(epoch, model, optimizer, hist, config={"lr": lr, "batch_size": batch_size})
    可以直接从 checkpoint.py 保存的检查点文件中读取 hist 字典来获取这些数据
    所以作业推荐这里直接保存必要信息，绘图的时候直接调用就行了
    """
    with open(save_path, 'w') as f:
        f.write("--- Final Training Results ---\n")
        f.write(f"Final Training Loss: {hist['train_loss'][-1]:.4f}\n")
        f.write(f"Final Training Accuracy: {hist['train_acc'][-1]:.2f}%\n")
        f.write(f"Final Validation Loss: {hist['val_loss'][-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {hist['val_acc'][-1]:.2f}%\n")
        f.write("\n--- Final Test Results ---\n")
        f.write(f"Test Loss: {hist['test_loss']:.4f}\n")
        f.write(f"Test Accuracy: {hist['test_acc']:.2f}%\n")
    print(f"Results summary saved to {save_path}")


#===================================================================================================================================




#===================================================================================================================================
# MODIFY YOUR CODES HERE
def run_experiment(main_func, model_type_name, output_dir_main, output_dir_momentum):
    """
    主程序，这个python脚本是最后推给HPC的，所以这里就是主程序
    包含实验中所需的所有内容：
    1、设置超参数
    2、设置检查点
    3、调用main函数
    4、保存实验结果

    Homework Requirements:
    100 epochs
    restults in a total of 15 combinations (5 optimizers x 3 models)
    1. TwoLayerMLP on MNIST (mnist_mlp.py)
    2. SimpleCNN on CIFAR-10 (cifar10_cnn.py)
    3. VGG13 on CIFAR-10 (cifar10_vgg.py

    Here I add polyak so it is 6 optimizers x 3 models

    Schedule 10 jobs in sequence

    附加一个保存：
    对比polyak momentum和nesterov momentum的动量设置
    设置0.5, 0.7, 0.9, 0.95, 0.99
    单独分别保存，之后报告中单独绘图展示作为附加实验

    """
    print("=====START TRAINING=====")
    print()
    # 1. 超参数
    LR = 1e-3
    EPOCHS = 100
    BATCH_SIZE = 128
    SEED = 42

    # 2. 根据不同文件调整输出文件名
    MODEL_TYPE_NAME = model_type_name                 
    OUTPUT_DIR_MAIN = output_dir_main          
    OUTPUT_DIR_MOMENTUM = output_dir_momentum  

    # 3. CUDA检查
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. 5+1种优化器（在作业要求上增加了polyak）
    MAIN_EXPERIMENT_CONFIG = {
        "sgd":      {"momentum": None},
        "adagrad":  {"momentum": None},
        "rmsprop":  {"momentum": None},
        "adam":     {"momentum": None},
        "polyak":   {"momentum": 0.9}, # 使用经验值0.9作为标准动量
        "nesterov": {"momentum": 0.9},
    }

    print(f"\n{'='*60}")
    print(f"  STARTING MAIN COMPARISON FOR MODEL: {MODEL_TYPE_NAME.upper()}")
    print(f"{'='*60}\n")
    os.makedirs(OUTPUT_DIR_MAIN, exist_ok=True)

    for optimizer_name, config in MAIN_EXPERIMENT_CONFIG.items():
        print(f"\n--- Running Main Task: Optimizer={optimizer_name.upper()} ---")

        momentum_val = config["momentum"]
        
        ckpt_filename = f"{MODEL_TYPE_NAME}_{optimizer_name}"
        if momentum_val is not None:
            ckpt_filename += f"_{momentum_val}"
        ckpt_filename += ".pth"
        ckpt_path = os.path.join(OUTPUT_DIR_MAIN, ckpt_filename)
        
        ckpt = Checkpointer(ckpt_path, device)

        # 调用 main 函数
        main_func(
            ckpt=ckpt, device=device, seed=SEED, lr=LR, epochs=EPOCHS,
            batch_size=BATCH_SIZE, optimizer_name=optimizer_name, momentum=momentum_val
        )

    # ==========================
    # 附加实验 (对比不同的动量值)
    # ==========================
    MOMENTUM_EXPERIMENT_CONFIG = {
        "polyak":   [0.5, 0.7, 0.9, 0.95, 0.99],
        "nesterov": [0.5, 0.7, 0.9, 0.95, 0.99]
    }
    
    print(f"\n{'='*60}")
    print(f"  EXTRA MOMENTUM STUDY FOR MODEL: {MODEL_TYPE_NAME.upper()}")
    print(f"{'='*60}\n")
    os.makedirs(OUTPUT_DIR_MOMENTUM, exist_ok=True)

    for optimizer_name, momentum_values in MOMENTUM_EXPERIMENT_CONFIG.items():
        for momentum_val in momentum_values:
            print(f"\n--- Running Bonus Task: Optimizer={optimizer_name.upper()}, Momentum={momentum_val} ---")

            # 设置检查点文件名，例如: mnist_mlp_nesterov_0.95.pth
            ckpt_filename = f"{MODEL_TYPE_NAME}_{optimizer_name}_{momentum_val}.pth"
            ckpt_path = os.path.join(OUTPUT_DIR_MOMENTUM, ckpt_filename)

            ckpt = Checkpointer(ckpt_path, device)

            # 调用 main 函数
            main_func(
                ckpt=ckpt, device=device, seed=SEED, lr=LR, epochs=EPOCHS,
                batch_size=BATCH_SIZE, optimizer_name=optimizer_name, momentum=momentum_val
            )

    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS FOR {MODEL_TYPE_NAME.upper()} COMPLETED.")
    print(f"{'='*60}\n")
#===================================================================================================================================



if __name__ == "__main__":
    print("THIS IS trainer.py")
    print("ALL CODE DETAILS ARE IN trainer.py FOR BETTER MODULARITY")
    print("PLEASE RUN mnist_mlp.py, cifar10_cnn.py, cifar10_vgg13.py TO EXECUTE THE EXPERIMENTS")