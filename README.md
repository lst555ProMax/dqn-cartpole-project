## 环境部署与依赖

### 1. Python 版本
本项目依赖 **Python 3.9.x**，具体版本由 `environment.yml` 文件精确指定。

### 2. 虚拟环境 (Anaconda / Miniconda)
为确保依赖隔离和环境一致性，强烈建议使用 Anaconda 或 Miniconda 创建并管理虚拟环境。

### 3. 依赖安装

1.  **前提条件：**
    *   已安装 Anaconda 或 Miniconda。
    *   **GPU 用户：** `environment.yml` 默认配置了 GPU 版本的 PyTorch (CUDA 12.1)。请确保您已安装兼容的 NVIDIA 显卡驱动。若无兼容 GPU 或希望使用 CPU 版本，请相应修改 `environment.yml` 文件中 PyTorch 及 CUDA 相关包的配置。

2.  **创建环境与安装依赖：**
    *   克隆或下载本项目。
    *   打开 Anaconda Prompt 或终端，导航至项目根目录（包含 `environment.yml` 文件的目录）。
    *   执行以下命令创建名为 `augmentLearning` (或 `environment.yml` 中 `name` 字段指定的名称) 的环境并安装所有依赖：
        ```bash
        conda env create -f environment.yml
        ```

3.  **激活环境：**
    ```bash
    conda activate augmentLearning
    ```

4.  **验证安装 (可选但推荐):**
    激活环境后，可以运行以下 Python 代码片段检查核心库（如 PyTorch）和 GPU 支持（如果适用）：
    ```python
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        # print(f"cuDNN version: {torch.backends.cudnn.version()}") # cuDNN版本检查可能需要额外配置
    ```

**说明:**
*   所有包及其精确版本均在 `environment.yml` 文件中定义，请直接参考该文件获取详细依赖列表。
*   如果 `conda env create` 失败，请仔细检查错误信息，通常与网络连接、包源或特定包的兼容性有关。GPU 用户尤其需要注意 CUDA 和驱动的匹配问题。

## 训练过程与版本迭代

本项目经历了以下主要的版本迭代和改进：

-   **`dqn_test`:**
    *   原始代码 + 注释

-   **`dqn_test1`:**
    *   去除了可视化的过程，加快了模型的运行速度
    *   修改模型逻辑，一旦frame达到500就停止运行，并且记录当前episode的值
    *   重复上一行操作30次，取episode的均值，并将该均值作为模型性能的衡量指标

-   **`dqn_test1_optuna`:**
    *   之前的代码里面超参数都是写死的，这个特殊的版本用optuna库和对应的算法寻找一组更加合理的超参数

-   **`dqn_test2`:**
    *   将上一版本学习得到的新的超参数应用到模型之中

-   **`dqn_test3`:**
    *   在上一版本的基础上继续改进DQN算法架构，改为Double DQN

-   **`dqn_test4`:**
    *   在上一版本的基础上继续改进神经网络，新加入一层大小为64的隐藏层


## 优化结果对比

下表总结了不同模型版本在 CartPole-v1 环境下，经过30次独立运行后的性能统计。
**评估标准：** 智能体在单个 episode 中达到 500 帧视为成功，记录达成此目标时的 episode 编号。目标是最小化平均达成 episode 数。

| 版本名称                 | 平均达成 Episode (越小越好) | 成功次数 (共30次) | 失败次数 (共30次) | 最佳成绩 (Episode) | 最差成绩 (Episode) |
| :----------------------- | :--------------------------: | :----------------: | :----------------: | :-----------------: | :-----------------: |
| **1. 原始 DQN** (`dqn_test`) |            127.6             |         30         |          0         |          41         |         159         |
| **2. DQN + 优化参数** (`dqn_test2`) |             81.5             |         27         |          3         |          39         |         181         |
| **3. DDQN + 优化参数** (`dqn_test3`) |             73.9             |         29         |          1         |          46         |         198         |
| **4. DDQN + 更深网络 + 优化参数** (`dqn_test4`) |            127.5             |         15         |         15         |          69         |         198         |

---

### 通过 Optuna 学习到的优化超参数
*(应用于版本 2, 3, 4)*

-   `BATCH_SIZE`: 256
-   `LR`: 0.0024303199955094507
-   `GAMMA`: 0.9556460180442318
-   `TARGET_NETWORK_REPLACE_FREQ`: 300
-   `MEMORY_CAPACITY`: 1000
-   `EPSILON_START`: 0.9212305525092727
-   `EPSILON_END`: 0.045039045287420844
-   `EPSILON_DECAY_STEPS`: 7000

---

### 结果分析总结

*   **有效措施：**
    *   **超参数优化 (Optuna)：** 显著提升了模型的平均学习效率。
    *   **引入 Double DQN (DDQN)：** 在优化参数的基础上，进一步提高了学习效率和成功率。

*   **无效/负面影响措施：**
    *   **增加网络深度 (更深网络)：** 在当前配置下，使模型性能大幅下降，学习效率降低，成功率减少。

---

## 进一步优化方向

*   **探索与隐藏层相关的超参数：** 层数、每一层的参数数量等等。
*   **优先经验回放 (PER)：** 重点学习更“有价值”的经验。
*   **竞争网络结构 (Dueling DQN)：** 改进 Q 值估计方式。
*   **噪声网络 (Noisy Nets)：** 替代 Epsilon-Greedy 进行更有效的探索。
*   **调整超参数以适应更深网络：** 如果仍想尝试更深网络，需针对性地调整学习率、初始化或增加 MEMORY_CAPACITY。
*   **Rainbow DQN：** 尝试结合多种 DQN 改进的集成方法。
*   **优化目标函数鲁棒性：** 在 Optuna 中对每组参数多次运行取平均，以获得更稳健的“最佳”参数。

---