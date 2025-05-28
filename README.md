# 深度强化学习算法在 CartPole-v1 环境下的优化实践

本项目在 OpenAI Gym 的经典环境 CartPole-v1 上，实践了深度强化学习 (Deep Reinforcement Learning) 算法的训练和优化过程，
主要围绕 Policy Gradient (PG) 和 Deep Q-Networks (DQN) 及各自的变体展开。通过版本迭代、超参数调优和网络结构改进，
探索提升智能体学习效率和性能的方法。

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

### 1. DQN系列

-   **`dqn_test`:**
    *   初始的 DQN 实现基础版本。

-   **`dqn_test1`:**
    *   基于 `dqn_test`。
    *   **优化评估流程和指标：** **去除了可视化渲染以加速运行；修改模型逻辑，一旦 episode 的 frame 达到 500 (即成功) 就立即停止该 episode；重复训练和评估 30 次独立的运行。评估指标为这 30 次运行的总体平均达成 episode 数（未在最大 episode 数 (200) 内成功的运行，计为耗费 200 个 episode），目标是最小化此平均值。** 

-   **`dqn_test1_optuna`:**
    *   一个专门用于超参数优化的版本，基于 `dqn_test1` 的 DQN 算法架构。
    *   使用 Optuna 库和对应的算法，在定义好的搜索空间内寻找一组相对最优的超参数，以最小化平均达成 episode 数。

-   **`dqn_test2`:**
    *   基于 `dqn_test1` 的 DQN 算法架构。
    *   **应用优化参数：** 将 `dqn_test1_optuna` 学习得到的新的超参数（针对普通 DQN 优化）应用到模型之中。

-   **`dqn_test3`:**
    *   基于 `dqn_test2` 的代码结构。
    *   **改进算法架构：** 将 DQN 算法架构改为 Double DQN (DDQN)。
    *   **参数沿用/初始参数：** **沿用了 `dqn_test1_optuna` 为普通 DQN 找到的优化超参数**。

-   **`dqn_test3_optuna`:**
    *   一个专门用于超参数优化的版本，基于 `dqn_test3` 的 DDQN 算法架构。
    *   使用 Optuna 库，为 DDQN 模型寻找一组更加合理的超参数。

-   **`dqn_test4`:**
    *   基于 `dqn_test3` 的 DDQN 算法架构。
    *   **应用新的优化参数：** 将 `dqn_test3_optuna` 学习得到的新的超参数（针对 DDQN 优化，包括网络结构参数如 HIDDEN_SIZE）应用到模型之中。

### 2. Policy Gradient (PG) 系列

-   **`pg_test`:**
    *   初始的 Policy Gradient (REINFORCE) 实现基础版本。使用自定义奖励 (`rtheta + r + rx`) 进行学习，包含回合平均回报的 Baseline 和标准化。

-   **`pg_test1`:**
    *   基于 `pg_test`。
    *   **优化评估流程和指标：** **去除了可视化渲染以加速运行；修改模型逻辑，一旦 episode 的 frame 达到 500 (即成功) 就立即停止该 episode；重复训练和评估 30 次独立的运行。评估指标为这 30 次运行的总体平均达成 episode 数（未在最大 episode 数 (200) 内成功的运行，计为耗费 200 个 episode），目标是最小化此平均值。**

-   **`pg_test1_optuna`:**
    *   一个专门用于超参数优化的版本，基于 `pg_test1` 的 PG 算法架构。
    *   使用 Optuna 库，在定义好的搜索空间内寻找一组针对 PG 的优化超参数 (LR, GAMMA, 隐藏层大小)。

-   **`pg_test2`:**
    *   基于 `pg_test1` 的 PG 算法架构。
    *   **应用优化参数：** 将 `pg_test1_optuna` 学习得到的针对 PG 的优化超参数应用到模型之中 (LR, GAMMA, 隐藏层大小)。网络使用 Tanh 激活，Normal 初始化。

-   **`pg_test3`:**
    *   基于 `pg_test2` 的代码结构。
    *   **改进算法架构：** 将 PG 框架改进为简单的 Actor-Critic (AC) Agent，引入了 Critic 网络学习 Value Function 作为 Baseline，并加入了 Entropy Bonus。
    *   **参数来源：** **使用了 `pg_test3_optuna` 学习得到的针对 AC 的优化超参数** (LR, GAMMA, HIDDEN_SIZE, Critic LR Ratio, Critic Loss Coeff, Entropy Coeff)。

-   **`pg_test3_optuna`:**
    *   一个专门用于超参数优化的版本，基于 `pg_test3` 的 Actor-Critic 算法架构。
    *   使用 Optuna 库，为 Actor-Critic 模型寻找优化超参数。

-   **`pg_test4`:**
    *   基于 `pg_test1` 的 PG 算法（使用原始环境奖励 + 回合平均 Baseline）框架。
    *   **放弃 AC 改进，专注于网络结构优化：** 在 `pg_test3` (应用优化参数后仍性能不佳) 表现不理想后，放弃了 AC 架构的进一步优化，**转而探索优化基础 PG 模型的网络结构**。
    *   **网络结构改进：** 将网络激活函数改为 ReLU，权重初始化改为 Kaiming 初始化。**沿用 `pg_test1_optuna` 的 LR 和 GAMMA 参数**。尝试了不同数量和大小的隐藏层组合（在此报告中，选取 `pg_test4.txt` 中表现最佳的配置，即 2 层隐藏层，大小为 16）。

## 优化结果对比

下表总结了不同主要模型版本在 CartPole-v1 环境下，经过30次独立运行后的性能统计。所有版本均采用统一的评估流程和指标。
**评估标准：** 智能体在单个 episode 中达到 500 帧视为成功。表格中的 **平均达成 Episode (总体)** 是包含失败运行（记为 200 episode）在内的总体平均值，目标是最小化此平均值。

| 版本名称                                            | 算法 | 平均达成 Episode (总体, 失败记为200) | 成功次数 (共30次) | 失败次数 (共30次) | 最佳成绩 (成功运行达成 Episode) | 最差成绩 (总体 Episode) | 备注                                                                 |
| :-------------------------------------------------- | :--- | :----------------------------------: | :----------------: | :----------------: | :-----------------------------: | :---------------------: | :------------------------------------------------------------------- |
| **1. 原始 DQN (Baseline Eval)** (`dqn_test1`)         | DQN  |                130.5                 |         30         |          0         |               34                |           150           | DQN基线性能                                                          |
| **2. DQN + DQN优化参数** (`dqn_test2`)                | DQN  |                 92.1                 |         27         |          3         |               36                |           200           | DQN应用Optuna优化结果                                                |
| **3. DDQN + DQN优化参数** (`dqn_test3`)               | DDQN |                 91.0                 |         28         |          2         |               42                |           200           | DDQN使用DQN优化参数，性能略优于DQN+DQN优化参数                       |
| **4. DDQN + DDQN优化参数** (`dqn_test4`)              | DDQN |                 **70.9**             |         27         |          3         |               **23**            |           200           | **DDQN应用专门Optuna优化结果，最优性能**                             |
| **5. 原始 PG (Baseline Eval, Custom Reward)** (`pg_test1`)| PG   |                164.6                 |         18         |         12         |               81                |           200           | PG基线性能 (使用自定义奖励)                                          |
| **6. PG + PG优化参数 (Custom Reward)** (`pg_test2`)   | PG   |                142.4                 |         20         |         10         |               39                |           200           | PG应用Optuna优化结果 (使用自定义奖励)                                |
| **7. AC + AC优化参数** (`pg_test3`)                   | AC   |                196.9                 |          2         |         28         |               146               |           200           | **AC应用专门Optuna优化参数，性能极差**           |
| **8. PG + 优化网络结构 (2层, size 16, Original Reward)** (`pg_test4`最佳) | PG   |                121.8                 |         20         |         10         |               29                |           200           | PG使用原始奖励，优化网络 (ReLU, Kaiming, 2L, size 16), 沿用PG Optuna的LR/GAMMA |

---

### 各版本使用的主要配置和参数来源

| 配置项                          | `dqn_test1` (基线) | `dqn_test2` (DQN优化) | `dqn_test3` (DDQN+DQN参数) | `dqn_test4` (DDQN优化) | `pg_test1` (PG基线) | `pg_test2` (PG优化) | `pg_test3` (AC优化) | `pg_test4` (PG+Net优化) |
| :------------------------------ | :----------------: | :-------------------: | :------------------------: | :--------------------: | :-----------------: | :-----------------: | :-----------------: | :----------------------: |
| **算法架构**                    |        DQN         |          DQN          |            DDQN            |          DDQN          |         PG          |         PG          |         AC          |            PG            |
| **主要参数来源**                |      硬编码值      |   `dqn_test1_optuna`  |    `dqn_test1_optuna`      |   `dqn_test3_optuna`   |     硬编码值       |  `pg_test1_optuna`  |  `pg_test3_optuna`  |    `pg_test1_optuna`     |
| `BATCH_SIZE`                    |        100         |          256          |            256             |           32           |        N/A          |        N/A          |        N/A          |           N/A            |
| `LR`                            |       0.01         |       0.00243         |         0.00243          |        0.00745         |        0.01         |      0.02016        |      0.02400        |         0.02016          |
| `GAMMA`                         |        0.9         |        0.9556         |          0.9556          |         0.9160         |         0.9         |       0.9860        |       0.9319        |          0.9860          |
| `TARGET_NETWORK_REPLACE_FREQ` |        100         |          300          |            300             |           50           |        N/A          |        N/A          |        N/A          |           N/A            |
| `MEMORY_CAPACITY`               |        1000        |          1000         |            1000            |           3000         |        N/A          |        N/A          |        N/A          |           N/A            |
| `EPSILON_START`                 |        0.9         |        0.9212         |          0.9212          |         0.8694         |        N/A          |        N/A          |        N/A          |           N/A            |
| `EPSILON_END`                   |        0.1         |        0.0450         |          0.0450          |         0.0622         |        N/A          |        N/A          |        N/A          |           N/A            |
| `EPSILON_DECAY_STEPS`           |        --          |          7000         |            7000            |           1500         |        N/A          |        N/A          |        N/A          |           N/A            |
| `HIDDEN_SIZE`                   |        10          |          64           |             64             |          128           |        10          |         16          |         16          |            16            |
| `CRITIC_LR_RATIO`               |        N/A         |          N/A          |            N/A             |           N/A          |        N/A          |        N/A          |       1.3849        |           N/A            |
| `CRITIC_LOSS_COEFF`             |        N/A         |          N/A          |            N/A             |           N/A          |        N/A          |        N/A          |       0.2085        |           N/A            |
| `ENTROPY_COEFF`                 |        N/A         |          N/A          |            N/A             |           N/A          |        N/A          |        N/A          |      0.00509        |           N/A            |
| **网络结构**                    |   1L, Size 10      |     1L, Size 64       |       1L, Size 64        |      1L, Size 128      |    2L, Size 10      |     2L, Size 16     |     2L, Size 16     |      2L, Size 16         |
| **网络激活/初始化**             |  Tanh, Normal      |    ReLU, Normal*      |     ReLU, Normal*        |     ReLU, Normal*      |  Tanh, Normal       |  Tanh, Normal       |  Tanh, Normal       |      ReLU, Kaiming       |
| **学习奖励信号**                |      自定义        |        自定义         |          自定义          |         自定义         |      自定义        |      自定义        |      自定义        |          原始          |
| **Baseline 方法**               |        无          |          无           |            无              |           无           | 回合均值+标准化      | 回合均值+标准化      |   Learned Value     |    回合均值+标准化       |

*注：DQN系列在 `dqn_test1_optuna` 引入了 ReLU 激活，但权重初始化保持 Normal，这并非 ReLU 的最佳实践（应使用 Kaiming init）。PG系列在 `pg_test4` 中纠正为 ReLU + Kaiming init。*

---

### 结果分析总结

本次实验通过多个版本的迭代和两波优化尝试，对 DQN/DDQN 和 Policy Gradient (PG) 两种核心的深度强化学习算法在 CartPole-v1 环境下的性能进行了对比评估。主要的发现聚焦于两种算法家族在学习效率上的差异，以及不同优化手段带来的影响。

*   **基线性能对比：**
    在初始实现和统一评估流程下 (`dqn_test1` vs `pg_test1`)，**DQN 的基线性能明显优于 Policy Gradient**。DQN 版本平均达成 episode 数为 130.5，且所有运行均成功；而 Policy Gradient 版本平均达成 episode 数为 164.6，仅有 60% 的运行成功。这表明在 CartPole 这样相对简单的任务上，基于值函数学习的 DQN 在初期展现出更高的稳定性和效率。

*   **第一波优化：超参数调优的影响：**
    通过 Optuna 对两种算法各自进行第一轮超参数优化后，两者性能均有所提升。`dqn_test2` (DQN + DQN优化参数) 的平均达成 episode 降至 92.1，`pg_test2` (PG + PG优化参数) 降至 142.4。**尽管两者都有进步，但 DQN 算法在应用优化参数后依然保持着对 Policy Gradient 的显著优势。** 这进一步印证了 Optuna 调优的有效性，同时也显示出即使经过调优，基础 PG 的学习效率仍不如 DQN。

*   **第二波优化：算法变体与网络结构改进：**
    在第一波优化的基础上，我们尝试了进一步的改进：
    *   **DQN -> DDQN：** 引入 Double DQN 架构 (`dqn_test3` 和 `dqn_test4`)。即使沿用为普通 DQN 优化的参数 (`dqn_test3` 91.0)，DDQN 性能也略优于 `dqn_test2`。而当为 DDQN 专门进行超参数优化 (`dqn_test3_optuna` 结果应用于 `dqn_test4`) 后，性能达到了本次实验的最佳 (平均 70.9)，最佳单次成绩为 23 episode。这强势证明了 DDQN 架构结合为其量身定制的超参数所带来的巨大提升。
    *   **PG -> AC / PG网络改进：** 在 Policy Gradient 方面，尝试了 Actor-Critic 变体 (`pg_test3`)。然而，**即使应用了专门为其优化的超参数，`pg_test3` 的性能却非常糟糕** (平均 196.9，几乎全部失败)。这提示该特定 AC 实现或其参数组合在此任务上存在严重问题。放弃 AC 后，回到基础 PG 框架 (`pg_test4`)，通过优化网络结构（ReLU激活、Kaiming初始化、调整隐藏层大小，并切回原始环境奖励）后，性能提升至 121.8，优于 `pg_test2`。这表明网络层面的优化是 PG 的一个有效改进方向。

*   **最终性能对比：**
    对比经过各自最有效优化后的版本 (`dqn_test4` vs `pg_test4` 的最佳配置)，**Double DQN (DDQN) 模型的平均达成 episode 数 (70.9) 显著低于优化网络结构后的 Policy Gradient 模型 (121.8)**。DDQN 不仅平均学习效率更高，也取得了本次实验中的最佳单次运行成绩 (23 episode)。

**结论：**

本次实验在 CartPole-v1 环境下，通过两波精心的优化尝试，清晰地展现了 **DQN/DDQN 算法家族在学习效率和性能上，始终优于 Policy Gradient 算法家族**。虽然 Policy Gradient 也从超参数调优和网络结构改进中受益，但未能弥合与最优 DQN/DDQN 之间的差距。Double DQN 结合为其专门调优的超参数，是本次实验中表现最佳的模型。同时，Actor-Critic 在本次实践中的不佳表现也提示算法变体的选择和实现对性能有重要影响。