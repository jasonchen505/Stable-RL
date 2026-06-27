# Stable-RL 增量学习笔记

> 基于 8卡4090 复现计划的学习过程记录
>
> 记录从"理解论文"到"动手复现"过程中新学到的关键点

---

## 一、从理论到实践的认知差距

### 1.1 论文中不会告诉你的事

| 论文描述 | 实际情况 | 学到的教训 |
|---------|---------|-----------|
| "DPPO 显著优于 GRPO" | 在简单任务（GSM8K）上优势不明显 | 选择合适的benchmark很重要 |
| "Binary 近似 O(1) 计算" | 需要额外存储 rollout_prob，增加5%内存 | 理论复杂度≠实际开销 |
| "超参数鲁棒" | 散度阈值 δ 需要仔细调优 | 没有免费的午餐 |
| "单卡可训练" | 需要全开 offload，训练很慢 | 实际可行≠实际高效 |

### 1.2 复现过程中的关键发现

**发现1: Offload 是把双刃剑**
```python
# 论文中说的
param_offload=True  # 省显存

# 实际体验
param_offload=True  # 显存省了，但训练速度慢了2-3倍
                    # 因为每次前向/反向都要从CPU加载参数
```

**发现2: vLLM 显存占比需要精细调优**
```python
# H100 配置
gpu_memory_utilization=0.1  # 只用8GB，很省

# 4090 配置
gpu_memory_utilization=0.3  # 用7.2GB，刚好够用
gpu_memory_utilization=0.5  # 用12GB，可能OOM
gpu_memory_utilization=0.1  # 只用2.4GB，推理太慢
```

**发现3: Batch size 不是越大越好**
```python
# 论文默认
nproc_per_gpu=128  # H100 上的配置

# 4090 实际
nproc_per_gpu=16   # 刚好能跑
nproc_per_gpu=32   # 可能OOM
nproc_per_gpu=8    # 稳定但慢
```

---

## 二、工程细节的深入理解

### 2.1 FSDP Offload 机制

**之前理解**: Offload 就是把参数移到 CPU

**深入学习后**:
```
┌─────────────────────────────────────────────────────────────────┐
│                    FSDP Offload 工作流程                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 初始化阶段:                                                  │
│     - 模型参数加载到 CPU                                          │
│     - Optimizer 状态在 CPU 创建                                   │
│                                                                  │
│  2. 前向传播:                                                    │
│     - 从 CPU 加载参数到 GPU (临时)                                │
│     - 计算激活                                                    │
│     - 参数可以立即 offload 回 CPU (节省显存)                       │
│                                                                  │
│  3. 反向传播:                                                    │
│     - 从 CPU 加载参数到 GPU (临时)                                │
│     - 计算梯度                                                    │
│     - 梯度 offload 到 CPU                                        │
│     - 参数 offload 回 CPU                                        │
│                                                                  │
│  4. 参数更新:                                                    │
│     - 在 CPU 上用梯度更新参数                                     │
│     - 更新后的参数留在 CPU                                        │
│                                                                  │
│  代价: 大量 CPU-GPU 数据传输，训练速度慢 2-3 倍                   │
└─────────────────────────────────────────────────────────────────┘
```

**代码位置**: `verl/workers/actor/dp_actor.py`

### 2.2 Gradient Checkpointing 的实际效果

**之前理解**: 省50-60%显存

**深入学习后**:
```
普通训练:
  - 保存所有层的激活用于反向传播
  - 显存: O(L) 其中 L 是层数

Gradient Checkpointing:
  - 只保存部分层的激活
  - 反向传播时重新计算缺失的激活
  - 显存: O(sqrt(L))
  - 时间: +30% 额外计算

实际效果:
  - 1.5B 模型: 节省约 2-3GB 显存
  - 7B 模型: 节省约 8-10GB 显存
  - 训练时间增加约 20-30%
```

### 2.3 vLLM 的显存管理

**之前理解**: vLLM 就是推理引擎

**深入学习后**:
```
vLLM 显存组成:
├── 模型权重 (BF16): ~3GB (1.5B)
├── KV Cache: 动态分配，取决于序列长度和并发数
├── CUDA Graph: 固定开销
└── 临时缓冲: 前向计算用

gpu_memory_utilization 的作用:
- 控制 vLLM 可用的最大显存比例
- 太小: KV Cache 不够，推理慢
- 太大: 训练时 OOM

4090 上的调优:
- 1.5B: 0.3 (约7GB)
- 3B: 0.3 (约7GB)
- 7B: 0.35 (约8.4GB)
```

---

## 三、DPPO 实现细节的深入理解

### 3.1 概率比 vs 散度的代码实现

**之前理解**: 就是换了个公式

**深入学习后**:
```python
# PPO 实现 (core_algos.py:1325-1418)
def compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, ...):
    ratio = torch.exp(log_prob - old_log_prob)  # π_θ / π_old
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1-eps, 1+eps)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)  # max 操作
    return pg_losses

# DPPO 实现 (core_algos.py:1264-1308)
def _compute_token_mask_for_policy_loss(...):
    prob = torch.exp(log_prob)
    rollout_prob = torch.exp(rollout_log_prob)
    
    if loss_mode == "dppo_binary_tv":
        # Binary TV: 直接比较概率差
        invalid_positive_mask = (prob - rollout_prob) > cliprange_high
        invalid_negative_mask = (prob - rollout_prob) < -cliprange_low
    
    # 生成 mask
    invalid_mask = torch.where(advantages > 0, 
                               invalid_positive_mask, 
                               invalid_negative_mask)
    valid_mask = 1.0 - invalid_mask.float()
    return valid_mask
```

**关键区别**:
1. PPO 用 `exp(log_prob - old_log_prob)` 计算 ratio
2. DPPO 用 `prob - rollout_prob` 直接比较概率
3. DPPO 的 mask 是基于散度的，不是基于 ratio 的

### 3.2 Rollout Log Prob 的存储

**之前理解**: 只需要存 old_log_prob

**深入学习后**:
```
传统 PPO:
- rollout 时: 计算 log_prob，存储为 old_log_prob
- 训练时: 用当前策略计算 log_prob，与 old_log_prob 比较

DPPO:
- rollout 时: 计算 log_prob，存储为 rollout_log_prob
- 训练时: 用当前策略计算 log_prob
- 关键区别: 必须用 rollout 时的概率，不能重算

代码实现:
# ray_trainer.py 中
batch.batch['rollout_log_prob'] = rollout_log_prob  # 额外存储
batch.batch['old_log_prob'] = rollout_log_prob       # DPPO 中两者相同
```

### 3.3 Clip Ratio 的含义差异

**之前理解**: DPPO 的 clip_ratio 和 PPO 一样

**深入学习后**:
```
PPO:
- clip_ratio = ε = 0.2
- 含义: ratio 被限制在 [0.8, 1.2]
- 单位: 无量纲 (ratio)

DPPO:
- clip_ratio = δ = 0.1
- 含义: 概率差被限制在 [-0.1, 0.1]
- 单位: 概率 (0-1)

对应关系:
- PPO ε=0.2 大约对应 DPPO δ=0.1
- 因为 TV = (1/2)|r-1| ≈ (1/2)*0.2 = 0.1
```

---

## 四、训练动态的深入理解

### 4.1 Clip Fraction 的含义

**之前理解**: clip fraction 越低越好

**深入学习后**:
```
Clip Fraction = 被 mask 的 token 比例

太低 (<5%):
- 信任域太宽松
- 可能导致训练不稳定
- 策略更新太激进

太高 (>30%):
- 信任域太严格
- 训练效率低
- 大部分梯度信号被丢弃

健康范围: 5-15%

监控方法:
- wandb.log({"actor/pg_clipfrac": clipfrac})
- 如果持续 >20%，考虑增大 δ
- 如果持续 <5%，考虑减小 δ
```

### 4.2 KL Divergence 的监控

**之前理解**: KL 越小越好

**深入学习后**:
```
KL Divergence 的几种含义:

1. actor/ppo_kl: 当前策略与 rollout 策略的 KL
   - 太大 (>0.5): 策略偏离太远，可能不稳定
   - 太小 (<0.01): 策略更新太保守
   - 健康范围: 0.01-0.1

2. ref_kl: 当前策略与参考策略的 KL (如果启用)
   - 用于防止 reward hacking
   - 通常作为正则项

3. DPPO 的散度:
   - Binary TV: |prob - rollout_prob|
   - Binary KL: KL(rollout || current)
   - 直接衡量分布差异，比 ratio 更准确
```

### 4.3 Entropy 的作用

**之前理解**: Entropy 越高越好（鼓励探索）

**深入学习后**:
```
Entropy 的动态:

训练初期:
- Entropy 较高（策略不确定）
- 模型在探索

训练后期:
- Entropy 下降（策略变确定）
- 正常现象

Entropy Collapse:
- Entropy 急剧下降到接近 0
- 模型只输出少数固定模式
- 通常是训练不稳定的信号

监控方法:
- 如果 Entropy 持续下降，可能需要:
  - 增大 entropy_coeff
  - 减小学习率
  - 使用 DPPO（更稳定的更新）
```

---

## 五、工程实践的关键教训

### 5.1 数据预处理的坑

**问题**: 数据格式不对导致训练失败

**解决方案**:
```python
# 正确的数据格式
{
    "data_source": "openai/gsm8k",
    "prompt": [
        {"role": "user", "content": "问题内容"}
    ],
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": "42"
    },
    "extra_info": {
        "split": "train",
        "index": 0
    }
}

# 常见错误:
# 1. prompt 格式不对（应该是 list of dict）
# 2. reward_model 缺少 ground_truth
# 3. 数据类型不对（应该是 string，不是 int）
```

### 5.2 内存不足的排查

**排查步骤**:
```bash
# 1. 检查 GPU 使用情况
nvidia-smi

# 2. 检查是否有其他进程占用
fuser -v /dev/nvidia*

# 3. 逐步降低参数
# 先降低 batch size
# 再降低 gpu_memory_utilization
# 再开启更多 offload
# 最后减小序列长度
```

### 5.3 训练中断的恢复

**检查点机制**:
```bash
# VeRL 会自动保存检查点
# 默认保存在 checkpoints/ 目录

# 恢复训练
python3 -m verl.trainer.main_ppo \
    trainer.resume_mode=auto \
    trainer.resume_from_checkpoint=checkpoints/xxx \
    ...
```

---

## 六、算法理解的深化

### 6.1 GRPO vs PPO 的本质区别

**之前理解**: GRPO 不需要 Critic

**深入学习后**:
```
PPO:
- 需要 Critic 网络估计 V(s)
- Advantage = R - V(s) + γ*V(s') - V(s)  (GAE)
- 优点: 精细的 token-level credit
- 缺点: Critic 训练不稳定，额外内存

GRPO:
- 不需要 Critic
- Advantage = R - mean(R_group)  (组内归一化)
- 优点: 简单，省内存
- 缺点: 粗糙的 trajectory-level credit

选择建议:
- 简单任务（GSM8K）: GRPO 足够
- 复杂任务（AIME）: 可能需要更好的 credit assignment
- Agent 任务: 考虑 ProxMO 或 VIMPO
```

### 6.2 为什么 DPPO 用 GRPO 而不是 GAE

**论文没说清楚的问题**:

```
DPPO 本身是独立于优势估计器的
- 可以用 GRPO
- 可以用 GAE (需要 Critic)
- 可以用 RLOO
- 可以用 REINFORCE++

论文选择 GRPO 的原因:
1. GRPO 是 LLM RL 的主流选择
2. 不需要 Critic，更简单
3. 实验对比更公平（都是 Critic-free）

实际上:
- DPPO + GAE 可能效果更好（更精细的 credit）
- 但需要额外训练 Critic
- 论文选择了最简单的配置
```

### 6.3 KL 惩罚的作用

**之前理解**: 防止策略偏离太远

**深入学习后**:
```
KL 惩罚的两种形式:

1. In-reward KL:
   r' = r - β * KL(π_θ || π_ref)
   - 直接修改奖励
   - 在 reward 中加入 KL 惩罚

2. KL Loss:
   L = L_policy + λ * KL(π_θ || π_old)
   - 在 loss 中加入 KL 惩罚
   - 控制每步更新的幅度

代码中的配置:
algorithm.use_kl_in_reward=False  # 不在 reward 中加 KL
actor.use_kl_loss=True           # 在 loss 中加 KL
actor.kl_loss_coef=0.001         # KL loss 系数
actor.kl_loss_type=low_var_kl    # KL 估计方法
```

---

## 七、复现过程中的问题与解决

### 7.1 问题1: 训练启动后立即 OOM

**症状**: 第一步就 OOM

**原因**: vLLM 初始化时需要分配 KV Cache

**解决方案**:
```bash
# 降低 vLLM 显存占比
actor_rollout_ref.rollout.gpu_memory_utilization=0.2

# 或者减少并发序列数
actor_rollout_ref.rollout.max_num_seqs=32
```

### 7.2 问题2: 训练速度极慢

**症状**: 一步要好几分钟

**原因**: Offload 导致大量 CPU-GPU 数据传输

**解决方案**:
```bash
# 1. 如果显存够，关闭部分 offload
actor_rollout_ref.actor.fsdp_config.param_offload=False

# 2. 减小序列长度
data.max_prompt_length=256
data.max_response_length=512

# 3. 减少采样数
actor_rollout_ref.rollout.n=3
```

### 7.3 问题3: 训练不收敛

**症状**: Reward 没有明显上升

**可能原因**:
1. 学习率太小
2. Batch size 太小
3. KL 惩罚太重
4. 数据有问题

**排查步骤**:
```bash
# 1. 检查数据
python -c "import pandas as pd; df = pd.read_parquet('~/data/gsm8k/train.parquet'); print(df.iloc[0])"

# 2. 增大学习率
actor_rollout_ref.actor.optim.lr=1e-4

# 3. 减小 KL 惩罚
actor_rollout_ref.actor.kl_loss_coef=0.0001

# 4. 增大 batch size（如果显存够）
data.train_batch_size=32
```

---

## 八、关键代码阅读笔记

### 8.1 core_algos.py 结构

```
core_algos.py (约2400行)
├── Advantage Estimator 注册机制 (L1-150)
│   ├── AdvantageEstimator 枚举类
│   ├── register_adv_est 装饰器
│   └── get_adv_estimator_fn 工厂函数
│
├── 优势估计器实现 (L150-1000)
│   ├── compute_gae_advantage_return (GAE)
│   ├── compute_grpo_outcome_advantage (GRPO)
│   ├── compute_rloo_outcome_advantage (RLOO)
│   ├── compute_reinforce_plus_plus_outcome_advantage
│   └── ... (共13种)
│
├── Policy Loss 注册机制 (L1000-1100)
│   ├── PolicyLossFn 类型定义
│   ├── POLICY_LOSS_REGISTRY
│   └── register_policy_loss 装饰器
│
├── Policy Loss 实现 (L1100-2000)
│   ├── compute_policy_loss_vanilla (PPO)
│   ├── compute_policy_loss_with_mask (DPPO 核心)
│   ├── compute_policy_loss_gspo (GSPO)
│   ├── compute_policy_loss_cispo (CISPO)
│   └── ... (共12种)
│
└── 工具函数 (L2000-2400)
    ├── agg_loss (loss 聚合)
    ├── kl_penalty (KL 计算)
    └── compute_entropy_loss
```

### 8.2 DPPO 核心代码路径

```
verl/trainer/ppo/core_algos.py:1241-1308

def _compute_token_mask_for_policy_loss(
    rollout_log_prob,    # rollout 时的概率
    old_log_prob,        # 旧策略概率
    log_prob,            # 当前策略概率
    advantages,          # 优势估计
    response_mask,       # 响应 mask
    config,              # 配置
    loss_mode,           # 损失模式 (ppo/dppo_*)
    ...
):
    prob = torch.exp(log_prob)
    rollout_prob = torch.exp(rollout_log_prob)
    
    if loss_mode == "dppo_binary_tv":
        # Binary TV: 直接比较概率
        invalid_positive_mask = (prob - rollout_prob) > cliprange_high
        invalid_negative_mask = (prob - rollout_prob) < -cliprange_low
    
    elif loss_mode == "dppo_binary_kl":
        # Binary KL: 计算 KL 散度
        kl = rollout_prob * (rollout_log_prob - log_prob) + \
             (1 - rollout_prob) * torch.log(...)
        invalid_positive_mask = (kl > cliprange_high) & (prob > rollout_prob)
        invalid_negative_mask = (kl > cliprange_low) & (prob < rollout_prob)
    
    # 生成 mask
    invalid_mask = torch.where(advantages > 0, 
                               invalid_positive_mask, 
                               invalid_negative_mask)
    valid_mask = 1.0 - invalid_mask.float()
    return valid_mask
```

---

## 九、下一步学习方向

### 9.1 算法层面

- [ ] 理解 VIMPO 的隐式 Value 机制
- [ ] 理解 ProxMO 的 PSC + PSA 机制
- [ ] 对比 DPPO 和 VIMPO 的异同
- [ ] 探索 DPPO 在 Agent 场景的应用

### 9.2 工程层面

- [ ] 学习 Megatron-LM 的并行策略
- [ ] 理解 vLLM 的 KV Cache 管理
- [ ] 探索 torch.compile 的优化效果
- [ ] 学习 slime 的 Chunked GAE 实现

### 9.3 实验层面

- [ ] 在 MATH 数据集上复现
- [ ] 尝试更大的模型（7B LoRA）
- [ ] 探索不同优势估计器的影响
- [ ] 分析训练动态的可视化

---

## 十、总结：从"知道"到"会做"

### 10.1 论文阅读 vs 动手复现

| 维度 | 论文阅读 | 动手复现 |
|------|---------|---------|
| **理解深度** | 知道公式和结论 | 理道实现细节和坑 |
| **超参数** | 看到最终值 | 理解调优过程 |
| **工程** | 忽略 | 必须面对 |
| **局限性** | 论文不会说 | 自己踩坑才知道 |

### 10.2 关键收获

1. **Offload 是必要的但有代价**: 省显存但慢2-3倍
2. **vLLM 显存需要精细调优**: 太小推理慢，太大训练OOM
3. **DPPO 的核心是 mask 机制**: 不是简单的公式替换
4. **监控指标很重要**: clip_fraction、kl_divergence、entropy
5. **简单任务上 DPPO 优势不明显**: 需要在复杂任务上验证

### 10.3 给后来者的建议

1. **先跑通最小实验**: 1.5B + GSM8K + 1 epoch
2. **逐步增加复杂度**: 先 GRPO，再 DPPO
3. **监控关键指标**: 不要只看 reward
4. **记录所有尝试**: 包括失败的实验
5. **多读代码**: 论文不会告诉你所有细节

---

*最后更新: 2026-06-26*
*基于 8x RTX 4090 复现过程的学习记录*
