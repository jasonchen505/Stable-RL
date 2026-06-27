# Stable-RL 8卡4090复现计划

> 基于实际硬件资源（8x RTX 4090 24GB）的完整复现方案
>
> 目标：系统学习并复现 DPPO 算法的核心实验

---

## 一、硬件资源评估

### 1.1 可用资源

| 资源 | 规格 |
|------|------|
| **GPU** | 8x RTX 4090 |
| **显存** | 24GB / 卡 |
| **总显存** | 192GB |
| **架构** | Ada Lovelace |
| **FP16/BF16** | 支持 |
| **NVLink** | 不支持（PCIe互联） |

### 1.2 资源限制分析

**4090 vs H100 关键差异**:

| 维度 | RTX 4090 | H100 (80GB) | 影响 |
|------|----------|-------------|------|
| **显存** | 24GB | 80GB | 需要更多offload |
| **互联带宽** | PCIe 4.0 (32GB/s) | NVLink (900GB/s) | 多卡通信瓶颈 |
| **FP16算力** | 82.6 TFLOPS | 989 TFLOPS | 训练速度慢~10x |
| **适合模型** | ≤7B (LoRA) | ≤72B | 模型规模受限 |

### 1.3 可行的模型配置

| 模型 | 训练方式 | 单卡可行性 | 8卡可行性 | 说明 |
|------|---------|-----------|-----------|------|
| **1.5B** | GRPO-LoRA | ✅ 完全可行 | ✅ 并行多实验 | 推荐首选 |
| **3B** | GRPO-LoRA | ✅ 可行 | ✅ 可行 | 需全开offload |
| **7B** | GRPO-LoRA | ⚠️ 勉强 | ✅ 可行 | 需精细调参 |
| **7B** | 全参数 | ❌ | ⚠️ 需4-8卡 | optimizer状态太大 |
| **14B+** | 任何 | ❌ | ❌ | 显存不足 |

---

## 二、复现目标与阶段规划

### 2.1 总体目标

```
Phase 0: 环境搭建 (1天)
    ↓
Phase 1: 数据准备 (0.5天)
    ↓
Phase 2: 基线复现 - GRPO (1-2天)
    ↓
Phase 3: DPPO 复现 (1-2天)
    ↓
Phase 4: 消融实验 (2-3天)
    ↓
Phase 5: 对比分析与总结 (1天)
```

**总预计时间**: 6-9天

### 2.2 阶段详细目标

| Phase | 目标 | 产出 | 预计时间 |
|-------|------|------|---------|
| **Phase 0** | 环境搭建，确保代码能跑通 | 可运行的环境 | 1天 |
| **Phase 1** | 准备 GSM8K/MATH 数据集 | 训练/测试数据 | 0.5天 |
| **Phase 2** | 复现 GRPO 基线 | GRPO 训练曲线 | 1-2天 |
| **Phase 3** | 复现 DPPO-Binary-TV | DPPO 训练曲线 | 1-2天 |
| **Phase 4** | 消融实验（不同超参/算法） | 对比结果 | 2-3天 |
| **Phase 5** | 分析总结 | 实验报告 | 1天 |

---

## 三、Phase 0: 环境搭建

### 3.1 创建 Conda 环境

```bash
# 创建环境
conda create -n stable-rl python=3.10 -y
conda activate stable-rl

# 安装 PyTorch (CUDA 12.1)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装 vLLM
pip install vllm==0.6.3

# 安装项目依赖
cd /data/home/yizhou/Stable-RL
pip install -e .

# 安装额外依赖
pip install wandb flash-attn --no-build-isolation
```

### 3.2 验证环境

```bash
# 验证 PyTorch + CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# 验证 vLLM
python -c "import vllm; print(vllm.__version__)"

# 验证 verl
python -c "import verl; print(verl.__version__)"
```

### 3.3 配置 WandB (可选)

```bash
wandb login
# 输入 API key
```

---

## 四、Phase 1: 数据准备

### 4.1 GSM8K 数据集

```bash
cd /data/home/yizhou/Stable-RL

# 预处理 GSM8K
python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

# 验证数据
ls -la ~/data/gsm8k/
# 应该看到 train.parquet 和 test.parquet
```

### 4.2 MATH 数据集 (可选)

```bash
# 预处理 MATH
python examples/data_preprocess/math_dataset.py --local_save_dir ~/data/math

# 验证数据
ls -la ~/data/math/
```

### 4.3 数据格式验证

```python
# 验证数据格式
import pandas as pd
df = pd.read_parquet("~/data/gsm8k/train.parquet")
print(df.columns)
print(df.iloc[0])
# 应该包含: data_source, prompt, ability, reward_model, extra_info
```

---

## 五、Phase 2: 基线复现 - GRPO

### 5.1 方案A: 1.5B GRPO-LoRA (推荐首选)

**单卡即可运行，最安全的选择**

```bash
cd /data/home/yizhou/Stable-RL

# 创建运行脚本
cat > run_grpo_1.5b_4090.sh << 'EOF'
#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=stable-rl-grpo-1.5b
export WANDB_EXP=grpo-lora-4090

MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct

# 4090 适配参数
nproc_per_gpu=16          # 从128降到16（4090显存小）
mini_batch_size=${nproc_per_gpu}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=${nproc_per_gpu} \
    data.val_batch_size=${nproc_per_gpu} \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXP} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    $@ 2>&1 | tee grpo_1.5b_4090.log
EOF

chmod +x run_grpo_1.5b_4090.sh
```

**运行**:
```bash
./run_grpo_1.5b_4090.sh
```

**预计时间**: 约 2-4 小时（GSM8K 7.5K 训练样本，1 epoch）

### 5.2 方案B: 3B GRPO-LoRA

```bash
cat > run_grpo_3b_4090.sh << 'EOF'
#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=stable-rl-grpo-3b
export WANDB_EXP=grpo-lora-3b-4090

MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
nproc_per_gpu=8
mini_batch_size=${nproc_per_gpu}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=${nproc_per_gpu} \
    data.val_batch_size=${nproc_per_gpu} \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    $@ 2>&1 | tee grpo_3b_4090.log
EOF
```

### 5.3 方案C: 多卡并行 GRPO (加速)

**利用8卡并行跑多个实验或加速单个实验**

```bash
# 方案C1: 8卡并行跑8个不同seed的1.5B实验
for seed in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$seed ./run_grpo_1.5b_4090.sh \
        trainer.experiment_name=grpo-seed-$seed \
        trainer.seed=$seed &
done
wait

# 方案C2: 2卡并行的3B实验
CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main_ppo \
    trainer.n_gpus_per_node=2 \
    ... # 其他参数同方案B
```

---

## 六、Phase 3: DPPO 复现

### 6.1 DPPO-Binary-TV 复现

**关键改动**: `policy_loss.loss_mode` 从 `ppo` 改为 `dppo_binary_tv`

```bash
cat > run_dppo_1.5b_4090.sh << 'EOF'
#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=stable-rl-dppo-1.5b
export WANDB_EXP=dppo-binary-tv-4090

MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
nproc_per_gpu=16
mini_batch_size=${nproc_per_gpu}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=${nproc_per_gpu} \
    data.val_batch_size=${nproc_per_gpu} \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    # ★★★ DPPO 核心配置 ★★★
    actor_rollout_ref.actor.policy_loss.loss_mode=dppo_binary_tv \
    actor_rollout_ref.actor.clip_ratio=0.1 \
    actor_rollout_ref.actor.clip_ratio_low=0.1 \
    actor_rollout_ref.actor.clip_ratio_high=0.1 \
    # ★★★ 配置结束 ★★★
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXP} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    $@ 2>&1 | tee dppo_1.5b_4090.log
EOF

chmod +x run_dppo_1.5b_4090.sh
```

### 6.2 DPPO-Binary-KL 复现

```bash
# 只需修改 loss_mode
actor_rollout_ref.actor.policy_loss.loss_mode=dppo_binary_kl \
```

### 6.3 关键配置说明

| 参数 | GRPO 值 | DPPO 值 | 说明 |
|------|---------|---------|------|
| `policy_loss.loss_mode` | `ppo` | `dppo_binary_tv` | 核心切换 |
| `clip_ratio` | 0.2 | 0.1 | 散度阈值 δ |
| `clip_ratio_low` | 0.2 | 0.1 | 负优势方向阈值 |
| `clip_ratio_high` | 0.2 | 0.1 | 正优势方向阈值 |

**注意**: DPPO 的 `clip_ratio` 对应论文中的散度阈值 δ，与 PPO 的 ε 含义不同但作用类似。建议从 0.1 开始调优。

---

## 七、Phase 4: 消融实验

### 7.1 实验矩阵

| 实验编号 | 算法 | 模型 | 目的 | 预计时间 |
|---------|------|------|------|---------|
| **Exp-1** | GRPO | 1.5B | 基线 | 2-4h |
| **Exp-2** | DPPO-Binary-TV | 1.5B | DPPO 效果 | 2-4h |
| **Exp-3** | DPPO-Binary-KL | 1.5B | KL vs TV | 2-4h |
| **Exp-4** | GRPO | 3B | 模型规模影响 | 4-6h |
| **Exp-5** | DPPO-Binary-TV | 3B | DPPO + 更大模型 | 4-6h |
| **Exp-6** | DPPO (δ=0.05) | 1.5B | 超参敏感性 | 2-4h |
| **Exp-7** | DPPO (δ=0.2) | 1.5B | 超参敏感性 | 2-4h |
| **Exp-8** | GRPO (高lr) | 1.5B | 学习率影响 | 2-4h |

### 7.2 并行实验策略

**8卡4090的高效利用**:

```bash
# 方案1: 8个实验并行（每个实验1卡）
for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i run_experiment_$i.sh &
done
wait

# 方案2: 4个实验并行（每个实验2卡，适合3B模型）
for i in 0 1 2 3; do
    GPU_START=$((i*2))
    CUDA_VISIBLE_DEVICES=$GPU_START,$((GPU_START+1)) run_experiment_$i.sh &
done
wait
```

### 7.3 超参搜索脚本

```bash
cat > run_hyperparam_search.sh << 'EOF'
#!/bin/bash

# 学习率搜索
for lr in 1e-5 3e-5 5e-5; do
    CUDA_VISIBLE_DEVICES=0 ./run_grpo_1.5b_4090.sh \
        actor_rollout_ref.actor.optim.lr=$lr \
        trainer.experiment_name=grpo-lr-$lr &
done

# DPPO 阈值搜索
for delta in 0.05 0.1 0.2 0.3; do
    CUDA_VISIBLE_DEVICES=1 ./run_dppo_1.5b_4090.sh \
        actor_rollout_ref.actor.clip_ratio=$delta \
        trainer.experiment_name=dppo-delta-$delta &
done

wait
EOF
```

---

## 八、Phase 5: 对比分析

### 8.1 评估指标

| 指标 | 含义 | 采集方式 |
|------|------|---------|
| **train_reward** | 训练奖励 | WandB |
| **val_reward** | 验证奖励 | WandB |
| **kl_divergence** | KL 散度 | WandB |
| **clip_fraction** | 裁剪比例 | WandB |
| **entropy** | 策略熵 | WandB |
| **response_length** | 响应长度 | WandB |

### 8.2 结果可视化

```python
import wandb
import pandas as pd
import matplotlib.pyplot as plt

# 获取实验数据
api = wandb.Api()
runs = api.runs("your-project/verl_grpo_example_gsm8k_math")

# 对比 GRPO vs DPPO
for run in runs:
    history = run.history()
    plt.plot(history['train_reward'], label=run.name)

plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()
plt.savefig('grpo_vs_dppo_comparison.png')
```

### 8.3 预期结果

**GSM8K 上的预期表现**:

| 算法 | 1.5B 最终准确率 | 3B 最终准确率 |
|------|----------------|--------------|
| GRPO | ~45-50% | ~55-60% |
| DPPO-Binary-TV | ~50-55% | ~60-65% |
| DPPO-Binary-KL | ~50-55% | ~60-65% |

**注意**: GSM8K 相对简单，DPPO 的优势在更难的任务（如 MATH、AIME）上更明显。

---

## 九、8卡并行实验方案

### 9.1 最优利用策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    8卡4090 并行实验布局                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPU 0: GRPO-1.5B (seed=0)        ─── 基线实验                 │
│  GPU 1: DPPO-TV-1.5B (seed=0)     ─── DPPO 实验               │
│  GPU 2: DPPO-KL-1.5B (seed=0)     ─── KL vs TV 对比           │
│  GPU 3: GRPO-1.5B (seed=1)        ─── 方差估计                 │
│  GPU 4: GRPO-3B (seed=0)          ─── 模型规模                 │
│  GPU 5: DPPO-TV-3B (seed=0)       ─── DPPO + 更大模型          │
│  GPU 6: DPPO-TV-1.5B (δ=0.05)    ─── 超参敏感性               │
│  GPU 7: DPPO-TV-1.5B (δ=0.2)     ─── 超参敏感性               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 并行运行脚本

```bash
cat > run_all_experiments.sh << 'EOF'
#!/bin/bash
set -x

# 确保数据已准备好
if [ ! -f "$HOME/data/gsm8k/train.parquet" ]; then
    echo "Please run data preparation first!"
    exit 1
fi

# 实验1: GRPO 1.5B (GPU 0)
CUDA_VISIBLE_DEVICES=0 ./run_grpo_1.5b_4090.sh \
    trainer.experiment_name=grpo-1.5b-seed0 \
    trainer.seed=0 &

# 实验2: DPPO-TV 1.5B (GPU 1)
CUDA_VISIBLE_DEVICES=1 ./run_dppo_1.5b_4090.sh \
    trainer.experiment_name=dppo-tv-1.5b-seed0 \
    trainer.seed=0 &

# 实验3: DPPO-KL 1.5B (GPU 2)
CUDA_VISIBLE_DEVICES=2 ./run_dppo_1.5b_4090.sh \
    actor_rollout_ref.actor.policy_loss.loss_mode=dppo_binary_kl \
    trainer.experiment_name=dppo-kl-1.5b-seed0 \
    trainer.seed=0 &

# 实验4: GRPO 1.5B (seed=1) (GPU 3)
CUDA_VISIBLE_DEVICES=3 ./run_grpo_1.5b_4090.sh \
    trainer.experiment_name=grpo-1.5b-seed1 \
    trainer.seed=1 &

# 实验5: GRPO 3B (GPU 4)
CUDA_VISIBLE_DEVICES=4 ./run_grpo_3b_4090.sh \
    trainer.experiment_name=grpo-3b-seed0 \
    trainer.seed=0 &

# 实验6: DPPO-TV 3B (GPU 5)
CUDA_VISIBLE_DEVICES=5 ./run_dppo_3b_4090.sh \
    trainer.experiment_name=dppo-tv-3b-seed0 \
    trainer.seed=0 &

# 实验7: DPPO-TV 1.5B δ=0.05 (GPU 6)
CUDA_VISIBLE_DEVICES=6 ./run_dppo_1.5b_4090.sh \
    actor_rollout_ref.actor.clip_ratio=0.05 \
    trainer.experiment_name=dppo-tv-1.5b-delta005 &

# 实验8: DPPO-TV 1.5B δ=0.2 (GPU 7)
CUDA_VISIBLE_DEVICES=7 ./run_dppo_1.5b_4090.sh \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    trainer.experiment_name=dppo-tv-1.5b-delta02 &

# 等待所有实验完成
wait

echo "All experiments completed!"
EOF

chmod +x run_all_experiments.sh
```

---

## 十、常见问题与解决方案

### 10.1 OOM (Out of Memory)

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 1. 降低 batch size
actor_rollout_ref.actor.ppo_micro_batch_size=2

# 2. 增加 offload
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# 3. 降低 vLLM 显存占比
actor_rollout_ref.rollout.gpu_memory_utilization=0.2

# 4. 减少采样数
actor_rollout_ref.rollout.n=3  # 从5降到3

# 5. 减小序列长度
data.max_prompt_length=256
data.max_response_length=512

# 6. 启用 activation offload (最后手段)
actor_rollout_ref.model.enable_activation_offload=True
```

### 10.2 训练速度慢

**可能原因**:
1. PCIe 互联带宽限制
2. CPU offload 开销
3. vLLM 推理瓶颈

**优化建议**:
```bash
# 1. 使用 use_remove_padding 减少无效计算
actor_rollout_ref.model.use_remove_padding=True

# 2. 调整 vLLM batch 参数
actor_rollout_ref.rollout.max_num_seqs=32
actor_rollout_ref.rollout.enable_chunked_prefill=True

# 3. 如果有多卡，使用 TP 加速推理
actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

### 10.3 数据加载问题

**症状**: `FileNotFoundError` 或数据格式错误

**解决方案**:
```bash
# 检查数据路径
ls -la ~/data/gsm8k/train.parquet

# 重新预处理数据
python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

### 10.4 WandB 日志问题

```bash
# 禁用 WandB，只用 console
trainer.logger='["console"]'

# 或者设置 offline 模式
export WANDB_MODE=offline
```

---

## 十一、学习检查清单

### Phase 0 完成标准
- [ ] 环境安装成功
- [ ] 能 `import verl` 不报错
- [ ] `torch.cuda.is_available()` 返回 True
- [ ] 能看到 8 张 GPU

### Phase 1 完成标准
- [ ] GSM8K 数据下载成功
- [ ] `train.parquet` 和 `test.parquet` 存在
- [ ] 数据格式验证通过

### Phase 2 完成标准
- [ ] GRPO 训练成功启动
- [ ] 能看到 reward 上升
- [ ] 训练完成，模型保存成功
- [ ] WandB 曲线正常

### Phase 3 完成标准
- [ ] DPPO 训练成功启动
- [ ] 能看到 `actor/pg_clipfraction` 指标
- [ ] 训练完成，与 GRPO 对比

### Phase 4 完成标准
- [ ] 至少完成 3 组消融实验
- [ ] 结果有可比性
- [ ] 能解释差异原因

### Phase 5 完成标准
- [ ] 绘制对比图表
- [ ] 总结 DPPO vs GRPO 的差异
- [ ] 记录遇到的问题和解决方案

---

## 十二、时间规划表

| 日期 | 任务 | 预计时间 | 产出 |
|------|------|---------|------|
| **Day 1** | Phase 0: 环境搭建 | 4-6h | 可运行环境 |
| **Day 1** | Phase 1: 数据准备 | 2h | 训练数据 |
| **Day 2** | Phase 2: GRPO 基线 | 4-6h | GRPO 训练曲线 |
| **Day 3** | Phase 3: DPPO 复现 | 4-6h | DPPO 训练曲线 |
| **Day 4-5** | Phase 4: 消融实验 | 8-12h | 对比结果 |
| **Day 6** | Phase 5: 分析总结 | 4-6h | 实验报告 |

**总计**: 约 6 天（如果并行实验可压缩到 4-5 天）

---

## 附录：关键文件路径

| 文件 | 路径 | 用途 |
|------|------|------|
| **主入口** | `verl/trainer/main_ppo.py` | 训练启动 |
| **核心算法** | `verl/trainer/ppo/core_algos.py` | DPPO 实现 |
| **配置文件** | `verl/trainer/config/ppo_trainer.yaml` | 默认配置 |
| **算法配置** | `verl/trainer/config/algorithm.py` | AlgoConfig 定义 |
| **数据预处理** | `examples/data_preprocess/gsm8k.py` | GSM8K 数据 |
| **训练脚本** | `examples/tuning/1.5b/` | 1.5B 配置参考 |
| **GRPO 示例** | `examples/grpo_trainer/run_qwen2-7b_math.sh` | GRPO 配置参考 |

---

*最后更新: 2026-06-26*
*基于 8x RTX 4090 (24GB) 硬件配置*
