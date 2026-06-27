# LLM RL 项目对照分析

> Stable-RL (DPPO) vs ProxMO-RL vs VIMPO vs slime
>
> 四个 LLM 后训练 RL 框架/算法的深度技术对比

---

## 目录

1. [项目概览与定位](#1-项目概览与定位)
2. [核心算法对比总览](#2-核心算法对比总览)
3. [Advantage Estimator 深度对比](#3-advantage-estimator-深度对比)
4. [Policy Loss 函数对比](#4-policy-loss-函数对比)
5. [训练稳定性技术对比](#5-训练稳定性技术对比)
6. [工程架构对比](#6-工程架构对比)
7. [应用场景与适用性](#7-应用场景与适用性)
8. [核心创新点详解](#8-核心创新点详解)
9. [代码实现位置索引](#9-代码实现位置索引)
10. [面试考察点与技术深挖](#10-面试考察点与技术深挖)

---

## 1. 项目概览与定位

### 1.1 四个项目一句话定位

| 项目 | 核心定位 | 一句话描述 |
|------|---------|-----------|
| **Stable-RL (DPPO)** | 信任域优化 | 用分布散度替代概率比做 PPO 裁剪，解决 LLM RL 训练不稳定 |
| **ProxMO-RL** | 多轮信用分配 | 面向 Agent 多轮交互的轻量级 credit assignment 框架 |
| **VIMPO** | 隐式价值函数 | 从 KL 正则化 RL 推导出隐式 token 级 value，无需 Critic |
| **slime** | 生产级框架 | 清华 THUDM 的大规模 RL 后训练框架，支持 GLM/Qwen/DeepSeek |

### 1.2 详细对比

| 维度 | Stable-RL | ProxMO-RL | VIMPO | slime |
|------|-----------|-----------|-------|-------|
| **机构** | Sail-SG (ByteDance) | 清华/腾讯 | UC Berkeley | 清华 THUDM |
| **论文** | arXiv 2602.04879 | arXiv 2602.19225 | arXiv 2606.20008 | 博客/开源 |
| **基础框架** | VeRL | VeRL | VeRL (fork) | Megatron + SGLang |
| **核心贡献** | DPPO 算法 | PSC + PSA 机制 | 隐式 token 级 value | 工程框架 + OPSM/TIS |
| **目标场景** | 数学推理 RL | 多轮 Agent RL | 数学推理 RLVR | 通用大规模 RL |
| **模型规模** | 1.5B ~ 30B | 1.5B ~ 7B | 4B | 0.5B ~ 671B |
| **代码量** | ~2000 行核心 | ~500 行核心 | ~1000 行核心 | ~3000 行核心 |

### 1.3 基础框架依赖

```
┌─────────────────────────────────────────────────────────────────────┐
│                        基础框架依赖图                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   VeRL (ByteDance) ─────────┬──────────── Stable-RL (DPPO)         │
│       │                     ├──────────── ProxMO-RL                 │
│       │                     └──────────── VIMPO (fork)              │
│       │                                                             │
│   Megatron-LM (NVIDIA) ─────┴──────────── slime                    │
│       + SGLang                                                      │
│                                                                     │
│   共同特性:                                                          │
│   - Ray 分布式调度                                                    │
│   - FSDP/Megatron 模型并行                                           │
│   - vLLM/SGLang 推理引擎                                             │
│   - Hydra 配置系统                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心算法对比总览

### 2.1 Advantage Estimator 支持矩阵

| 优势估计器 | Stable-RL | ProxMO-RL | VIMPO | slime |
|-----------|:---------:|:---------:|:-----:|:-----:|
| **GAE** (需要 Critic) | ✅ | ✅ | ✅ | ✅ |
| **GRPO** | ✅ | ✅ | ✅ | ✅ |
| **GRPO Pass@k** | ✅ | ✅ | ✅ | ❌ |
| **GRPO Vectorized** | ✅ | ❌ | ✅ | ❌ |
| **RLOO** | ✅ | ✅ | ✅ | ❌ |
| **RLOO Vectorized** | ✅ | ❌ | ✅ | ❌ |
| **REINFORCE++** | ✅ | ✅ | ✅ | ✅ |
| **REINFORCE++ Baseline** | ✅ | ✅ | ✅ | ✅ |
| **ReMax** | ✅ | ✅ | ✅ | ❌ |
| **OPO** | ✅ | ❌ | ✅ | ❌ |
| **GPG** | ✅ | ❌ | ✅ | ❌ |
| **GDPO** | ❌ | ❌ | ✅ | ❌ |
| **OTB** (Optimal Token Baseline) | ✅ | ❌ | ✅ | ❌ |
| **ProxMO** (本项目创新) | ❌ | ✅ | ❌ | ❌ |
| **VIMPO** (本项目创新) | ❌ | ❌ | ✅ | ❌ |
| **GSPO** | ✅ | ❌ | ✅ | ✅ |

### 2.2 Policy Loss 支持矩阵

| Policy Loss | Stable-RL | ProxMO-RL | VIMPO | slime |
|-------------|:---------:|:---------:|:-----:|:-----:|
| **PPO (Vanilla)** | ✅ | ✅ | ✅ | ✅ |
| **PPO Dual-Clip** | ✅ | ✅ | ✅ | ✅ |
| **DPPO Binary-TV** | ✅ (核心) | ❌ | ✅ | ❌ |
| **DPPO Binary-KL** | ✅ (核心) | ❌ | ✅ | ❌ |
| **DPPO TopK-TV/KL** | ✅ | ❌ | ❌ | ❌ |
| **GSPO** | ✅ | ❌ | ✅ | ✅ |
| **SAPO** | ✅ | ❌ | ✅ | ❌ |
| **CISPO** | ✅ | ❌ | ✅ | ✅ |
| **GPG** | ✅ | ❌ | ✅ | ❌ |
| **Clip-COV** | ✅ | ❌ | ✅ | ❌ |
| **KL-COV** | ✅ | ❌ | ✅ | ❌ |
| **Geo-Mean (GMPO)** | ✅ | ❌ | ✅ | ❌ |
| **VIMPO Loss** | ❌ | ❌ | ✅ (核心) | ❌ |
| **Bypass Mode** | ✅ | ❌ | ✅ | ❌ |

### 2.3 算法创新点对比

| 项目 | 核心创新 | 解决的问题 | 技术手段 |
|------|---------|-----------|---------|
| **Stable-RL** | DPPO | PPO ratio clipping 对低概率 token 过度惩罚 | Binary/TopK 散度近似 |
| **ProxMO-RL** | PSC + PSA | 多轮 Agent 的 credit assignment | 成功率感知调制 + TF-IDF 软聚合 |
| **VIMPO** | 隐式 Value | 需要 Critic 的 token 级 credit assignment | KL 正则化 RL 的最优条件推导 |
| **slime** | OPSM + TIS | Off-policy 数据的负面影响 | 序列级 KL mask + 截断 IS |

---

## 3. Advantage Estimator 深度对比

### 3.1 GRPO 实现对比

**公式**: `Â = R(y) - mean(R(y_i))` (可选除以 std)

| 实现细节 | Stable-RL | ProxMO-RL | VIMPO | slime |
|---------|-----------|-----------|-------|-------|
| **分组方式** | `index` 数组 | `index` + `traj_index` | `uid` | 隐式 (按 prompt) |
| **std 归一化** | 可选 (`norm_adv_by_std_in_grpo`) | 可选 (`remove_std`) | 可选 | 可选 (`--disable-grpo-std-normalization`) |
| **Dr.GRPO 支持** | ✅ | ✅ | ✅ | ✅ |
| **向量化实现** | ✅ (`grpo_vectorized`) | ❌ | ✅ | ❌ |

**Stable-RL 实现** (`core_algos.py:267-331`):
```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, ...):
    scores = token_level_rewards.sum(dim=-1)
    # 按 prompt index 分组
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    # 组内归一化
    if norm_adv_by_std_in_grpo:
        scores[i] = (scores[i] - mean) / (std + epsilon)
    else:
        scores[i] = scores[i] - mean  # Dr.GRPO
```

**slime 实现** (`ppo_utils.py:244-251`):
```python
def get_grpo_returns(rewards, kl):
    # 简单广播: 每个 token 获得相同的 group-normalized reward
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```

### 3.2 REINFORCE++ 实现对比

**公式**: `G_t = r_t + γ * G_{t+1}` (折扣回报)

| 实现细节 | Stable-RL | ProxMO-RL | VIMPO | slime |
|---------|-----------|-----------|-------|-------|
| **折扣因子** | `gamma` (默认 1.0) | `gamma` (默认 0.95) | `gamma` | `gamma` |
| **KL 惩罚** | 可选 | 可选 | 内置 (token KL) | `kl_coef` |
| **CP 支持** | ❌ | ❌ | ❌ | ✅ (Context Parallelism) |

**slime 实现** (`ppo_utils.py:254-321`):
```python
def get_reinforce_plus_plus_returns(rewards, kl, loss_masks, ...):
    # Token-level rewards = -kl_coef * KL + terminal reward
    token_level_rewards = -kl_coef * masked_kl
    token_level_rewards[last_idx] += rewards[i]
    
    # 折扣回报
    for t in reversed(range(len)):
        running_return = token_level_rewards[t] + gamma * running_return
        returns_for_seq[t] = running_return
```

### 3.3 独特的 Advantage Estimator

#### ProxMO: 联合优势估计

**公式**: `A = A_episode + ω * A_step`

**核心创新**:
1. **Episode 级**: PSC (Polarized Signal Controller) 调制
2. **Step 级**: PSA (Proximity-Based Soft Aggregation) 基于 TF-IDF 相似度

```python
# ProxMO 核心 (core_proxmo.py:133-182)
def compute_proxmo_outcome_advantage(...):
    # Episode 级优势 (含 PSC)
    episode_advantages = episode_norm_reward(..., enable_psc=True)
    
    # Step 级优势 (软聚合)
    step_advantages = step_soft_reward(...)  # TF-IDF + softmax
    
    # 联合优势
    scores = episode_advantages + step_advantage_w * step_advantages
```

#### VIMPO: 隐式 Token 级 Value

**公式**: `V(s_t) = β * (log π_θ/π_ref - KL)` (从 KL 正则化 RL 推导)

**核心创新**:
1. 无需训练 Critic 网络
2. 从最优条件推导 token 级 value
3. Terminal MSE loss 训练

```python
# VIMPO 核心 (vimpo_core_algos.py:27-37)
def build_vimpo_token_term(log_prob, ref_log_prob, token_kl, response_mask, beta):
    # ρ_t - κ_t = β * (log π_θ/π_ref - KL_t)
    return beta * (log_prob - ref_log_prob - token_kl) * response_mask

# Terminal MSE Loss (vimpo_core_algos.py:423-501)
def compute_vimpo_terminal_mse_loss(vimpo_token_term, token_level_rewards, ...):
    terminal_value = sum_t (ρ_t - κ_t)  # 隐式 value
    terminal_target = R_final - mean(R_final)  # 组内归一化奖励
    loss = 0.5 * (terminal_value - terminal_target)^2
```

---

## 4. Policy Loss 函数对比

### 4.1 PPO 实现对比

**公式**: `L = -E[min(r*A, clip(r, 1-ε, 1+ε)*A)]`

| 实现细节 | Stable-RL | ProxMO-RL | VIMPO | slime |
|---------|-----------|-----------|-------|-------|
| **裁剪范围** | `clip_ratio_low/high` | `clip_ratio_low/high` | `clip_ratio_low/high` | `eps_clip/eps_clip_high` |
| **Dual-Clip** | ✅ (`clip_ratio_c`) | ✅ | ✅ | ✅ (`eps_clip_c`) |
| **Ratio 定义** | `π_θ/π_old` 或 `π_θ/π_rollout` | `π_θ/π_old` | `π_θ/π_old` | `exp(log_π_θ - log_π_old)` |
| **@torch.compile** | ❌ | ❌ | ❌ | ✅ |

**slime 实现** (`ppo_utils.py:124-148`):
```python
@torch.compile(dynamic=True)
def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
    ratio = (-ppo_kl).exp()  # ppo_kl = log_old - log_new
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    
    if eps_clip_c is not None:  # Dual-Clip
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

### 4.2 DPPO 实现对比 (Stable-RL 独有)

**公式**: `L = -E[M_t * r_t * Â_t]`，其中 `M_t` 基于散度阈值

| 变体 | 散度计算 | 代码位置 |
|------|---------|----------|
| **DPPO Binary-TV** | `\|π_rollout(a_t) - π_θ(a_t)\|` | `core_algos.py:1275-1277` |
| **DPPO Binary-KL** | `π_rollout*log(π_rollout/π_θ) + (1-π_rollout)*log(...)` | `core_algos.py:1264-1267` |
| **DPPO TopK-TV** | Top-K tokens 的 TV 散度 | `core_algos.py:1281-1283` |
| **DPPO TopK-KL** | Top-K tokens 的 KL 散度 | `core_algos.py:1272-1274` |

```python
# DPPO Binary-TV 核心实现 (core_algos.py:1275-1277)
if loss_mode == "dppo_binary_tv":
    invalid_positive_mask = (prob - rollout_prob) > cliprange_high
    invalid_negative_mask = (prob - rollout_prob) < -cliprange_low
```

### 4.3 CISPO 实现对比

**公式**: `-sg(clip(ratio)) * A * log π_θ`

| 实现 | 位置 | 特点 |
|------|------|------|
| **Stable-RL** | `core_algos.py:1889-1947` | 标准实现 |
| **VIMPO** | `core_algos.py` (同上) | 继承自 verl |
| **slime** | `ppo_utils.py:151-171` | `@torch.compile` 优化 |

**slime 实现**:
```python
@torch.compile(dynamic=True)
def compute_cispo_loss(ppo_kl, log_probs, advantages, eps_clip, eps_clip_high):
    ratio = (-ppo_kl).exp()
    ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
    pg_losses = -ratio_truncated.detach() * advantages * log_probs  # stop-gradient
    return pg_losses, (ratio_truncated != ratio).float()
```

### 4.4 VIMPO Loss 实现 (VIMPO 独有)

**公式**: `L_VIMPO = L_V + c_A * L_A`

- `L_V`: Terminal MSE loss (隐式 value 预测)
- `L_A`: 可选的 PPO/GRPO actor loss
- `c_A`: 自适应 actor coefficient

```python
# VIMPO 完整 loss (vimpo_core_algos.py:504-568)
def compute_vimpo_loss(log_prob, ref_log_prob, token_kl, token_level_rewards, ...):
    # 1. Token KL 聚合
    vimpo_kl_agg = agg_loss(token_kl, response_mask, ...)
    
    # 2. 构建 token term: β * (log π/π_ref - KL)
    vimpo_token_term = build_vimpo_token_term(log_prob, ref_log_prob, token_kl, ..., beta)
    
    # 3. Terminal MSE loss
    vimpo_terminal_mse = compute_vimpo_terminal_mse_loss(vimpo_token_term, ...)
    
    return vimpo_terminal_mse, vimpo_kl_agg, metrics
```

### 4.5 OPSM 实现 (slime 独有)

**公式**: `M_t = 0 if (Â < 0 and seq_KL > δ)`

```python
# OPSM 核心 (ppo_utils.py:54-92)
def compute_opsm_mask(args, full_log_probs, full_old_log_probs, advantages, loss_masks):
    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(...):
        # 序列级 KL
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / loss_mask.sum()
        
        # Mask: 负优势且 KL 大的序列被 mask
        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        opsm_mask_list.append(1 - mask)
```

---

## 5. 训练稳定性技术对比

### 5.1 问题诊断

| 问题 | Stable-RL 诊断 | ProxMO-RL 诊断 | VIMPO 诊断 | slime 诊断 |
|------|---------------|---------------|-----------|-----------|
| **Training-Inference Mismatch** | 核心问题 | 次要问题 | 次要问题 | 通过 OPSM/TIS 处理 |
| **低概率 Token 过度惩罚** | 核心问题 | 未涉及 | 通过隐式 value 缓解 | 未涉及 |
| **多轮 Credit Assignment** | 未涉及 | 核心问题 | 未涉及 | 未涉及 |
| **Critic 训练不稳定** | 使用 Critic-free 方法 | 使用 Critic-free 方法 | 隐式 Critic | 可选 |
| **Off-policy 数据偏差** | Rollout Correction | 未涉及 | 未涉及 | OPSM + TIS |

### 5.2 解决方案对比

| 技术 | Stable-RL | ProxMO-RL | VIMPO | slime |
|------|-----------|-----------|-------|-------|
| **信任域约束** | DPPO (散度) | PPO (ratio) | PPO (ratio) | PPO (ratio) + OPSM |
| **Off-policy 修正** | IS + RS | ❌ | ❌ | TIS (截断 IS) |
| **序列级 Mask** | ❌ | ❌ | ❌ | OPSM |
| **Token 级 Credit** | DPPO mask | PSA (step 级) | 隐式 value | 标准 PPO |
| **成功率感知** | ❌ | PSC | ❌ | ❌ |
| **相似度聚合** | ❌ | TF-IDF 软聚合 | ❌ | ❌ |

### 5.3 关键 Takeaway 对比

**Stable-RL 的三个 Takeaway**:
1. 信任域即使在低学习率下也必要
2. 信任域必须锚定到 rollout 策略
3. 0.5% 的 bad updates 导致崩溃

**ProxMO-RL 的发现**:
1. GRPO 的 z-score 存在信息不对称
2. 硬匹配导致 30-36% singleton groups
3. 软聚合 + PSC 显著提升 Agent 性能

**VIMPO 的发现**:
1. KL 正则化 RL 有隐式最优 value
2. 无需 Critic 即可获得 token 级 credit
3. 隐式 value 优化避免 entropy collapse

**slime 的发现**:
1. OPSM 可有效过滤 off-policy 数据
2. TIS 截断 IS ratio 控制方差
3. Chunked GAE 提升并行效率

---

## 6. 工程架构对比

### 6.1 训练框架

| 特性 | Stable-RL | ProxMO-RL | VIMPO | slime |
|------|-----------|-----------|-------|-------|
| **分布式框架** | Ray + FSDP/Megatron | Ray + FSDP | Ray + FSDP/Megatron | Ray + Megatron |
| **推理引擎** | vLLM/SGLang | vLLM | vLLM | SGLang (深度整合) |
| **配置系统** | Hydra YAML | Hydra YAML | Hydra YAML | argparse |
| **入口文件** | `main_ppo.py` | `main_ppo.py` | `main_vimpo.py` | `train.py` / `train_async.py` |
| **异步训练** | ❌ | ❌ | ❌ | ✅ (`train_async.py`) |

### 6.2 代码组织

```
Stable-RL (VeRL fork):
├── verl/trainer/ppo/core_algos.py      # 核心算法 (~2400 行)
├── verl/trainer/config/algorithm.py    # 配置定义
└── examples/ppo_trainer/               # 训练脚本

ProxMO-RL (VeRL fork):
├── proxmo/core_proxmo.py               # ProxMO 核心 (~500 行)
├── verl/trainer/ppo/core_algos.py      # 基础算法
├── verl/trainer/ppo/ray_trainer.py     # 训练器 (含 ProxMO 集成)
└── agent_system/                       # 多轮交互系统

VIMPO (VeRL fork):
├── recipe/vimpo/vimpo_core_algos.py    # VIMPO 核心 (~1000 行)
├── recipe/vimpo/vimpo_ray_trainer.py   # VIMPO 训练器
├── verl/trainer/ppo/core_algos.py      # 基础算法
└── verl/workers/actor/dp_actor.py      # Actor (含 VIMPO 集成)

slime (独立框架):
├── slime/utils/ppo_utils.py            # RL 算法核心 (~730 行)
├── slime/backends/megatron_utils/      # Megatron 训练后端
├── slime/backends/sglang_utils/        # SGLang 推理后端
├── slime/rollout/                      # Rollout 管理
└── train.py / train_async.py           # 训练入口
```

### 6.3 性能优化

| 优化技术 | Stable-RL | ProxMO-RL | VIMPO | slime |
|---------|-----------|-----------|-------|-------|
| **@torch.compile** | ❌ | ❌ | ❌ | ✅ |
| **Chunked GAE** | ❌ | ❌ | ❌ | ✅ |
| **Sequence Balancing** | ✅ | ✅ | ✅ | ✅ |
| **Gradient Checkpointing** | ✅ | ✅ | ✅ | ✅ |
| **Delta Weight Sync** | ❌ | ❌ | ❌ | ✅ |
| **Context Parallelism** | ❌ | ❌ | ❌ | ✅ |
| **向量化 GRPO/RLOO** | ✅ | ❌ | ✅ | ❌ |

### 6.4 支持的模型

| 模型系列 | Stable-RL | ProxMO-RL | VIMPO | slime |
|---------|-----------|-----------|-------|-------|
| **Qwen** | ✅ (2/2.5/3) | ✅ (2.5) | ✅ (3) | ✅ (2.5/3/3MoE) |
| **DeepSeek** | ✅ (V3/R1) | ❌ | ❌ | ✅ (V3/R1) |
| **Llama** | ✅ (2/3) | ❌ | ❌ | ✅ (3) |
| **GLM** | ❌ | ❌ | ❌ | ✅ (4.5-5.2) |
| **Gemma** | ✅ | ❌ | ❌ | ❌ |

---

## 7. 应用场景与适用性

### 7.1 场景匹配

| 场景 | 最佳选择 | 次优选择 | 原因 |
|------|---------|---------|------|
| **数学推理 (单轮)** | VIMPO | Stable-RL | 隐式 value 提供精细 credit assignment |
| **数学推理 (大规模)** | slime | Stable-RL | 工程成熟度高，支持 671B 模型 |
| **多轮 Agent** | ProxMO-RL | slime | PSC + PSA 专门设计用于多轮 |
| **RLHF 对齐** | Stable-RL | slime | DPPO 对 reward model 更鲁棒 |
| **代码生成** | slime | Stable-RL | 支持 sandbox 和 code execution |
| **搜索 Agent** | ProxMO-RL | slime | 内置搜索环境支持 |
| **MoE 模型** | slime | Stable-RL | Routing Replay + DPPO |

### 7.2 计算资源需求

| 配置 | Stable-RL | ProxMO-RL | VIMPO | slime |
|------|-----------|-----------|-------|-------|
| **最小 GPU** | 2x A100 | 2x A100 | 2x A100 | 8x A100 |
| **推荐配置** | 8x A100 | 8x A100 | 8x A100 | 64x A100 |
| **内存优化** | LoRA/FSDP | FSDP | FSDP | Megatron PP/TP |
| **额外开销** | ~0% (DPPO) | +1.09% (ProxMO) | ~5% (VIMPO loss) | ~0% |

### 7.3 超参数调优建议

| 超参数 | Stable-RL | ProxMO-RL | VIMPO | slime |
|--------|-----------|-----------|-------|-------|
| **学习率** | 1e-6 ~ 5e-6 | 1e-6 | 1e-6 | 1e-6 |
| **Clip ratio** | 0.2 (DPPO: δ=0.1-0.3) | 0.2 | 0.2 | 0.2 |
| **组大小 N** | 8-64 | 8 | 8 | 8-64 |
| **折扣因子 γ** | 1.0 | 0.95 | 1.0 | 1.0 |
| **KL 系数** | 0.001 | ❌ | β=5e-4 | 可选 |
| **特殊参数** | `dppo_threshold` | `step_advantage_w=1.0`, `psc_alpha=4.0` | `vimpo_actor_coeff=5e-3` | `opsm_delta=1e-4` |

---

## 8. 核心创新点详解

### 8.1 Stable-RL: DPPO 信任域

**问题**: PPO 的 ratio clipping 是 TV 散度的单样本估计，对低概率 token 过度惩罚。

**解决方案**:
```
PPO:  D ≈ |r_t - 1|  (单样本，噪声大)
DPPO: D = |π_rollout(a_t) - π_θ(a_t)|  (Binary TV，稳定)
```

**关键设计**:
1. 不对称 mask: 只阻断 "远离" 方向的更新
2. 锚定 rollout 策略: 不使用重算策略
3. Binary 近似: O(1) 计算，下界保证

### 8.2 ProxMO-RL: PSC + PSA

**问题 1**: GRPO 的 z-score 在不同成功率下信息不对称。
```
当 p→1: 罕见失败被过度惩罚 (|z_fail|→∞)
当 p→0: 常见失败惩罚不足 (|z_fail|→0)
```

**PSC 解决方案**:
```python
w_success = 1 + β * (sigmoid((1-p)^α) - 0.5)    # 低成功率时放大
w_failure = 1 + β * (sigmoid(-p^α) - 0.5)        # 高成功率时衰减
```

**问题 2**: 硬匹配导致 30-36% singleton groups (advantage=0)。

**PSA 解决方案**:
```python
# TF-IDF 向量化
tfidf_matrix = TfidfVectorizer().fit_transform(observations)
# 余弦相似度矩阵
sim_matrix = tfidf_matrix @ tfidf_matrix.T
# Softmax 软聚合
weights = softmax(sim_matrix / temperature)
baseline = weights @ step_rewards
```

### 8.3 VIMPO: 隐式 Token 级 Value

**理论推导**:
```
KL 正则化 RL 目标: max E[R - β*KL(π||π_ref)]
最优条件推导: V*(s_t) = β * (log π_θ/π_ref - KL_t)
```

**Terminal MSE Loss**:
```python
terminal_value = sum_t β * (log π_θ/π_ref - KL_t)  # 隐式 value
terminal_target = R_final - mean(R_final)             # 组内归一化
loss = 0.5 * (terminal_value - terminal_target)^2
```

**优势**:
1. 无需训练 Critic 网络 (节省内存)
2. Token 级 credit assignment (比 GRPO 精细)
3. 避免 entropy collapse (比 GRPO 灵活)

### 8.4 slime: OPSM + TIS

**OPSM (Off-Policy Sequence Masking)**:
```python
seq_kl = mean(log π_old - log π_θ)  # 序列级 KL
if advantage < 0 and seq_kl > delta:
    mask = 0  # 整个序列被 mask
```

**TIS (Truncated Importance Sampling)**:
```python
ratio = π_θ / π_old
if ratio > threshold or ratio < 1/threshold:
    ratio = 0  # 截断
```

**ICEPOP 变体**:
```python
if ratio > threshold:
    ratio = 0  # 超出阈值直接置零
```

---

## 9. 代码实现位置索引

### 9.1 Advantage Estimator 位置

| 算法 | Stable-RL | ProxMO-RL | VIMPO | slime |
|------|-----------|-----------|-------|-------|
| **GRPO** | `verl/trainer/ppo/core_algos.py:267` | 同左 | 同左 | `slime/utils/ppo_utils.py:244` |
| **REINFORCE++** | `core_algos.py:583` | 同左 | 同左 | `ppo_utils.py:254` |
| **RLOO** | `core_algos.py:477` | 同左 | 同左 | ❌ |
| **GAE** | `core_algos.py:215` | 同左 | 同左 | `ppo_utils.py:354` |
| **ProxMO** | ❌ | `proxmo/core_proxmo.py:133` | ❌ | ❌ |
| **VIMPO** | ❌ | ❌ | `recipe/vimpo/vimpo_core_algos.py:27` | ❌ |

### 9.2 Policy Loss 位置

| 算法 | Stable-RL | ProxMO-RL | VIMPO | slime |
|------|-----------|-----------|-------|-------|
| **PPO** | `core_algos.py:1325` | 同左 | 同左 | `ppo_utils.py:124` |
| **DPPO** | `core_algos.py:1241` | ❌ | 继承 | ❌ |
| **CISPO** | `core_algos.py:1889` | 同左 | 同左 | `ppo_utils.py:151` |
| **GSPO** | `core_algos.py:1421` | 同左 | 同左 | `ppo_utils.py:95` |
| **VIMPO** | ❌ | ❌ | `vimpo_core_algos.py:504` | ❌ |
| **OPSM** | ❌ | ❌ | ❌ | `ppo_utils.py:54` |

### 9.3 配置文件位置

| 项目 | 主配置 | 算法配置 | 训练脚本 |
|------|--------|---------|---------|
| **Stable-RL** | `verl/trainer/config/ppo_trainer.yaml` | `verl/trainer/config/algorithm.py` | `examples/ppo_trainer/*.sh` |
| **ProxMO-RL** | `verl/trainer/config/ppo_trainer.yaml` | `proxmo/core_proxmo.py` | `examples/proxmo_trainer/*.sh` |
| **VIMPO** | `recipe/vimpo/config/vimpo_trainer.yaml` | `recipe/vimpo/vimpo_core_algos.py` | `recipe/vimpo/run_vimpo.sh` |
| **slime** | `slime/utils/arguments.py` | 同左 | `scripts/run-*.sh` |

---

## 10. 面试考察点与技术深挖

### 10.1 基础概念考察

**Q1: GRPO 和 PPO 的核心区别是什么？**

| 方面 | PPO | GRPO |
|------|-----|------|
| **Critic** | 需要训练 value network | 无需 Critic |
| **优势估计** | GAE (token 级) | 组内对比 (trajectory 级) |
| **内存开销** | 高 (actor + critic) | 低 (只有 actor) |
| **Credit Assignment** | 精细 (per-token) | 粗糙 (per-trajectory) |

**Q2: 为什么需要信任域？**

没有信任域 → 策略更新过大 → 训练-推理 mismatch 累积 → 性能崩溃

四种解决方案:
1. **Stable-RL**: 散度约束 (DPPO)
2. **ProxMO-RL**: 标准 PPO ratio (依赖低学习率)
3. **VIMPO**: 隐式 value 约束
4. **slime**: OPSM 序列级 mask

### 10.2 进阶技术考察

**Q3: Binary 近似为什么是 TV 散度的下界？**

```
TV = (1/2) Σ_a |P(a) - Q(a)|
Binary TV = |P(a_t) - Q(a_t)|  (只看采样 token)

由三角不等式: |Σ_{a∈C} (P(a)-Q(a))| ≤ Σ_{a∈C} |P(a)-Q(a)|
因此: Binary TV ≤ TV
```

**Q4: VIMPO 的隐式 value 和 PPO 的 Critic 有什么区别？**

| 方面 | PPO Critic | VIMPO 隐式 Value |
|------|-----------|-----------------|
| **参数** | 额外网络参数 | 无额外参数 |
| **训练** | 需要 value loss | 通过 terminal MSE |
| **Credit** | 学习得到 | 从最优条件推导 |
| **稳定性** | 可能不稳定 | 理论保证 |

**Q5: ProxMO 的 PSC 如何解决信息不对称？**

```
二元奖励 R∈{0,1}，成功率 p:
- z_succ = sqrt((1-p)/p)     # 成功时的 z-score
- z_fail = -sqrt(p/(1-p))    # 失败时的 z-score

问题: z_succ * z_succ(1-p) = 1 (互为倒数)
- p→1 时: z_fail→-∞ (罕见失败被过度惩罚)
- p→0 时: z_fail→0 (常见失败惩罚不足)

PSC 修正:
- 低成功率: 放大成功信号 (w≈1.05)
- 高成功率: 衰减失败噪声 (w≈0.95)
```

### 10.3 工程实践考察

**Q6: 如何选择合适的 RL 框架？**

```
需求分析:
├── 单轮数学推理 (4B-30B)
│   ├── 需要精细 credit → VIMPO
│   └── 需要训练稳定 → Stable-RL (DPPO)
├── 多轮 Agent
│   └── 必须 → ProxMO-RL
├── 大规模训练 (>100B)
│   └── 必须 → slime (Megatron)
└── MoE 模型
    ├── 有 R3 → Stable-RL
    └── 无 R3 → slime (Routing Replay)
```

**Q7: 如何调试训练不稳定？**

| 症状 | 可能原因 | 检查方法 | 解决方案 |
|------|---------|---------|---------|
| Reward 骤降 | Bad updates | 监控 clip fraction | 减小 clip ratio / 用 DPPO |
| KL 爆炸 | 学习率过大 | 监控 KL divergence | 降低 lr / 加 KL 惩罚 |
| Entropy collapse | 过度利用 | 监控 entropy | 增大 clip_ratio_high / 用 VIMPO |
| Advantage=0 | Singleton groups | 统计 group size | 用 ProxMO PSA |

**Q8: 如何处理 off-policy 数据？**

| 方法 | 来源 | 机制 | 适用场景 |
|------|------|------|---------|
| **DPPO** | Stable-RL | 散度 mask | 通用 |
| **IS + RS** | Stable-RL | 重要性采样 + 拒绝采样 | Agent/长序列 |
| **OPSM** | slime | 序列级 KL mask | 大规模训练 |
| **TIS** | slime | 截断重要性采样 | 通用 |
| **PSC** | ProxMO-RL | 成功率调制 | Agent |

### 10.4 深挖追问

**Q9: 为什么 DPPO 锚定 rollout 策略而非重算策略？**

```
MiniRL (重算): r = π_θ / π_old (重算)
DPPO (rollout): r = π_θ / π_rollout (实际生成)

问题: π_old 和 π_rollout 有数值差异 (BF16 vs FP32)
- 重算会累积这个差异
- 每步误差 ε，T 步后 O(Tε)
- 导致 mismatch 增长 → 崩溃

实验证据:
- DPPO (rollout): mismatch 保持 ~0.01
- DPPO (recompute): mismatch 增长到 0.1+ → 崩溃
```

**Q10: VIMPO 的 terminal value 和 GRPO 的 group normalization 有什么联系？**

```
GRPO: Â = R - mean(R)  (直接归一化)
VIMPO: V = Σ_t β*(log π/π_ref - KL)  (隐式 value)

联系:
- 当 β→0 时，VIMPO 退化为 GRPO
- VIMPO 的 target 是 R - mean(R) (同 GRPO)
- 但 VIMPO 通过 token 级项提供更精细的梯度信号

区别:
- GRPO: 所有 token 共享相同 advantage
- VIMPO: 每个 token 有不同的隐式 value 贡献
```

---

## 附录 A: 快速参考卡

### A.1 算法选择决策树

```
你的任务是什么？
├── 单轮数学推理
│   ├── 模型 < 30B → VIMPO (精细 credit) 或 Stable-RL (DPPO)
│   └── 模型 ≥ 30B → slime (工程成熟)
├── 多轮 Agent
│   └── 必须 → ProxMO-RL
├── RLHF 对齐
│   └── Stable-RL (DPPO) 或 slime
└── 代码生成
    └── slime (sandbox 支持)
```

### A.2 关键超参数速查

| 参数 | Stable-RL | ProxMO-RL | VIMPO | slime |
|------|-----------|-----------|-------|-------|
| `lr` | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| `clip_ratio` | 0.2 | 0.2 | 0.2 | 0.2 |
| `dppo_threshold` | 0.1-0.3 | ❌ | ❌ | ❌ |
| `step_advantage_w` | ❌ | 1.0 | ❌ | ❌ |
| `psc_alpha` | ❌ | 4.0 | ❌ | ❌ |
| `vimpo_beta` | ❌ | ❌ | 5e-4 | ❌ |
| `vimpo_actor_coeff` | ❌ | ❌ | 5e-3 | ❌ |
| `opsm_delta` | ❌ | ❌ | ❌ | 1e-4 |
| `tis_clip` | ❌ | ❌ | ❌ | 2.0 |

---

## 附录 B: 相关论文

| 论文 | 链接 | 相关项目 |
|------|------|---------|
| PPO | https://arxiv.org/abs/1707.06347 | 全部 |
| TRPO | https://arxiv.org/abs/1502.05477 | Stable-RL |
| GRPO (DeepSeekMath) | https://arxiv.org/abs/2402.03300 | 全部 |
| DAPO | https://arxiv.org/abs/2503.14476 | slime |
| CISPO (MiniMax-M1) | https://arxiv.org/abs/2506.13585 | slime |
| Dr.GRPO | https://arxiv.org/abs/2503.20783 | slime |
| REINFORCE++ | https://arxiv.org/abs/2501.03262 | 全部 |
| Dual-Clip PPO | https://arxiv.org/abs/1912.09729 | 全部 |

---

*最后更新: 2026-06-26*
*基于四个项目的最新代码分析*
