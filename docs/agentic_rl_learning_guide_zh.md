# Search-R1 Agentic RL 中文学习指南

这份指南面向会 Python、接触过 PyTorch，但强化学习和分布式训练经验较少的读者。目标不是立刻复现论文曲线，而是先建立一个可以反复验证的心智模型：**数据怎样变成 prompt，模型怎样把文本变成工具动作，环境怎样返回 observation，奖励怎样变成 advantage，advantage 最后怎样更新模型参数。**

建议用 5～7 周完成。每一阶段都包含阅读目标、代码入口、动手任务和验收问题。不要在第一次阅读时进入 `verl/third_party/`、Megatron 或 FSDP 的内部实现。

## 1. 学完后应该具备什么能力

完成主线后，你应该能够：

1. 从 `train_grpo.sh` 出发，追踪一个 Hydra 参数最终在哪个 Python 函数中生效。
2. 画出一条包含两次搜索的轨迹，区分 prompt、模型 action 和环境 observation。
3. 解释 `attention_mask`、`info_mask`、`loss_mask` 分别控制什么。
4. 解释为什么 GRPO 不需要 critic，以及同一问题为什么要采样多条轨迹。
5. 从最终 EM reward 推导到 token-level advantage，再推导到 PPO clipped loss。
6. 修改奖励或工具协议，并用小测试证明分组、mask 和 reward placement 没有被破坏。

## 2. 先建立项目心智模型

Search-R1 可以分成四层：

```text
实验层       shell scripts + Hydra config
                 |
训练编排层   RayPPOTrainer + Ray workers
                 |
Agent 层     LLMGenerationManager <-> Search HTTP server
                 |
算法与模型层 PPO/GRPO math + Actor/Critic + FSDP/vLLM
```

其中，真正体现“Agentic RL”的不是分布式框架，而是下面这个闭环：

```text
question
  -> LLM 生成 <think>...<search>query</search>
  -> 环境解析 action，并请求 /retrieve
  -> 环境返回 <information>documents</information>
  -> LLM 基于新 observation 再生成
  -> LLM 生成 <answer>final answer</answer>
  -> 规则奖励比较答案与 ground truth
  -> PPO/GRPO 增大高收益轨迹中模型 action 的概率
```

这里的“环境状态”并没有单独的结构体，而是被序列化为 token 并拼入模型上下文。动作同样不是离散整数，而是带 XML 风格标签的文本。

## 3. 目录和职责

### 3.1 第一批：先读这些文件

| 文件 | 先回答的问题 |
| --- | --- |
| `infer.py` | 不训练时，模型和搜索引擎如何形成循环？ |
| `scripts/data_process/nq_search.py` | 原始 question 和 golden answers 怎样进入 parquet？ |
| `train_grpo.sh` | 一次实验选择了哪些模型、算法和资源参数？ |
| `search_r1/llm_agent/generation.py` | 多条不同长度的轨迹如何并行推进？ |
| `verl/trainer/ppo/core_algos.py` | reward 怎样变成 advantage 和 policy loss？ |

### 3.2 第二批：理解完整训练数据流

| 文件 | 职责 |
| --- | --- |
| `verl/utils/dataset/rl_dataset.py` | parquet、chat template、tokenize、left padding |
| `verl/protocol.py` | `DataProto` 在 driver 和 worker 间传递 tensor/metadata |
| `verl/trainer/main_ppo.py` | tokenizer、worker role、reward manager、trainer 入口 |
| `verl/trainer/ppo/ray_trainer.py` | rollout、log-prob、reward、advantage、update 的总编排 |
| `verl/workers/actor/dp_actor.py` | policy loss、entropy、KL、backward、optimizer step |
| `verl/workers/fsdp_workers.py` | driver RPC 怎样落到 actor、rollout、critic 上 |

### 3.3 第三批：主链路清楚后再读

- `verl/workers/rollout/vllm_rollout/vllm_rollout.py`：vLLM 输入输出、sampling 和 padding。
- `verl/workers/critic/`：仅 PPO/GAE 需要的 value model。
- `verl/workers/sharding_manager/`：训练权重与 rollout 引擎之间的切换。
- `verl/single_controller/ray/`：资源池、worker group 和 RPC dispatch。
- `search_r1/search/retrieval_server.py`：BM25、dense encoder、FAISS 与 API 封装。

### 3.4 初学阶段暂时跳过

- `verl/third_party/vllm/` 的不同版本兼容代码。
- `verl/models/llama/megatron/` 和 `megatron_workers.py`。
- multinode、Ulysses sequence parallel 和 checkpoint 转换细节。

这些模块影响性能和规模，但不会改变 Agent rollout、reward 和 policy gradient 的核心语义。

## 4. 四个必须记住的接口契约

### 4.1 数据契约

每条训练数据至少包含：

```python
{
    "data_source": "nq",
    "prompt": [{"role": "user", "content": "..."}],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {"target": ["gold answer", "alias"]},
    },
    "extra_info": {"split": "train", "index": 123},
}
```

`extra_info.index` 不只是日志字段。训练时它会成为 GRPO 的 `uid`，用于标识哪些轨迹来自同一个问题。

### 4.2 文本动作协议

模型有效动作只有两种：

```text
<search>query</search>
<answer>final answer</answer>
```

`postprocess_predictions` 使用正则提取第一个完整动作。没有合法标签时，环境返回纠错提示，轨迹继续保持 active。

### 4.3 检索 HTTP 协议

请求：

```json
{"queries": ["query one"], "topk": 3, "return_scores": true}
```

响应：

```json
{"result": [[{"document": {"id": "0", "contents": "title\ntext"}, "score": 0.9}]]}
```

`LLMGenerationManager` 把响应转换为：

```text
<information>Doc 1(Title: title) text</information>
```

因此替换搜索引擎时，只要保持 HTTP 形状不变，训练侧不需要知道背后是 BM25、FAISS 还是在线搜索。

### 4.4 `DataProto` 契约

`DataProto.batch` 存 tensor，`non_tensor_batch` 存 object numpy array，`meta_info` 存不随 batch 切分的控制信息。

- `pop`：把生成所需 tensor 从原 batch 暂时取出。
- `union`：把 rollout、log-prob、value 等结果合并回来。
- `repeat(interleave=True)`：把每个问题连续复制成 `n_agent` 条轨迹。
- `reorder`：为了平衡各 GPU token 数会改变样本顺序。

由于 `reorder` 会破坏相邻关系，GRPO 必须按稳定的 `uid/index` 分组，不能假设同题样本仍然相邻。

## 5. 一次 GRPO 训练 step 的完整追踪

建议打印这张清单，每读完一步就写下输入、输出和 shape。

1. `RLHFDataset.__getitem__` 应用 chat template，生成 left-padded `input_ids/attention_mask/position_ids`。
2. dataloader batch 被包装成 `DataProto`。
3. `batch.repeat(n_agent, interleave=True)` 为每个问题产生多个独立采样机会。
4. `batch.pop(...)` 产生 `gen_batch`，其余的 ground truth 和 index 留在原 batch。
5. `run_llm_loop` 为所有轨迹建立 `active_mask`，只对未结束轨迹调用 vLLM。
6. `_postprocess_responses` 在首个 `</search>` 或 `</answer>` 处截断当前 turn。
7. `execute_predictions` 把 search action 批量发给检索器；answer action 将轨迹标为 done。
8. `_update_rolling_state` 把 action 与 observation 拼入下一轮上下文。
9. `_update_right_side` 同时维护完整 response 和屏蔽 observation 后的版本。
10. `_compose_final_output` 生成统一长度的最终轨迹和 `info_mask`。
11. actor 重新计算 `old_log_probs`。这一步不能直接信任分 turn 的 vLLM log-prob，因为最终训练序列已经重新拼接和 padding。
12. reference policy 计算 `ref_log_prob`，用于约束策略不要偏离初始模型过远。
13. `RewardManager` 解码 prompt + response，规则函数计算 EM，并把 reward 放到最后一个有效 response token。
14. `compute_grpo_outcome_advantage` 按 `uid` 收集同题 reward，执行组内标准化，再把标量 advantage 广播到有效 response token。
15. `_create_loss_mask` 用 `info_mask` 排除检索 observation token。
16. `DataParallelPPOActor.update_policy` 重新前向，计算 clipped policy loss、entropy 和 KL，最后 backward 和 optimizer step。

需要特别留意：`max_turns=N` 的循环结束后，仍可能执行一次不再允许搜索的 final rollout，因此最多会发生 `N+1` 次生成。

## 6. 张量账本

假设原始问题 batch 为 `B`，每题采样 `G=n_agent` 条轨迹，最终 response pad 到 `R`，保留的初始 prompt 长度为 `P`。

| key | 典型 shape | 含义 | 生产者 |
| --- | --- | --- | --- |
| `prompts` | `[B*G, P]` | 左侧原始 prompt | generation manager |
| `responses` | `[B*G, R]` | action 与 observation 交错的右侧序列 | generation manager |
| `input_ids` | `[B*G, P+R]` | 完整模型输入 | generation manager |
| `attention_mask` | `[B*G, P+R]` | 非 pad token | generation manager |
| `info_mask` | `[B*G, P+R]` | observation 位置为 0 | generation manager |
| `old_log_probs` | `[B*G, R]` | rollout policy 对 response token 的 log-prob | actor |
| `ref_log_prob` | `[B*G, R]` | frozen reference policy 的 log-prob | ref worker |
| `token_level_scores` | `[B*G, R]` | 通常仅末 token 非零 | reward manager |
| `advantages` | `[B*G, R]` | 轨迹 advantage 广播后乘有效 token mask | core algos |
| `loss_mask` | `[B*G, R]` | actor 真正参与 loss 的 token | trainer |

每次读到 tensor 操作，都问三个问题：batch 维是否被复制或重排？sequence 是左 pad 还是右 pad？环境 observation 是否应该参与梯度？

## 7. PPO 与 GRPO 的对应关系

### 7.1 共同部分

两者都使用当前策略生成轨迹，都计算 outcome reward，都使用 PPO clipped surrogate objective 更新 actor：

```text
ratio_t = exp(new_log_prob_t - old_log_prob_t)
loss_t  = max(-A_t * ratio_t,
              -A_t * clip(ratio_t, 1-epsilon, 1+epsilon))
```

正 advantage 希望提高该 token 概率，负 advantage 希望降低概率。clip 防止一次更新离旧策略过远。

### 7.2 PPO/GAE 分支

- 创建 critic worker，为每个 response token 预测 value。
- reward 减去 reference KL penalty 后进入 GAE。
- 得到 `advantages` 和 `returns`。
- actor 和 critic 都更新。

### 7.3 GRPO 分支

- 不创建 critic。
- 同一 question 必须有多个 sampled trajectories。
- 对该组 reward 计算 `(reward - group_mean) / group_std`。
- 配置中通常启用 actor 内部的 KL loss，reference policy 仍然存在。

仓库中 `n_agent` 才是 Agent 任务的完整轨迹复制数。`rollout.n` 是一次 vLLM 调用内部的多采样数；当前主流程通常保持 `rollout.n=1`。

## 8. Mask 为什么是稳定训练的关键

完整 response 同时包含两类 token：

```text
模型生成：<think> ... <search> ... </search>
环境注入：<information> ... </information>
模型生成：<think> ... <answer> ... </answer>
```

模型需要看到检索内容，所以上述 token 都在 `attention_mask` 中；但检索内容不是策略采取的 action，不应该让 policy gradient 去“学习生成搜索结果”，所以它们在 `info_mask/loss_mask` 中为 0。

检查 mask 时不要只看总数。构造一条很短的已知轨迹，逐 token decode 并并排打印 `attention_mask` 与 `loss_mask`，确认 `<information>...</information>` 被排除而前后的模型 token 被保留。

## 9. 奖励代码中的隐式依赖

`qa_em.extract_solution` 只有在整段文本里找到至少两个 `<answer>...</answer>` 时才返回最后一个答案。原因是数据 prompt 自带示例 `<answer> Beijing </answer>`，模型的最终回答成为第二个匹配。

这不是“模型必须回答两次”，而是 prompt template 与 reward parser 之间的隐式契约。修改 prompt 示例时必须同步测试 reward parser，否则所有正确答案都可能得到 0 分。

v0.3 的 `qa_em_format.py` 还会检查完整标签顺序，并可加入 structure format 和 retrieval reward。建议先理解 v0.2 的纯 EM，再读 v0.3 的奖励塑形。

## 10. 六阶段学习安排

### 阶段 A：PyTorch 与 token loss，3～5 天

学习内容：tensor shape、broadcast、boolean mask、`log_softmax`、`gather`、causal shift、autograd、optimizer、梯度累积。

动手：

```bash
python example/learning/agentic_rl_basics.py
python -m unittest discover -s example/learning -p 'test_*.py' -v
```

先用纯 Python 查看每个中间量，再将 `group_normalized_advantages` 自己改写为 torch 版本，并与 `core_algos.py` 的输出对照。

验收：能解释为什么 `[0,0,1,1,0]` 中 reward 1 的轨迹是正 advantage，reward 0 的轨迹是负 advantage；能解释 mask 为 0 的 token 改变 log-prob 后为什么 loss 不变。

### 阶段 B：只跑 Agent，不训练，2～4 天

先读 `infer.py`，再启动学习用 mock retriever：

```bash
uvicorn example.learning.mock_retriever:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Leonardo da Vinci Pavia Cathedral"],"topk":2,"return_scores":true}'
```

mock server 只做确定性的单词重叠排序，不用于质量实验；它的用途是验证 Search-R1 的 HTTP 数据形状和 Agent 控制流。

验收：手工写出 search、invalid action、answer 三种输入分别对应的 observation 和 done 值。

### 阶段 C：数据与多轮 rollout，1 周

阅读顺序：`nq_search.py → rl_dataset.py → tensor_helper.py → generation.py`。

在纸上模拟 `B=2`、两条轨迹先结束、一条继续搜索时，`active_mask`、`responses_ids` 和 `_example_level_pad` 如何变化。然后跟踪一次 `_update_rolling_state` 和 `_update_right_side`，理解为何同一轮同时维护“模型下轮输入”和“最终训练 response”。

验收：能够为两轮轨迹填写第 6 节的完整张量账本。

### 阶段 D：奖励、GRPO、PPO，1～2 周

阅读顺序：`RewardManager → qa_em.py → compute_advantage → core_algos.py → dp_actor.py`。

先忽略 FSDP/Ray，只把它们看成远程函数调用。用断点或临时日志观察一个 batch 的 `uid`、sequence reward、group mean/std、advantages 和 loss mask。

验收：不看代码写出 `reward → group advantage → token advantage → ratio → clipped loss` 的公式和 shape。

### 阶段 E：训练编排和系统层，1 周

阅读顺序：`main_ppo.py → RayPPOTrainer.init_workers/fit → fsdp_workers.py → vllm_rollout.py`。

建立 role 表：Actor 负责训练，Rollout 负责采样，Reference 负责 KL，Critic 只服务 PPO/GAE。理解 hybrid engine 为什么让 actor 与 rollout 共用 GPU，以及参数 offload 为什么能降低显存但增加通信。

验收：能从一条 `actor_rollout_ref.*` 配置定位到消费它的 worker 或 actor。

### 阶段 F：小规模训练与修改，1 周

准备好 parquet 和端口 8000 的检索器后，使用：

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 \
  bash example/learning/train_grpo_smoke.sh
```

smoke 配置固定为 3 steps、3B model、`train_batch_size=8`、`n_agent=2`、`max_turns=1`，目的是暴露 shape、显存和协议错误，不用于评价模型效果。共享 GPU 不足时先不要强行改成单卡全量训练；继续完成算法与 mock 环境测试即可。

推荐的第一个功能修改是每次搜索扣除固定成本：

```text
final_reward = answer_em - search_cost * number_of_searches
```

先在 `agentic_rl_basics.py` 理解 `reward_with_search_cost`，再把相同逻辑接入 `RewardManager`。修改后至少测试：零次搜索不扣分、两次搜索正确扣分、不同轨迹按各自搜索次数扣分、reward 仍放在最后有效 token。

## 11. 观察哪些训练指标

| 指标 | 用途 | 常见异常 |
| --- | --- | --- |
| `critic/score/mean` | 原始规则奖励 | 长期全 0：检查格式、答案提取和数据 |
| `actor/pg_loss` | policy gradient 主损失 | 数值爆炸：检查 advantage、ratio、mask |
| `actor/ppo_kl` / `actor/kl_loss` | 新旧/reference 策略偏离 | 快速升高：学习率或 KL 系数可能不合适 |
| `actor/entropy_loss` | 采样分布不确定性 | 很快接近 0：策略可能过早坍缩 |
| `actor/pg_clipfrac` | 被 PPO clip 的 token 比例 | 长期很高：每次更新变化过大 |
| `env/number_of_actions/mean` | 平均 action 数 | 达到上限：模型没有正常结束 |
| `env/ratio_of_valid_action` | 标签协议遵守率 | 很低：prompt、tokenizer 或格式奖励有问题 |
| `env/number_of_valid_search` | 搜索行为变化 | 只升不降：可能缺少搜索成本或结束激励 |
| `state_tokens/coverage` | 实际参与 actor loss 的比例 | 接近 1：observation masking 可能失效 |

smoke test 首先检查指标是否存在、是否为有限数值、轨迹是否能结束，不要根据 3 个 step 判断算法效果。

## 12. 版本和环境注意事项

- README 推荐 Python 3.9，仓库约束 `transformers<4.48`、`vllm<=0.6.3`、`tensordict<0.6`。
- 不要在现有新版本深度学习环境中直接升级或降级这些核心依赖；为项目创建独立环境。
- 检索器最好使用独立进程或独立环境。学习阶段优先 mock/BM25，避免让 dense retriever 与训练争抢共享 GPU。
- `docs/experiment_log.md` 说明 v0.2 修复了 retrieval token masking 和 GRPO sample indexing。代码阅读以 v0.2 逻辑为基线，再研究 v0.3 format reward。
- 根目录训练脚本是官方规模参数，不适合作为首次运行参数；学习 smoke 脚本也只是结构验证起点，实际显存仍取决于模型、CUDA、vLLM 和 offload 组合。

## 13. 最终自测清单

在开始自己的功能修改前，逐项确认：

- [ ] 我能画出 `question → action → observation → action → reward`。
- [ ] 我知道 observation token 为什么可见但不参与 policy loss。
- [ ] 我知道 `index/uid` 如何在 batch reorder 后维持 GRPO 分组。
- [ ] 我能解释 `n_agent` 与 `rollout.n` 的区别。
- [ ] 我能解释 PPO 为什么需要 critic，而 GRPO 如何替代这个 baseline。
- [ ] 我能指出 reward 位于哪个 token，以及它怎样传播成 token advantage。
- [ ] 我能用 mock retriever 验证 `/retrieve` 协议。
- [ ] 我能运行纯 Python 算法测试，并为自己的 reward/mask 修改补测试。
- [ ] 我能根据 score、KL、entropy、clip fraction 和 action 指标定位常见异常。

当以上问题都能脱离代码回答时，再深入 Ray、FSDP、vLLM 权重同步和多节点扩展，学习效率会高很多。
