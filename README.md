# Transformer 学习实验工程（Decoder-only GPT，从零手写）

这是一个**从零手写 GPT** 的中文教学工程：不调用 `nn.Transformer`，不用 HuggingFace，
每一行关键代码都自己写出来，并且用**数值实验**证明「为什么这么设计」。

面向的场景：想彻底搞懂 Transformer / GPT 内部机制，并且能亲手改、亲手做消融实验。
默认在 **CPU** 上跑小模型，一节课通常几分钟内出结果，不需要显卡。

---

## 一、五分钟上手

```powershell
cd F:\26362\AI\transformer

# 1) 如果 .venv 还不存在，先建环境（会下载官方 Python 与 CPU 版 PyTorch）
powershell -File scripts\setup_env.ps1

# 2) 下载语料（英文 TinyShakespeare + 中文诗词）
powershell -File scripts\download_data.ps1

# 3) 把语料编码成 id 序列
& '.\.venv\Scripts\python.exe' scripts\prepare_data.py

# 4) 自检：公共库是否正常
& '.\.venv\Scripts\python.exe' tests\test_common.py

# 5) 开始上课
& '.\.venv\Scripts\python.exe' src\01_tensor_basics.py --quick
```

> 提示：如果中文输出是乱码，先执行 `$env:PYTHONIOENCODING = 'utf-8'`。
> 每个脚本都支持 `--quick`，用少量步数快速跑通；去掉它就是完整的实验规模。

---

## 二、课程路线（按顺序学）

| 讲 | 脚本 | 你会亲手做出来的东西 | 核心收获 |
|---|---|---|---|
| 01 | `src\01_tensor_basics.py` | 手写 softmax / 交叉熵 + bigram 语言模型 | 语言模型的任务本质；张量形状 `(B,T,C)`；交叉熵为什么初始等于 `ln(V)` |
| 02 | `src\02_embeddings.py` | Embedding 与两种位置编码 | 注意力是置换等变的，所以必须显式注入位置信息 |
| 03 | `src\03_attention.py` | **自注意力**：从加权平均一路推到 `Attention(Q,K,V)` | Q/K/V 的分工、`1/√d_k` 为什么不能省、因果掩码的意义 |
| 04 | `src\04_multihead.py` | 多头注意力 + 头数消融实验 | 多头几乎免费（参数不增），换来多种关注模式 |
| 05 | `src\05_transformer_block.py` | 残差 / LayerNorm / MLP / Transformer Block | 残差是深层可训练的前提；MLP 才是参数大头（约 2/3） |
| 06 | `src\06_gpt.py` | 组装完整 GPT，逐层追踪形状 | 残差流宽度全程不变；权重共享；初始 loss 自检 |
| 07 | `src\07_training.py` | 训练循环 + 学习率调度 + checkpoint | warmup + 余弦退火、分组 weight decay、验证集评估、断点续训 |
| 08 | `src\08_generate.py` | 采样策略：temperature / top-k / top-p | 同一份权重，采样策略决定「文风」 |
| 09 | `src\09_kv_cache.py` | KV Cache 与推理加速 | 缓存 K/V 把每步复杂度从 O(T²) 降到 O(T) |

配套自检：

```powershell
& '.\.venv\Scripts\python.exe' tests\test_common.py     # 公共库：公式 vs 实现，61 项
& '.\.venv\Scripts\python.exe' tests\test_lessons.py    # 课程脚本冒烟测试（逐讲跑 --quick）
```

---

## 三、目录结构

```
transformer\
├─ AGENTS.md              工程约定（引号规范、接口、风格）—— 改代码前先读
├─ requirements.txt       依赖清单
├─ common\                可复用库（课程脚本都基于它）
│   ├─ config.py          GPTConfig：超参数与合法性检查
│   ├─ tokenizer.py       字符级 tokenizer
│   ├─ data.py            语料下载后的读取与批采样
│   ├─ attention.py       Head / MultiHeadAttention / CausalSelfAttention
│   ├─ gpt.py             LayerNorm / MLP / Block / GPT / 参数量估算
│   ├─ train.py           学习率调度 / 评估 / checkpoint / 训练循环
│   └─ utils.py           种子、计时、参数统计、控制台编码
├─ src\                   01~09 讲课程脚本
├─ tests\                 自检测试
├─ scripts\               环境与数据脚本
│   ├─ setup_env.ps1      创建 .venv 并安装依赖
│   ├─ download_data.ps1  下载语料
│   └─ prepare_data.py    语料 -> id 序列
├─ data\                  语料（raw 原始文本 / processed 编码结果）
├─ out\                   训练产物：checkpoint、训练曲线、图片
└─ tools\                 工程内自带的 Python 解释器（由 setup_env.ps1 下载）
```

---

## 四、把模型跑起来（07 讲的最小闭环）

```powershell
# 训练一个小模型（CPU 上约几分钟）
& '.\.venv\Scripts\python.exe' src\07_training.py --preset tiny --steps 1200

# 用训练好的权重续写
& '.\.venv\Scripts\python.exe' src\08_generate.py `
    --checkpoint out\gpt_tiny_tinyshakespeare_best.pt `
    --prompt "ROMEO: " --temperature 0.8 --top-k 40

# 看采样策略与 KV Cache 的性能差异
& '.\.venv\Scripts\python.exe' src\08_generate.py --compare
& '.\.venv\Scripts\python.exe' src\09_kv_cache.py
```

模型预设（`--preset`）：

| 预设 | n_layer | n_head | n_embd | 参数量级 | 用途 |
|---|---|---|---|---|---|
| `micro` | 2 | 2 | 64 | 约 0.1M | 单元测试、冒烟验证 |
| `tiny` | 4 | 4 | 128 | 约 0.8M | 默认教学规模 |
| `mini` | 6 | 6 | 192 | 约 2.5M | CPU 上认真训练一段时间 |
| `small` | 6 | 6 | 384 | 约 9M | 对照实验（CPU 上较慢） |

---

## 五、做实验（这个工程真正的价值）

课程脚本末尾都留了「对照实验」，训练脚本也支持换语料、换规模。推荐这样玩：

1. **换语料**：`--dataset zh_poetry` 训练中文诗词，观察字符级模型在中文上的表现，
   对比英文的 loss 量级（中文词表大得多，所以 `ln(V)` 基准也高得多，别直接横向比 loss）。
2. **改结构**：在 `common\gpt.py` 里把 `Pre-LN` 换成 `Post-LN`、把 GELU 换成 ReLU/SiLU、
   关掉权重共享、改 `n_head`，然后跑 05/06 讲的对照实验看差异。
3. **看内部**：`model(x, return_weights=True)` 能拿到每一层每个头的注意力权重，
   03 讲给了可视化方法，可以自己画句子里的指代关系。
4. **消融记录习惯**：任何改动都要 `set_seed`、固定其它变量、用**验证集** loss 判断，
   并且把结论写进自己的笔记 —— 这个工程的所有结论都是这么得出的。

---

## 六、环境说明（为什么这么装）

* 本机只有 MSYS2 自带的 Python，它的平台标签是 `mingw_x86_64_ucrt_gnu`，
  `pip` 不认 Windows 的 `win_amd64` 轮子，装 numpy/torch 会退化成源码编译并失败。
  因此 `setup_env.ps1` 会在工程目录内安装一份**官方 CPython**（不污染系统），
  再用它创建 `.venv`。
* 依赖：`numpy`、`torch`（CPU 版）、`requests`、`matplotlib`（可选，用于画图）。
  实测版本：Python 3.12.10、torch 2.14.0+cpu、numpy 2.5.3。
* 想用 GPU：把 `--device cuda` 传进脚本即可，代码里已经留好分支（`pick_device`）。
  但需要自行安装 CUDA 版 torch。

---

## 七、语料来源

| 语料 | 规模 | 来源 |
|---|---|---|
| `tinyshakespeare` | 约 1.1 MB，65 个字符 | [karpathy/char-rnn](https://github.com/karpathy/char-rnn) |
| `zh_poetry` | 约 4.4 MB（元曲 + 全唐诗 + 宋词 + 诗经 + 楚辞） | [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) |

`zh_poetry` 是多个来源拼在一起的，**简繁混用**（全唐诗为繁体，元曲/宋词等为简体）。
字符级建模时同字的简繁两种写法会各占一个 id，词表因此偏大 ——
这本身就是一个很好的练习：先做简繁统一，再比较词表大小与最终 loss。

---

## 八、学完之后可以往哪走

按难度递增，都是在本工程上直接改代码就能做的：

* **位置编码升级**：把可学习位置嵌入换成 RoPE（旋转位置编码），比较长序列外推能力。
* **归一化与激活**：RMSNorm 替代 LayerNorm、SwiGLU 替代 GELU（LLaMA 的三大改动占了两个）。
* **分词器**：把字符级 tokenizer 换成 BPE，体会「词表大小 vs 序列长度」的取舍。
* **注意力变体**：分组查询注意力 GQA、滑动窗口注意力、FlashAttention 的分块思想。
* **规模与数据**：在更大的中文语料上训练，观察 loss 随参数量/数据量的缩放规律。
* **对齐**：从「预训练」走到「指令微调（SFT）」，理解 loss 掩码怎么处理。
