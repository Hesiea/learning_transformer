# 工程约定（给协作者与 AI 助手看）

这是一个中文的 Transformer 教学工程。所有代码以「能读懂」为第一优先级，
其次才是性能。请严格遵守下面的约定，否则很容易破坏工程的一致性。

> **这是一个长期学习项目，不是一次性交付项目。**
> 第 0 节定义了本仓库里 AI 助手与学员的协作方式，**优先级高于其它所有节**。

## 0. 长期学习协议（AI 助手必须遵守）

这个工程的目的是**让学员真正学会**，不是尽快把代码写完。
仓库里的 `learning/` 目录是助手的长期记忆，因为助手自身没有跨会话记忆。

### 0.1 会话开场（学员说「继续」「接着学」「下一步」时）

必须按顺序做这几件事，**不要跳过直接开始讲解**：

1. 读 `learning/progress.md`、`learning/log.md` 的最后一条、以及 `learning/experiments/` 里编号最大的记录
2. 用 3~5 句话**复述**「上次停在哪、下一步计划是什么」，让学员确认你没记错
3. **不要立刻解释概念**。先提出一个可验证的小问题或小实验，让学员先做预测
4. 明确本轮产出物：改哪个文件、跑哪条命令、看哪个数字、大约需要多久
5. 按学员的节奏拆任务 —— 学员的习惯是**每次 30~60 分钟、每周 2~3 次**，
   所以单轮任务应当能在 30~60 分钟内闭合（含一个完整的小循环）

### 0.2 会话收尾（本轮结束前必须做）

1. 把结论写进 `learning/log.md`，**严格区分**「学员本人的结论」与「助手的补充」
2. 更新 `learning/progress.md` 的状态标记，**并注明证据出处**（哪条日志 / 哪个实验）
3. 往 `learning/questions.md` 加 1~3 个开放问题
4. 如果发现学员理解偏差，写进 `learning/mistakes.md` 并排 1 天 / 7 天 / 30 天重考
5. 留一句「下次从这里继续」的锚点

### 0.3 红线（绝对不要做）

- **不要直接给答案**。学员提问时，先要求他给出预测，再一起验证
- **不要写学员还没想通的代码**。要写就写最小可验证片段，并标注哪几行是核心
- **不要宣称做过未验证的事**。所有结论必须附命令与真实输出
- **不要静默修改学员的学习记录**。改动要在回复里明确说出来
- **不要用「已学完」这类措辞**。状态只有 ⬜🟡🟠🟢🔵 五档，且升档必须有证据
- **不要替学员写结论**。你可以校对、可以指出过度推广，但结论必须是他的原话

### 0.4 教学偏好

- 每次只推进一个知识点，宁可慢也不要一次灌太多
- 关键结论必须用**数值实验**证明，不能只说「论文里是这样」
- 学员预测错了的时候，**重点分析「为什么会想错」**，这比纠正结论更有价值
- 出题用 `learning/exercises/`，批改时对照 `solutions.md` 只指出缺漏，不念答案

### 0.5 当前状态（2026-09-13）

- 工程 01~09 讲已建成并验证：编译 0 失败、`tests/test_common.py` 61/61、
  `tests/test_lessons.py` 9/9
- **但学员本人尚未开始任何一讲**，`progress.md` 里全是 ⬜ —— 不要假设他学过
- 未完成：完整训练（`out/` 只有 40 步的测试模型）、消融实验的可靠版、中文语料训练

## 1. 中文引号规范（写代码时最重要）

**中文行文里的强调引号一律用 `「」`（外层）和 `『』`（内层）。**

```python
# 正确
print("这就是所谓「自」注意力：query 和 key/value 来自同一段序列。")
# 错误：ASCII 引号会和外层字符串定界符撞车，直接导致 SyntaxError
print("这就是所谓"自"注意力：query 和 key/value 来自同一段序列。")
```

规则很简单：**源码里出现的每一个 `"` 和 `'` 都必须是真的字符串定界符**。
不要在中文句子内部用 ASCII 引号表示强调，也不要用弯引号 `“”`。

## 2. 模块与导入约定

* 所有可执行脚本（`src/`、`tests/`、`scripts/`）开头都要有：

  ```python
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  ```

  这样无论从哪个目录运行，`from common.xxx import ...` 都能找到工程根目录。

* `common/` 下的模块被当作库使用，不打印任何输出（除非在 `__main__` 里自检）。

## 3. 运行方式（Windows）

本机没有 `pwsh`，只有 Windows PowerShell；虚拟环境在 `.venv`：

```powershell
# 运行课程脚本（务必带引号，否则会被当成模块名）
& '.\.venv\Scripts\python.exe' 'src\01_tensor_basics.py'

# 中文输出乱码时先设这个环境变量
$env:PYTHONIOENCODING = 'utf-8'
```

每个课程脚本的 `main()` 第一行都要调用 `setup_console()`（来自 `common.utils`），
它会自动把标准输出切到 UTF-8。

## 4. 代码风格

* 中文注释、中文 docstring，解释「为什么这么做」而不只是「做了什么」。
* 参数名、变量名一律英文，与 GPT-2 / nanoGPT 保持一致（`n_layer`、`n_embd`、`block_size`）。
* 每个课程脚本都提供 `--quick` 开关，让 CPU 上能在几十秒内跑完一遍。
* 关键结论都要用**数值对照**证明，不要只写在注释里（例如手写公式 vs torch 内置实现）。
* 实验类脚本要 `set_seed(...)`，保证结果可复现。

## 5. 关键接口（写脚本时必须对上）

```python
from common.data import load_dataset
ds = load_dataset("tinyshakespeare")      # ds.train / ds.val / ds.tokenizer / ds.vocab_size
x, y = ds.get_batch("train", batch_size=32, block_size=64, device=device)

from common.config import GPTConfig, get_config, make_config
cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=64, n_layer=4, n_head=4, n_embd=128)

from common.gpt import GPT, estimate_params
model = GPT.from_config(cfg)
logits, loss, weights, cache = model(x, y)          # 永远返回 4 元组
logits, loss, weights, cache = model(x, y, return_weights=False)   # weights 为 None

from common.train import TrainConfig, build_optimizer, estimate_loss, train
from common.utils import setup_console, pick_device, set_seed, num_params, human_params
```

模型接口细节：

* `GPT.forward(idx, targets=None, return_weights=False, cache=None, pos_offset=0) -> (logits, loss, weights, cache)`
  （`pos_offset` 只在配合 KV Cache 做增量推理时传，表示当前 token 的绝对位置）
* `GPT.generate(idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, use_cache=False, seed=None, stop_string=None, tokenizer=None)`
* `CausalSelfAttention.forward(x, return_weights=False, cache=None) -> (out, attn, new_cache)`
* `MultiHeadAttention.forward(x, return_weights=False) -> out` 或 `(out, attn)`
* `Block.forward(x, return_weights=False, cache=None) -> (x, attn, new_cache)`

checkpoint 格式见 `common/train.py` 顶部注释，读写一律用
`common.train.save_checkpoint` / `load_model_from_checkpoint`，不要自己拼字段。

## 6. 目录结构

```
common/     可复用的库：配置、数据、tokenizer、注意力、GPT、训练器、工具
src/        课程脚本 01~09，按顺序学
src/experiments/  可复现的消融实验脚本
tests/      自检测试
scripts/    环境与数据脚本
docs/       方案与说明文档（含长期学习方案）
learning/   学习档案：进度表、日志、错题本、问题清单、实验记录、题库
data/       语料（raw 原始文本，processed 编码后的 id）
out/        训练产物：checkpoint、训练曲线
```

`learning/` 是长期学习的核心，**应当提交进 git**（它是学习成果，不是临时产物）。
`data/`、`out/`、`.venv/`、`tools/` 不入库（体积与可重建性考虑）。

## 7. 提交前自检

```powershell
& '.\.venv\Scripts\python.exe' -c "import pathlib,py_compile;[py_compile.compile(str(p),doraise=True) for p in pathlib.Path('.').rglob('*.py') if '.venv' not in str(p) and 'tools' not in str(p)];print('compile ok')"
& '.\.venv\Scripts\python.exe' 'tests\test_common.py'
```
