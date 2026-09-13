# 工程约定（给协作者与 AI 助手看）

这是一个中文的 Transformer 教学工程。所有代码以「能读懂」为第一优先级，
其次才是性能。请严格遵守下面的约定，否则很容易破坏工程的一致性。

## 1. 中文引号规范（最重要）

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

* `GPT.forward(idx, targets=None, return_weights=False, cache=None) -> (logits, loss, weights, cache)`
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
tests/      自检测试
scripts/    环境与数据脚本
data/       语料（raw 原始文本，processed 编码后的 id）
out/        训练产物：checkpoint、训练曲线
```

## 7. 提交前自检

```powershell
& '.\.venv\Scripts\python.exe' -c "import pathlib,py_compile;[py_compile.compile(str(p),doraise=True) for p in pathlib.Path('.').rglob('*.py') if '.venv' not in str(p) and 'tools' not in str(p)];print('compile ok')"
& '.\.venv\Scripts\python.exe' 'tests\test_common.py'
```
