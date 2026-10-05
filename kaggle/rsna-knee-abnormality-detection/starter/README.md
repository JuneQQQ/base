# 起步代码：RSNA Knee Abnormality Detection

一条完整、可运行的基线流水线：**DICOM → 固定槽位切片缓存 → 报告软标签 → 2.5D CNN + 逐病灶注意力 MIL → OOF 伪标签 → Kaggle 离线推理**。
设计取舍都来自讨论区里被反复验证过的结论，详见上一级 [`README.md`](../README.md)。

> 已在本地 CPU 上用 3 个样例测试 study 跑通全流程（`scripts/smoke_test.py`），并对缓存图像做了方向、内外侧归一化的人工目检；
> 正式的 GPU 训练、Claude/vLLM 标注后端没有在这里实际跑过（没有 GPU / 不花你的 API 费用），首次使用请先跑 `--limit` 小样本。

## 目录

```
starter/
├── knee/                       # 训练与推理共用的包（推理 notebook 里需要以 dataset 形式挂载）
│   ├── config.py               # 12 个标签名、槽位/裁剪/分辨率配置（训练和推理必须一致）
│   ├── dicom_io.py             # DICOM 读取：按 IPP 排序、方向规范化、左右膝镜像、按 mm 裁剪
│   ├── model.py                # KneeNet：timm 2D 主干 + slot embedding + 每病灶独立注意力池化
│   ├── data.py                 # 缓存 → 2.5D 窗口袋；GPU 端增强（不做左右翻转）
│   └── report_labels.py        # 状态→软标签映射、静默先验、LLM rubric/JSON schema、多语言正则基线
└── scripts/
    ├── 01_build_cache.py       # 每个 study 解码一次，存成 (60, 256, 256) uint8 .npy
    ├── 02_extract_labels.py    # regex / claude(Batches API) / vllm(本地 Qwen 等) / table(公开标签) / merge
    ├── 03_make_folds.py        # 5 折；58 个 gold 设为 fold -1；相同报告分到同一折
    ├── 04_train.py             # 单折训练，输出 best.pt、OOF、gold 预测
    ├── 05_pseudo_label.py      # 报告标签 × OOF 预测 软融合（伪标签）
    ├── 06_infer_submit.py      # Kaggle 提交：多进程解码 + 多 GPU + rank 融合 + 兜底
    └── smoke_test.py           # 3 个样例 study 端到端连通性检查
```

## 0. 环境

```bash
pip install -r requirements.txt
python scripts/smoke_test.py --data-dir <竞赛数据目录>     # 约 1 分钟，CPU 即可
```

Kaggle 提交 notebook 必须断网：`pydicom`、`timm`、`opencv` 镜像里自带；JPEG Lossless / JPEG2000 需要 `pylibjpeg*`（或 `python-gdcm`），
可以在编辑器 *Add-ons → Install Dependencies* 里装，或者提前把 wheel 打成 dataset 离线安装（讨论 735854）：

```bash
pip download --only-binary=:all: --platform manylinux2014_x86_64 --python-version 312 --no-deps \
    -d ./wheels pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm
# notebook 里：pip install --no-index --find-links /kaggle/input/<wheels-dataset> pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm
```

## 1. 构建切片缓存（CPU，不占 GPU 配额）

```bash
python scripts/01_build_cache.py --data-dir /kaggle/input/competitions/rsna-knee-abnormality-detection \
    --split train --out /kaggle/working/cache_256 --workers 4 --shard 0/2
```

- 默认 5 个槽位共 60 张切片：矢状 FS 16 / 矢状非 FS 12 / 冠状 FS 12 / 冠状非 FS 8 / 轴位 FS 12，每个序列取 6%–94% 区间均匀采样（外侧切片里有侧副韧带和外侧半月板，别只取中间）。
- 每张切片按 PixelSpacing 中心裁 140 mm（FOV 不够时补零而不是缩放），再缩放到 256。
- 切片顺序按 `ImagePositionPatient` 在法向上的投影排序（文件名是随机 UID，按文件名排序 = 打乱，讨论 745759 实测 −0.028 AUC）。
- 统一方向（矢状/冠状头在上、轴位前方在上），右膝镜像成左膝：冠状/轴位图像左侧恒为内侧，矢状切片顺序恒为内→外。
- 4407 个 study × 3.9 MB ≈ 17 GB，接近 `/kaggle/working` 上限，建议 `--shard 0/2`、`--shard 1/2` 分两个 session 跑，各存成一个 private dataset，训练时把两个目录合并（或软链接到同一目录）。可断点续跑。

## 2. 训练标签

```bash
# 2a. 规则基线（几秒，CPU）：本仓库实测与 58 个 gold 的 macro AUC = 0.816（未针对 gold 调参）
python scripts/02_extract_labels.py regex --train-csv train.csv --out labels_regex.csv

# 2b. 公开 LLM 标签（讨论区实测：stevenleehans v4 blend ≈ 0.893，pilkwang v2 ≈ 0.870）
python scripts/02_extract_labels.py table --train-csv train.csv --table llm_labels_v4_blend.csv --out labels_pub.csv

# 2c. 自己用 Claude 读报告（Message Batches API 半价、异步；结构化 JSON 输出；可断点续跑）
export ANTHROPIC_API_KEY=...
python scripts/02_extract_labels.py claude --train-csv train.csv --out labels_claude.csv --limit 50   # 先小样本看质量
python scripts/02_extract_labels.py claude --train-csv train.csv --out labels_claude.csv

# 2d. 本地开源模型（免费，需要 GPU）
python scripts/02_extract_labels.py vllm --train-csv train.csv --model Qwen/Qwen3-14B --out labels_qwen.csv

# 2e. 多个独立来源取平均（两个读者的噪声会相互抵消）
python scripts/02_extract_labels.py merge --train-csv train.csv --inputs labels_claude.csv labels_pub.csv --out labels.csv
```

LLM 后端让模型对每个病灶输出 6 选 1 的状态：`positive / probable / equivocal / below_threshold / negative / not_mentioned`，
rubric 里写进了主办方的判读标准（中重度积液才算阳性、>50% 软骨缺失才算 OA、高级别撕裂才算 ACL/MCL 阳性等）。
`statuses_to_soft()` 再把状态映射成软标签和逐格损失权重，并做两件讨论区验证过的事：

- **"未提及"不等于阴性**：每个病灶有自己的静默先验（Baker 囊肿静默时几乎都是阴性；滑膜炎静默时仍约 34% 为阳性），静默格子降权。
- **滑膜炎从积液推断**：只填补"未提及"的滑膜炎格子（讨论 733932：0.678 → 0.790）。

这些先验是经验默认值，在 `knee/report_labels.py` 顶部，可按 gold + LB 调。脚本每次都会打印新标签与 58 个 gold 的逐类 AUC——
但 58 个样本太小（每类 9–35 个阳性），差 0.02 以内基本是噪声，**不要用它来挑标签版本**，用 LB / OOF 才更可靠。

关于 Claude 后端：默认模型 `claude-opus-5-5`、`effort=low`，`--model` 可换；Batches API 价格减半，rubric 作为 system 前缀加了 prompt caching。
被拒/失败的请求会自动改走同步接口并开启服务端 `fallbacks`（Batches API 不支持该参数）。主办方已明确允许把报告发给商用 LLM（讨论 733965）。

## 3. 折与训练

```bash
python scripts/03_make_folds.py --train-csv train.csv --labels labels.csv --out folds.csv
for f in 0 1 2 3 4; do
  python scripts/04_train.py --cache cache_256 --labels labels.csv --folds folds.csv --train-csv train.csv \
      --fold $f --backbone resnet34 --img 224 --k 16 --bs 8 --epochs 15 --out runs/r34/f$f
done
```

- 模型：每个窗口 = 同一槽位内 3 张相邻切片当 RGB，走 2D 主干；窗口特征加槽位 embedding 后做**每个病灶独立的 softmax 注意力池化**，再加一个 masked-mean 残差头。
- 训练时每个 study 随机取 `--k` 个窗口，验证/推理时用全部 50 个窗口。
- 归一化常数从 timm 的 `pretrained_cfg` 读取（换主干时最常见的静默 bug，讨论 735154）。
- 自动选 AMP 精度：A100 等用 bf16，T4/P100 用 fp16（T4 上 bf16 是模拟的，慢 50 倍，讨论 744230）。
- 验证：OOF 上对报告标签（≥0.5 二值化）算 macro AUC 选 epoch；同时打印 58 个 gold 的 AUC 作 sanity check。
- 改进方向（讨论区有正收益报告）：`coatnet_rmlp_2_rw_384`/`convnext_tiny`/`tf_efficientnetv2_s` 多主干、`--img 288/384`、增加切片密度（改 `config.py` 的 slots 后需重建缓存）、`--ema 0.998`、RadImageNet 主干（注意其输入约定是单切片×3，而非相邻三切片）。

## 4. 伪标签（软自举）

```bash
python scripts/05_pseudo_label.py --labels labels.csv --oof "runs/*/f*/oof_fold*.csv" --out labels_pl.csv
# 再用 labels_pl.csv 训练第二轮；验证仍然用原始 labels.csv（否则 CV 会因构造而虚高）
```

必须用 **OOF** 预测；报告有明确陈述的格子 50/50，报告静默的格子更偏向模型（默认 30/70）。

## 5. 提交

1. 把 `starter/` 上传成一个 Kaggle dataset（代码），把 `runs/*/f*/best.pt` 上传成另一个 dataset（权重）。
2. 新建 notebook：挂载竞赛数据 + 上述两个 dataset，加速器选 **T4 ×2**（P100 与新版 PyTorch 不兼容，讨论 735854），关闭网络。
3. 先用训练集演练真实规模（隐藏测试集约 1300 个 study，3 个样例跑通不代表能跑完）：

```bash
!python /kaggle/input/<code>/starter/scripts/06_infer_submit.py --ckpt "/kaggle/input/<weights>/**/best.pt" \
    --split train --limit 1300 --out /kaggle/working/dryrun.csv
```

4. 正式提交时去掉 `--split/--limit`，输出 `/kaggle/working/submission.csv`。脚本会打印所有兜底（fallback）的 study——如果数量不为 0，先查原因再提交。

效率赛道（`AUC / (Benchmark − maxAUC) + RuntimeSeconds/32400`，越小越好）：最终两个提交都会参与两个赛道，
可以一个放精度最高的集成、一个放单模型快速版（讨论区效率榜第一是 5–6 分钟跑完的小 ResNet）。
