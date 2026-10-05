# RSNA 2026 Knee Abnormality Detection 调研报告

> 数据截至 **2026-10-05**（距最终提交截止 10-22 还有 17 天；报名/组队截止 10-15）。
> 来源：通过 Kaggle API 抓取的全部 156 个讨论帖（含评论）、约 450 个公开 notebook 列表及其中 16 个重点 notebook 源码、
> 公开榜与效率榜、往届 4 个 RSNA 赛的 TOP 方案帖，以及 GitHub / 论文。
> 你的账号状态：已报名，0 次提交。起步代码在 [`starter/`](starter/README.md)。

---

## 0. 结论先行

1. **这是一个"造标签"的比赛。** 4,407 个训练 study 中只有 58 个有放射科医生标注；其余只有 9 种语言的原始报告，测试集没有报告。
   所有人都在用报告生成训练目标，标签质量和"如何使用不确定/缺失的标签"决定上限。LLM 读报告 ≈ 0.88–0.89 macro AUC（对 58 个 gold），
   规则匹配 ≈ 0.81。图像模型（学生）在 gold 上能反超报告标签（老师），说明模型能学到报告里没写的东西。
2. **金标准来自读图而不是读报告，且阈值更严**（中/大量积液、高级别撕裂、>50% 软骨缺失才算阳性），报告与 gold 存在系统性不一致，主办方确认以图像为准。
3. **不需要大模型。** 讨论区多位高分选手：ResNet34/50、EfficientNet-B0/B4 在 224–288px 的 2.5D 方案就能到 0.94+；DINOv2-S→B 无增益；
   真正的杠杆是：切片物理排序、按 mm 裁剪、切片覆盖范围与密度、每病灶独立的注意力池化、软标签 + OOF 伪标签融合。
4. **验证要用报告标签的 OOF**（几千个样本，噪声约 0.002–0.003），LB 做校准；58 个 gold 只能做 sanity check。公开榜只显示 3 位小数，靠调融合权重刷榜几乎全是过拟合。
5. **推理工程决定能否稳定得分，也决定效率赛道。** 效率榜第一（0.959，公开）5–6 分钟跑完；常见翻车：T4 上用 bf16（慢 50 倍触发超时兜底）、P100 不兼容、缺 JPEG 解码器、3 个样例能跑但 1300 个隐藏样本崩溃。
6. **奖牌线（5,184 队）**：金 ≈ 0.958（第 20 名），银 ≈ 0.945（第 259 名），铜 ≈ 0.943（第 518 名）。有 **1,007 支队伍正好卡在 0.943**——绝大多数来自同一条公开集成 notebook 的 fork 链，private 榜大概率洗牌。
   想拿牌必须有**独立于公开权重的自有模型**（讨论区也观察到：公开 stack 0.943 + 自有 ConvNeXt 0.929 按 7:3 融合 → 0.944）。

---

## 1. 赛题速览

| 项目 | 内容 |
|---|---|
| 任务 | 每个膝关节 MRI study 预测 12 个异常的概率：ACL、MCL、内/外侧半月板撕裂、内/外侧/髌股关节 OA、积液、滑膜炎、Baker 囊肿、骨挫伤、骨折 |
| 指标 | 12 个标签 ROC-AUC 的宏平均（只看排序，校准无关；集成用 rank 平均即可） |
| 训练数据 | `train.csv`：4,407 study，`Report` 原始报告（英/西/土/克罗地亚-波斯尼亚/希腊/德/保加利亚/荷/法），12 列标签**仅 58 行非空**；`train_series.csv`：24,371 个序列（平面 + `Fluid_Sensitive`/`Fat_Suppression`，两列在训练集完全相同）；DICOM 共 ~570 GB / 82 万文件，序列多为 20–45 张（中位 30），长尾到数百张 |
| 测试数据 | 约 1,300 study，**无报告**；只有 3 个样例可见；公开/私有划分方式（按站点？）未公布 |
| 提交 | Code competition：CPU/GPU notebook ≤ 9 小时，断网，输出 `submission.csv`；可选 2 个最终提交（两个都参与主赛道和效率赛道） |
| 效率赛道 | `Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400`，越小越好；RuntimeSeconds 含装包、加载模型、读 DICOM 全过程 |
| 奖金 | 主赛道 $9,000 → $5,000（前 10），效率赛道 $7,000 / $6,000 / $5,000；获奖需开源代码与权重（CC-BY-NC 4.0）、录视频 |
| 时间线 | 7-30 开赛；**10-15 报名 & 组队截止**；**10-22 最终提交截止**；11-05 获奖者材料截止 |
| 外部数据/工具 | 公开且"人人可合理获取、成本极低"的数据和模型允许；**商用 LLM API 读报告明确允许**（[733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)）；KneeCoT 明确禁止（[734109](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109)） |

12 个标签的判读标准（主办方 [733343](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)，两名 MSK 医生独立读图 + 第三人仲裁，模棱两可判阴性）：

| 标签 | 阳性标准 | 不算阳性 |
|---|---|---|
| ACL | 高级别部分撕裂（>50% 纤维）或完全撕裂 | 信号改变、退变、增粗而无连续性中断 |
| MCL | 急性高级别部分/完全撕裂 | 低级别扭伤、慢性/陈旧改变 |
| 内/外侧半月板 | 异常信号确定触及表面 ≥2 张，或截断/变小/移位碎片 | 未达表面的内部退变 |
| 内/外/髌股 OA | 该间室中/大面积（约 ≥1 cm）高级别软骨缺失（>50% 厚度） | 轻度软骨病 |
| 积液 | 中量或大量 | 少量 |
| 滑膜炎 | 滑膜炎症、增厚 | |
| Baker 囊肿 | 中/大的腘窝囊肿 | 小囊肿 |
| 骨挫伤 | 撞击性骨髓水肿样信号，无明确骨折线 | |
| 骨折 | 急性皮质断裂或骨折线 | |

58 个 gold 的阳性率被刻意富集（积液 60%、ACL 41%，而全库 ACL 约 20%），不要拿它估先验。

---

## 2. 讨论区要点（按主题）

### 2.1 标签：比赛的主战场

| 结论 | 证据 |
|---|---|
| LLM 读报告明显好于规则：0.878 vs 0.814（gold macro AUC） | [733932](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932) |
| 让 LLM 显式输出"报告未提及"：25.4% 的格子是未提及；**沉默的含义因病灶而异**——滑膜炎 84% 未提及但 gold 阳性仍有 34%；Baker 囊肿未提及时阳性仅 3% | 733932 |
| 只对"未提及"的滑膜炎用积液字段填补：滑膜炎 0.678→0.790；对所有病灶做学习式填补反而变差 | 733932 |
| 公开标签表对 gold：stevenleehans v4 blend 0.893、v2 0.887、pilkwang v2 0.870；两张表取平均在多个标签上更好 | [737454](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737454)、[notebook](https://www.kaggle.com/code/starkhushi/rsna-knee-which-report-labels-should-you-train-on) |
| 报告 vs gold 不一致：主办方确认标签由读图独立产生、以图像为准；报告是单个医生为临床写的，阈值更松 | [733826](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826)、[733491](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733491) |
| 软标签优于硬 0/1（3 个种子全胜，gold +0.014，但 CI 跨 0）；把软标签取整到 0/1 会掉分 | [734105](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105)、[735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304) |
| **OOF 伪标签融合有效**：用 OOF 预测与报告标签混合（不是替换、不是屏蔽），50/50 已明显提升，报告静默的格子更偏向模型更好；必须是 OOF，in-fold 预测只是复读标签 | 735304、[743148](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148) |
| 多来源弱标签（公开 + API + 本地）→ 训练多个 teacher → 伪标签 → 最终模型，CoAtNet 单模 0.949；API 总花费 < $5 | 735304 |
| 换 LLM 几乎没用：qwen2.5-14b / qwen3-14b / qwen3-30b-a3b 在 gold 上差 0.01 以内；qwen3 的 `<think>` 会和 JSON 约束解码冲突 | 743148 |
| 抽取已接近天花板：240 个 gold 阳性中抽出 207 个，漏掉的 33 个里 22 个报告根本没写 | 743148 |
| 反方观点：有人认为"单 prompt LLM 抽取"之后再怎么精修标签都不涨 LB；也有人换标签版本 DINOv2 从 0.931→0.942 | 735304、[745214](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745214) |
| 规则抽取要点：多语言词表，**土耳其语否定在词后**（"efüzyon izlenmedi"），OA 几乎不直接写"OA"而是写软骨缺失/骨赘/间隙变窄 | [734106](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734106)、[734095](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734095) |
| 用 VLM 从图像生成报告再抽标签：负收益；报告 embedding 线性探针也不如 12 个抽取数值 | 743148 |
| 131 个 study 共享相同报告（46 组，最大一组 37 个），划分折时必须分组 | [starkhushi notebook](https://www.kaggle.com/code/starkhushi/rsna-knee-which-report-labels-should-you-train-on) |

### 2.2 预处理：最容易"白送"分数的地方

- **切片顺序**：文件名是随机 SOP UID，按文件名排序≈打乱；按 `ImagePositionPatient` 投影到法向排序。一个公开 baseline 修复后 **+0.0276**（[745759](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745759)）。`InstanceNumber` 在 1/3 以上序列中与位置方向相反（[741310](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741310)）。
- **物理尺度**：FOV 有 71 种取值（中位 160 mm），先按 PixelSpacing 裁固定 mm 再缩放；常用 130–150 mm（[735154](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154)、735304）。
- **切片覆盖与密度**：只取中间 20–80% 会丢掉侧副韧带和外侧半月板；dreaddevelopment 把 44 张切片从 6–94%（0.926）摊到 2–98% 反而降到 0.917，在 2–98% 覆盖下增加到 64/80 张才回到 0.928/0.932；他的结论是"密度才是杠杆"（[737696](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696)）。另一方用注意力 head 测出相反方向，说明和聚合方式有关。
- **2.5D 输入**：相邻 3 张当 RGB 比 9 张均匀散开的切片好（+0.018，[737597](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737597)）；层间距约是像素间距的 10 倍，3D from scratch 不划算。
- **序列选择**：每个 study 3–14 个序列，所有 study 都有三个平面和两种对比；常用 4–6 个槽位（平面 × 是否液体敏感），缺失槽位用 mask。
- **左右膝**：Laterality 标签约一半缺失，可由图像中心在病人坐标系的 x 符号推出（与标签 98.4% 一致）。归一化左右的收益被测为很小（737597），但做了不亏。约 7 个训练 study 两侧膝都在视野内（[735639](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735639)）。
- **"权重自带预处理契约"**：RAD-DINO 用了错的均值方差导致白白落后（735154）；RadImageNet 是单通道复制×3 训练的：RadImageNet-R50 喂相邻三切片 0.834，改成中心切片×3 后 0.854（+0.020；同设置 ImageNet-R34 为 0.815）（737696）。
- 两个训练 DICOM 的 PixelData 被截断，必须 try/except（[737163](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737163)）。
- DICOM 元数据捷径不存在：只用头信息 0.65（随机折）/0.60（按扫描仪分组），站点记忆不可迁移（[733517](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517)）。测试集头信息里有 PatientSex（男性 ACL 多、女性内侧 OA 多，[734004](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734004)）。

### 2.3 模型

| 方案 | 公开榜 | 来源 |
|---|---|---|
| ResNet50 224px 5 折 | 0.954 | 735304 |
| Qwen3.5-2B VLM（LoRA + 视觉编码器联合微调，384px）单折，约 3.5 小时 | 0.950 | 735304 |
| CoAtNet 224 2.5D 单折（核心是标签处理） | 0.950 | 735304 |
| 小 ResNet，自建 Gemma 标签（gold≈0.89），单模单折，~5 分钟推理（效率榜第一） | 0.949 | 735304、[740474](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740474) |
| CoAtNet 288 全量训练 + 多 teacher 伪标签 | 0.949 | 735304 |
| DINOv2 0.942 / EffNet-B4 0.940 / ResNet50 0.940（各 5 折） | — | 735304 |
| CoAtNet-rmlp-2 384 + 逐病灶注意力 MIL（dreaddevelopment "Raptor"，早期权重公开；0.944 为后期与 ConvNeXt 双臂、13 分钟推理） | 0.924 → 0.944 | [737696](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696) |
| 公开大集成（20 个 DINOv2 + DINOv3 + RadImageNet + CoAtNet，"Bend the Knee" 系列） | 0.941–0.943 | [742050](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742050) |

- 编码器变大无收益：DINOv2-S→B +0.0011（噪声 0.002）；Base 只是收敛更快（735154、[738096](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738096)）。
- DINOv3 多人反馈不如 v2，且许可存疑（[733313](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733313)）。
- 逐病灶独立注意力池化很关键：换成 masked mean −0.014；训练用注意力、推理改池化 −0.02（737696）。
- 医学预训练（RadImageNet）单独拉高滑膜炎（0.62–0.72 → 0.78，[737566](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737566)），适合当集成成员。
- 没有带来提升的：EMA、mixup、更长 schedule（都在 ±0.003 内）、Asymmetric loss（−0.139）（737597）；Pillar-0（乳腺 MRI 基础模型）迁移 0.797（[738036](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738036)）；报告作为辅助损失（[740162](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740162)）。
- 未被充分验证：GRU/LSTM 切片序列头（往届 RSNA 冠军常用，本赛只有提问 [738495](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738495)）、膝关节 ROI 检测裁剪（[736999](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736999)）。
- 单折全量训练 ≈ 5 折集成（0.941/23 分钟 vs 0.939/5 小时，[740375](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740375)）。

### 2.4 验证与洗牌风险

- 推荐做法（tuckerarrants，[740610](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740610)）：在报告标签上做 OOF，先测出种子噪声（约 0.003），只保留超过噪声的改动；用 LB 建立 CV→LB 对应；58 个 gold 每类只有 9–35 个阳性，区分不了 0.02 以内的差距。
- 多人观察到 LB 普遍高于 OOF（报告标签有噪声，而 LB 是图像标签，[736635](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736635)）。
- 公开榜 3 位小数：有人 13 次提交都是 0.937，两周调融合权重只换来 0.001（[740439](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740439)）；约 200 个 fork 卡在 0.941，差别只是融合权重（742050）。
- 洗牌：铜牌线附近历史上只有约 24% 的队伍最终保住奖牌（[742327](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742327)）；公开/私有是否按站点划分未知（[734681](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734681)）。

### 2.5 推理与效率

- 时间去向：解码/排序占大头。优化清单（[743374](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743374)，CoAtNet 40 → <10 分钟）：每张 GPU 一份模型、分片处理（不用 DataParallel）；每个 DICOM 只读一次、16 线程并发读（59 → 372 MB/s）；直接解析头（0.307 → 0.058 ms/文件）；`np.frombuffer` 直接读 PixelData；百分位在下采样像素上算；fp16 权重 + half 输入；uint8 上 GPU 再归一化。
- 坑：T4/P100 没有 bf16，PyTorch 会模拟且慢约 50 倍，触发时间兜底导致一半样本 0.5，分数 0.851→0.710（[744230](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744230)）；P100 与新版 PyTorch 不兼容，要选 T4；JPEG 解码器需离线安装；竞赛数据路径是 `/kaggle/input/competitions/<slug>/`（[735854](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735854)）。
- 在 ~1,300 个训练 study 上演练完整推理，3 个样例跑通不代表隐藏集能跑通（[743779](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743779)）。

### 2.6 规则与外部数据（尚有悬而未决的问题）

- 商用 LLM API 处理报告：**允许**（733965）。
- KneeCoT：**禁止**（需机构协议）。OAI：需要机构签署 DUC，主办方回答"需机构审批的不符合"，社区普遍解读为不可用（[741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819)）。MRNet / KneeMRI 带 No-Derivatives 条款，而获奖必须公开权重——用了可能无法履行获奖义务（[737950](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737950)、[743420](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743420)）。多人追问主办方逐一表态（[743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)，62 票）至今无答复。**冲奖就别碰外部膝关节数据。**
- 使用别人用本赛数据训练并公开的权重（Raptor、DINOv2 20 模型等）是否具备获奖资格、如何满足"提交训练代码"义务：未答复（[744056](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744056)、[741212](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741212)）。DINOv3、RadImageNet（CC-BY-NC-SA）许可兼容性也未答复。

---

## 3. 开源代码与数据集

**值得读的 notebook**（讨论区多人提醒：高票 notebook 里大量是 AI 生成或调权重的 fork，优先看有原创训练代码、有评论的）

| Notebook | 票数 | 价值 |
|---|---|---|
| [pilkwang / RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | 666 | 最早的完整基线：多语言规则抽取 + LLM 标签表、DINOv2 |
| [dreaddevelopment / training the twelve-finding model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model) + [inference](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) | 126 / 171 | **最推荐**：CoAtNet-384 + 逐病灶注意力 MIL、5 槽位 64 切片、140 mm、span 6–94%、top-3 epoch 权重平均（被点名为"少数真有用的 notebook"） |
| [mattiaangeli / Bend the Knee to the Dinosaurs](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs) / [Speedy Raptors](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original) | 316 / 214 | 公开最高集成（0.941–0.943）的源头；元数据里记录了每版改动 |
| [jiweiliu / RSNA Knee Fast 2xT4 Inference](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference) | 236 | 双 T4 推理工程：共享冻结前缀、CPU/GPU 重叠、缓存 slice 顺序 |
| [skarin / Reproducing the 0.943 public stack](https://www.kaggle.com/code/skarin/rsna-knee-reproducing-the-0-943-public-stack) · [goodpjw2008 / Stack + 2.5D ConvNeXt MIL 0.944](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944) | — / 17 | 公开 stack 复现；+ 独立训练的 ConvNeXt-T（0.929）按 7:3 融合到 0.944 |
| [prvsiyan / read the report, then the knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) | 262 | 多语言报告解析与序列头信息恢复（TR/TE 推断加权、希腊 µ 编码问题） |
| [stevenleehans / 500GB → 11GiB CPU pixel cache](https://www.kaggle.com/code/stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache) | — | 用 CPU session 一次性缓存像素 |
| [wguesdon / DINOv2 at meniscus resolution](https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution) · [barun2104 / 2.5D ResNet34](https://www.kaggle.com/code/barun2104/baseline-rsna-knee-mri-2-5d-resnet34) | 88 / 34 | 早期基线 |
| [starkhushi / which report labels](https://www.kaggle.com/code/starkhushi/rsna-knee-which-report-labels-should-you-train-on) · [nekkon / 58 labels for 4,407 studies](https://www.kaggle.com/code/nekkon/knee-mri-58-labels-for-4-407-studies) · [gchauhan / data traps EDA](https://www.kaggle.com/code/gchauhan/eda-rsna-knee-what-the-data-tells-us) · [zhukovoleksiy / metadata probe](https://www.kaggle.com/code/zhukovoleksiy/rsna-metadata-probe) | — | 标签表评测、数据陷阱、元数据捷径检验 |
| [ryanholbrook / Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb) | 493 | 官方效率榜（每日更新，可用 `kaggle kernels output` 拉 CSV） |

**数据集**

| 类别 | 数据集 |
|---|---|
| 报告标签 | [stevenleehans/rsna-knee-llm-report-labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)（gold 0.893，含"未提及"=0.5）· [pilkwang/rsna-knee-llm-labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels) · [barun2104/…-stratified-folds-and-llm-soft-labels](https://www.kaggle.com/datasets/barun2104/rsna-knee-stratified-folds-and-llm-soft-labels) · [lixin73/…-llm-report-labels-sol56](https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56) |
| 预处理像素 | [dreaddevelopment/knee-raptor-corpus](https://www.kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus)（+`-ext`，44×336×336）· [barun2104/rsna-knee-mri-processed-3d-volumes](https://www.kaggle.com/datasets/barun2104/rsna-knee-mri-processed-3d-volumes) |
| 公开权重 | [raptor-knee-widedense](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense) / [-finespacing](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-finespacing)（单模 0.93）· [pilkwang/rsna-knee-weights](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights)（20 个 DINOv2） |

本赛相关 GitHub：[ibrahimdodo/knee-mri-raptor](https://github.com/ibrahimdodo/knee-mri-raptor)（对 Raptor 权重的切片归因与鲁棒性分析：1.25× 缩放不掉分、1.5× −0.018）。

---

## 4. 往届类似竞赛 TOP 方案

| 比赛 | 相似点 | TOP 方案要点 |
|---|---|---|
| **RSNA 2024 腰椎退变分级**（MRI，多序列、多病灶） | 同为 MRI、多平面多序列、按"位置×病灶"多标签 | **1st**：3D ConvNeXt 预测各椎间隙切片号 + 2D 回归坐标 → 裁剪 → ConvNeXt-S/EffNetV2-S + **双向 LSTM + 注意力 MIL**；大模型反而更差，ViT 不如 CNN（[540091](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091)）。**2nd**：切片/关键点定位 → 每目标 27 个偏移裁剪当 TTA；简单 2D CNN 最好；最后一周的**标签去噪**明显提分（[539452](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452)，[代码](https://github.com/brendanartley/RSNA-2024-Competition)）。**3rd**：中心/侧别分类器（2D 编码器 + 注意力），不区分椎间隙、右侧翻转后与左侧合并，数据量 ×10，30 模型平均（[539453](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)）。**4th**：多视图 transformer + Nelder-Mead/GBDT stacking。**5th**：热图检测 + 2.5D CNN+RNN，**CutMix p=1** 最有效。[总结帖 541279](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541279)：几乎所有 TOP 都是"2D 主干逐切片 → LSTM/GRU → 注意力池化"的 MIL，且主干都偏小 |
| **RSNA 2025 颅内动脉瘤**（CTA/MRA/MRI，13+1 个标签，宏 AUC，code comp） | 同为多标签 AUC、9 小时代码赛、异构扫描 | **1st**：nnU-Net 粗→精血管分割定位 ROI，再 3D 分类（[611846](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611846)，[代码](https://github.com/uchiyama33/rsna2025_1st_place)）。**2nd**：nnU-Net 3D 多任务（分割 + 分类），**左右翻转 TTA 同时交换左右标签**。**3rd**：YOLO 找 ROI + 3D ResNet18，在特征图每个位置挂分类头再取 top-N 均值。**4th**：DINOv3 回归裁剪框 + CoaT-Lite 2.5D + 软伪标签蒸馏，8 天冲进金区。**6th**：2.5D `coatnet_rmlp_2_rw_384` / `maxvit_rmlp` + 序列模型（正是本赛公开最强单模的主干）。共性：**先定位 ROI 再分类**、2.5D/3D 都有、TTA |
| **RSNA 2023 腹部创伤**（CT，多器官多标签） | 研究级多标签、study 级输出 | **1st**：3D 分割 → 器官裁剪 → 2.5D（相邻切片 3 通道）CNN + **GRU**，分割辅助损失稳定训练（[447449](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449)，[代码](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)）。**2nd**：用器官可见性模型**挑选喂给 2D 模型的帧**以降低标签噪声，2D 帧模型 + RNN 聚合，重增强 + CutMix |
| **RSNA 2022 颈椎骨折**（CT） | 2.5D 范式的源头之一 | **1st**：3D 分割椎体 → 每椎 15 张、每张 ±2 相邻切片 + mask 共 6 通道 → 2D CNN + LSTM（[362607](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362607)）；2nd：分割 + CNN + BiGRU + 注意力；3rd：2.5D CNN + 1D RNN |
| **Stanford MRNet**（学术，膝关节 MRI：ACL/半月板/异常） | 同部位同任务 | 每个平面每个任务一个 AlexNet，切片特征 max-pool，再逻辑回归融合三平面；内部 AUC：ACL 0.965、半月板 0.847；外部数据 ACL 掉到 0.824（[论文](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002699)） |

---

## 5. 共性分析：往届经验 → 本赛映射

| 往届共性 | 在本赛的体现 / 建议 |
|---|---|
| **2.5D 优于 3D**：2D 预训练主干 + 相邻切片当通道 + 切片聚合（层厚 3–4 mm、像素 0.3 mm 时尤其如此） | 讨论区共识一致；起步代码采用 3 切片窗口 |
| **切片聚合用注意力 / MIL / RNN** 而不是简单平均 | 本赛实测逐病灶注意力比 masked-mean 高 0.014；GRU/LSTM 本赛尚无人验证，可作为差异化方向 |
| **先定位后分类**（关键点、分割、检测框） | 本赛靠"物理 mm 裁剪 + 槽位选择"已经做了粗定位；进一步：按内/外侧间室或髁间窝裁 ROI（dreaddevelopment 代码里有 compartment ROI 开关，未见公开结果） |
| **小主干 + 多样化集成**，大模型/长训练常无益 | 本赛 S→B 无增益、ResNet34/50 就到 0.94+；集成要选**误差不相关**的成员（不同标签源、不同主干族、RadImageNet） |
| **标签噪声处理**：软标签、伪标签、去噪（腰椎 2nd 的 noise reduction、腹部 2nd 的帧挑选） | 本赛是极端版本：几乎全部是弱标签。软标签 + 静默先验 + OOF 融合是讨论区验证过的路径 |
| **左右对称性利用**：动脉瘤 2nd/5th 翻转同时交换左右标签，腰椎 3rd 左右合并 ×2 数据 | 本赛可尝试：内外侧镜像并交换 MM↔LM、MedOA↔LatOA 标签（注意 MCL 无对应的 LCL 标签，镜像样本需屏蔽 MCL 损失；腓骨只在外侧，镜像图像解剖上不真实——属于需要 ablation 的想法） |
| **重增强 + CutMix**、TTA | 本赛反馈 mixup/EMA 在噪声内；可试 CutMix 与轻度缩放 TTA（1.25× 缩放不掉分） |
| **推理工程**：在时间限制下稳健跑完、无静默兜底 | 本赛有效率赛道，直接加分 |

**本赛独有**：训练目标需要自己从多语言报告构造；gold 用更严的读图阈值；测试只有图像。因此"标签工程 × 图像模型"两条线要同时推进，而前者往届几乎没有可借鉴的经验。

---

## 6. 建议路线（剩余 17 天）

| 日期 | 目标 |
|---|---|
| 10-05 ~ 10-07 | 跑 `starter/` 的 smoke test；Kaggle CPU session 分片构建 256px 缓存（约 17 GB，分两片存 private dataset）；准备标签：公开 stevenleehans v4 + 自己的 Claude/Qwen 抽取 + 正则，`merge` 取平均；看 gold AUC 只做 sanity check |
| 10-07 ~ 10-09 | ResNet34 224 5 折基线，测种子噪声；**尽早做一次真实提交**（先在 1,300 个训练 study 上演练推理），建立 OOF→LB 对应 |
| 10-09 ~ 10-13 | 单变量 ablation：切片密度/槽位、288/384 分辨率、CoAtNet/ConvNeXt/EffNetV2 主干、RadImageNet（中心切片×3）、OOF 伪标签融合（50/50，静默格 30/70）、CutMix |
| **10-15 前** | 决定是否组队（报名与组队同日截止）；组队应找**模型与标签来源不同**的队伍 |
| 10-15 ~ 10-20 | 多样化集成（rank 平均，权重用 OOF 定，不用 LB 调）；可选：融合公开 stack 但保留一个纯自有模型的提交 |
| 10-21 ~ 10-22 | 选 2 个最终提交：① CV 最好的集成；② 快速单模（兼顾效率赛道与洗牌风险） |

---

## 7. 起步代码

见 [`starter/README.md`](starter/README.md)。要点：

- `knee/dicom_io.py`：IPP 法向排序、方向规范化（矢状/冠状头朝上、轴位前方朝上）、右膝镜像为左膝（冠状/轴位左侧恒为内侧、矢状顺序恒为内→外）、140 mm 物理裁剪（不足补零），已在 3 个样例 study 上目检；
- `knee/report_labels.py`：6 种状态 → 软标签 + 静默先验 + 滑膜炎由积液填补；LLM rubric 内置主办方判读标准；多语言正则基线在 58 个 gold 上 macro AUC **0.816**（未针对 gold 调参，与讨论区报告的规则基线 0.814 一致）；
- `scripts/02_extract_labels.py claude`：Message Batches API（半价）+ 结构化 JSON 输出 + 断点续跑 + 拒答时同步重试并启用服务端 fallback；
- `knee/model.py`：timm 主干 + 槽位 embedding + 每病灶独立注意力池化 + masked-mean 残差头；归一化常数从 `pretrained_cfg` 读；
- `scripts/04_train.py`：软标签加权 BCE、GPU 增强（不翻转）、自动 bf16/fp16、OOF 与 gold 预测输出；
- `scripts/06_infer_submit.py`：多进程解码与多 GPU 推理重叠、rank 融合、逐 study 兜底并打印所有 fallback；
- 全流程已在 CPU 上用样例数据端到端跑通；GPU 训练、Claude/vLLM 后端在这里未实际运行。

**注意**：比赛数据不得再分发（规则 2.4.b），本目录 `.gitignore` 已排除 DICOM、缓存、标签 CSV 和权重；切勿把报告文本或像素提交到公开仓库。
