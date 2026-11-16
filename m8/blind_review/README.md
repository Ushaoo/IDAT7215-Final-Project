# 盲评工具链（Blind Review Toolkit）— M8 任务③

用于量化"Grad-CAM 热图是否定位到缺陷区域"（PPT 第 13 页素材）。
三人盲评（M1、M5、M8），评审只看热图、**不知道模型名**，避免"大模型一定好"的偏见。

## 文件一览

| 文件 | 作用 |
|---|---|
| `heatmap_manifest_template.csv` | M5 交接契约模板（热图清单） |
| `make_blind_sheets.py` | 热图清单 → 三份盲评表 + 保密映射（+ 可选热图改名） |
| `stats_blind_review.py` | 三份填好的表 → 定位成功率、三人一致性、p13 柱状图 |

## M5 的交接契约（manifest 列定义）

M5 交付一个 CSV（照 `heatmap_manifest_template.csv` 填）：

| 列 | 说明 |
|---|---|
| `image_id` | 测试图文件名，如 `IMG_8915.JPG` |
| `model` | 模型名：VGG16 / MobileNetV2 / ResNet50 / MobileNetV3-Small |
| `correct` | 该模型在这张干净测试图上预测对（1）/ 错（0）——用于"判对图 vs 判错图"分组统计，**不会出现在评审表上** |
| `heatmap_file` | 热图文件相对路径（M5 的交付文件） |

约定：每模型抽 20–30 张，判对与判错都要有；四个模型同一批图（方便横向比较）。

## 使用流程（真实执行时）

```bash
# 第 1 步：M5 给 manifest + 热图文件夹后，M8 生成盲评表
python make_blind_sheets.py --manifest M5的清单.csv --heatmap-dir M5的热图文件夹

# 第 2 步：把 sheets/reviewer_M1.csv、reviewer_M5.csv、reviewer_M8.csv
#         分别发给三人（M8 自己留一份），各独立评 30 分钟
#         填写 covers 列：1=热图覆盖缺陷区域 / 0=未覆盖 / NA=无法判断

# 第 3 步：收齐三份表后统计
python stats_blind_review.py
```

### 评审须知（发给三人）

1. 独立评审，**评审期间不讨论、不互看答案**；
2. 表上只有盲号（H001…）、图片名和热图文件名，**没有模型名**——也不要问 M8 要对照表；
3. 判断标准：热图的高亮区域是否覆盖/落在**缺陷区域**上（不是"覆盖整个零件"，而是"看对了缺陷的位置"）；
4. 看不清就填 `NA`（无法判断），不要硬猜——NA 会单独统计，不影响"成功率"分母。

### 保密约定

- 盲号→模型名的对照表在 `sheets/_secret/blind_mapping.csv`，**评完自己那份前不要打开**（M8 本人也是评审）。保险做法：生成后请 M1 保管，统计时再取回。
- 若 M5 的热图文件名含模型名（如 `vgg16_xxx.png`），务必用 `--heatmap-dir` 让脚本复制改名为 `H###__图片名.png`，否则盲评失效（脚本会给出警告）。

## 统计口径（写进 PPT/报告时直接引用）

- **定位成功率** = 评审判"是"数 ÷（"是"+"否"数）；`NA` 不计入分母、单独报告
- **判对图 vs 判错图分组**：模型"预测对了"的图上定位如何、"预测错了"的图上又如何（回答"准确率高但看不看对地方"）
- **三人一致性**：三人全一致的图占比；两两一致率 + Cohen's kappa（sklearn）
- **多数结论**：每张热图 covered / not covered / no consensus（票数打平）

## 输出文件（results/）

- `blind_review_stats.csv` — 每模型成功率（PPT 第 13 页统计表素材）
- `agreement_stats.csv` — 一致性指标
- `per_image_votes.csv` — 每张热图三人投票明细（附录/答辩备查）
- `p13_localization_rate.png` — PPT 第 13 页柱状图（风格与 p05 数据卡片图一致）

## 模型配色提案（请 M6 全局确认）

VGG16 = 蓝 `#2a78d6` · MobileNetV2 = 橙 `#eb6834` · ResNet50 = 青 `#1baf7a` · MobileNetV3-Small = 黄 `#eda100`。
建议 p10/p11/p12 全 PPT 统一沿用这组模型色（已通过色盲可读性校验）。

## 本项目实际执行记录（2026-11-16）

三份评审表已回收（M1 用 0/1、M5 与 M8 用 yes/no），统一到 1=yes 后统计，
结果存放在 `../results/`：

| 文件 | 内容 |
|---|---|
| `blind_review_stats.csv` | 每模型定位成功率（yes / (yes+no)），NA 单独计数 |
| `agreement_stats.csv` | 三人完全一致率、两两一致率、Cohen's κ |
| `per_image_votes.csv` | 每张热图的三人原始票与多数结论 |
| `../figures/p13_localization_rate.png` | PPT 第 13 页柱状图（脚本自动复制） |
| `blind_key.csv` | 解盲键（blind_id → model / 原图名 / 预测对错）；盲评结束后公开 |
| `review_M1.csv` / `review_M5.csv` / `review_M8.csv` | 三份原始评审表（审计材料） |

复现命令：

```bash
python m8/blind_review/stats_blind_review.py \
  --sheets-dir m8/results --mapping m8/results/blind_key.csv --out-dir m8/results
```

编码说明：M1 交回 0/1，M5 与 M8 用 yes/no。1=yes 的极性经实证核验——
按 1=yes 解释时 M1 与 M8 一致率 93.8%，按 1=no 解释时仅 6.2%。
`parse_vote()` 同时接受 1/0 与 yes/no，无需手工改表。
