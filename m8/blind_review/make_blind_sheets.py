# -*- coding: utf-8 -*-
"""make_blind_sheets.py — 盲评表生成器（M8 任务③ 第一部分）

把 M5 的热图清单（manifest）变成三份"盲"的评审表：
  1. 每张热图分配一个随机盲号 blind_id（H001, H002, ...），
     盲号与模型名的对应关系只写进保密映射表；
  2. 三位评审（M1 / M5 / M8）各得到一份顺序不同的 CSV 表，
     表上绝不出现模型名、也不出现"该图预测对错"；
  3. （可选）把热图文件复制并改名为 H###__图片名.png，
     防止文件名里的模型名泄露身份。

用法：
  python make_blind_sheets.py --manifest heatmap_manifest.csv
                             [--out-dir sheets] [--seed 20261115]
                             [--heatmap-dir M5的热图文件夹]

manifest 列约定（M5 交接契约）：
  image_id      测试图文件名（如 IMG_8915.JPG），不泄露模型身份
  model         模型名（VGG16 / MobileNetV2 / ResNet50 / MobileNetV3-Small）
  correct       该模型在这张干净测试图上预测是否正确（1/0）
  heatmap_file  热图文件的相对路径（M5 交付）

评审表列：blind_id, image_id, heatmap_file, covers, notes
  covers 由评审填写：1=热图覆盖缺陷区域 / 0=未覆盖 / NA=无法判断
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

REVIEWERS = ("M1", "M5", "M8")


def load_manifest(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "model", "correct", "heatmap_file"}
        if not required.issubset(reader.fieldnames or []):
            sys.exit("manifest 缺少列：%s" % sorted(required - set(reader.fieldnames or [])))
        rows = list(reader)
    if not rows:
        sys.exit("manifest 为空")
    for row in rows:
        row["correct"] = int(row["correct"])
        if row["correct"] not in (0, 1):
            sys.exit("correct 必须是 0/1：%r" % row)
    # 同一张图 + 同一模型重复出现则报错（防止一张热图被评两次）
    seen = set()
    for row in rows:
        key = (row["image_id"], row["model"])
        if key in seen:
            sys.exit("manifest 里重复的热图：%s" % (key,))
        seen.add(key)
    return rows


def main():
    parser = argparse.ArgumentParser(description="生成三份盲评表 + 保密映射")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "sheets")
    parser.add_argument("--heatmap-dir", type=Path, default=None,
                        help="M5 热图所在文件夹；提供后热图会被复制改名为盲号文件")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--reviewers", default=",".join(REVIEWERS))
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    reviewers = [r.strip() for r in args.reviewers.split(",") if r.strip()]

    # 1) 在"保密乱序"下分配盲号：连续盲号不来自同一模型/同一批图
    rng = random.Random(args.seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    mapping = []
    for index, row in enumerate(shuffled, 1):
        mapping.append({
            "blind_id": "H%03d" % index,
            "image_id": row["image_id"],
            "model": row["model"],
            "correct": row["correct"],
            "heatmap_file": row["heatmap_file"],
        })

    secret_dir = args.out_dir / "_secret"
    secret_dir.mkdir(parents=True, exist_ok=True)
    with (secret_dir / "blind_mapping.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["blind_id", "image_id", "model", "correct", "heatmap_file"])
        writer.writeheader()
        writer.writerows(mapping)
    print("!! 保密映射已写入 %s" % (secret_dir / "blind_mapping.csv"))
    print("!! 评审结束前不要打开（M8 本人也是评审，评完自己那份前同样不要看）")

    # 2) （可选）复制热图并改名，消除文件名泄露
    renamed = {}
    if args.heatmap_dir:
        image_dir = args.out_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for row in mapping:
            source = args.heatmap_dir / row["heatmap_file"]
            if not source.is_file():
                print("警告：找不到 %s，该图评审表保留原文件名（请确认文件名不含模型名）" % source)
                continue
            target = image_dir / ("%s__%s" % (row["blind_id"], row["image_id"]))
            shutil.copyfile(source, target)
            renamed[row["blind_id"]] = target.name
    elif any(r["model"].lower() in r["heatmap_file"].lower() for r in mapping):
        print("!! 警告：heatmap_file 里含模型名，评审前请用 --heatmap-dir 让脚本重命名，否则盲评失效")

    # 3) 每位评审一张表：顺序不同、无模型名、无预测对错
    sheet_dir = args.out_dir
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for position, reviewer in enumerate(reviewers):
        order = mapping[:]
        random.Random(args.seed + 1000 * (position + 1)).shuffle(order)
        path = sheet_dir / ("reviewer_%s.csv" % reviewer)
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["blind_id", "image_id", "heatmap_file", "covers", "notes"])
            for row in order:
                writer.writerow([
                    row["blind_id"], row["image_id"],
                    renamed.get(row["blind_id"], row["heatmap_file"]),
                    "", "",  # covers / notes 留给评审填写
                ])
        print("评审表：%s（%d 行，顺序与其他评审不同）" % (path, len(order)))

    print("共 %d 张热图，%d 位评审。" % (len(mapping), len(reviewers)))


if __name__ == "__main__":
    main()
