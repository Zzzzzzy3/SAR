import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


# ========== 1. 数据转换：DOTA -> YOLOv8-OBB ==========
import os, cv2
import math

def polygon_area_xy(xy):
    """xy: [x1,y1,x2,y2,x3,y3,x4,y4]  计算四边形有向面积的绝对值"""
    assert len(xy) == 8
    pts = [(xy[i], xy[i+1]) for i in range(0, 8, 2)]
    area = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i+1) % 4]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5

def clamp_quad(xy, w, h, eps=1e-6):
    """将四边形顶点裁剪进图像边界 [0,w) x [0,h)"""
    clamped = []
    for i in range(0, 8, 2):
        x = min(max(xy[i], 0.0), w - eps)
        y = min(max(xy[i+1], 0.0), h - eps)
        clamped += [x, y]
    return clamped

def convert_dota_to_yoloobb(txt_path, save_path, class2id, img_dir, img_suffix=".png",
                             allow_difficult=False,
                             max_outside_ratio=0.5,     # 顶点越界比例阈值（> 则丢弃）
                             min_area_ratio=0.2,        # 裁剪后面积 / 原面积 < 则丢弃
                             min_abs_area=10.0):        # 裁剪后绝对面积过小则丢弃（像素^2）
    """
    - 遇到越界点：先统计比例，大于 max_outside_ratio 丢弃；否则 clamp 到边界
    - clamp 后用面积检查，过小或退化则丢弃
    """
    os.makedirs(save_path, exist_ok=True)

    for file in os.listdir(txt_path):
        if not file.endswith('.txt'):
            continue

        base = os.path.splitext(file)[0]
        img_path = os.path.join(img_dir, base + img_suffix)
        if not os.path.exists(img_path):
            print(f"[警告] 找不到对应图像: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[警告] 图像读取失败: {img_path}")
            continue
        h, w = img.shape[:2]

        out_lines = []
        in_file = os.path.join(txt_path, file)
        with open(in_file, 'r', encoding='utf-8') as f:
            for li, line in enumerate(f, 1):
                vals = line.strip().split()
                if len(vals) < 9:   # 至少 8 个坐标 + 类别
                    continue

                try:
                    xy = list(map(float, vals[:8]))
                except ValueError:
                    print(f"[警告] 非法坐标数值: {in_file}:{li} -> {vals[:8]}")
                    continue

                cls = vals[8]
                difficult = None
                if len(vals) >= 10:
                    # DOTA 有时第 10 列为 difficult(0/1)，或 crowd 等
                    difficult = vals[9]
                    if not allow_difficult and str(difficult) in ("1", "true", "True"):
                        continue

                if cls not in class2id:
                    # 未知类别直接跳过
                    continue

                # 统计顶点越界比例
                outside = 0
                for i in range(0, 8, 2):
                    xi, yi = xy[i], xy[i+1]
                    if xi < 0 or xi >= w or yi < 0 or yi >= h:
                        outside += 1
                outside_ratio = outside / 4.0
                if outside_ratio > max_outside_ratio:
                    # 越界太多，直接忽略该目标
                    # 你也可以选择记录到一个日志文件里
                    # print(f"[信息] 越界比例 {outside_ratio:.2f}，丢弃: {in_file}:{li}")
                    continue

                # 面积检查（裁剪前）
                area0 = polygon_area_xy(xy)
                if area0 <= 0:
                    # 极小面积或退化
                    continue

                # 裁剪到边界
                clamped = clamp_quad(xy, w, h)

                # 面积检查（裁剪后）
                area1 = polygon_area_xy(clamped)
                # 如果裁剪后面积过小或相对面积过低，跳过
                if area1 < min_abs_area or (area1 / area0) < min_area_ratio:
                    continue

                # 归一化（确保不出现 1.0）
                eps = 1e-6
                normalized = []
                for i in range(0, 8, 2):
                    nx = max(0.0, min(clamped[i]   / w, 1.0 - eps))
                    ny = max(0.0, min(clamped[i+1] / h, 1.0 - eps))
                    normalized += [nx, ny]

                out_lines.append(
                    f"{class2id[cls]} " + " ".join(f"{v:.6f}" for v in normalized) + "\n"
                )

        with open(os.path.join(save_path, file), 'w', encoding='utf-8') as f:
            f.writelines(out_lines)

    print(f"数据转换完成: {txt_path} -> {save_path}")


# ========== 2. 生成数据配置文件 ==========
def write_yaml(save_path="rsar.yaml"):
    # 使用绝对路径确保YOLO能找到数据
    current_dir = os.getcwd().replace('\\', '/')

    yaml_str = f"""# RSAR SAR目标检测数据集 (YOLOv8-OBB)
path: {current_dir}
train: train/images
val: val/images

names:
  0: ship
  1: aircraft
  2: car
  3: tank
  4: bridge
  5: harbor
"""
    with open(save_path, "w") as f:
        f.write(yaml_str)
    print(f"YAML配置文件已生成: {save_path}")
    print(f"数据集路径: {current_dir}")


# ========== 3. 检查标注文件格式 ==========
def check_label_format(label_dir):
    ok = True
    a=0
    for name in os.listdir(label_dir):
        if not name.endswith('.txt'): continue
        with open(os.path.join(label_dir, name)) as f:
            for li, line in enumerate(f, 1):
                vals = line.strip().split()
                if len(vals) != 9:
                    print(f"[格式] {name}:{li} 列数={len(vals)} != 9")
                    ok = False; continue
                try:
                    cls = int(vals[0]); coords = list(map(float, vals[1:]))
                    if any(c<0 or c>1 for c in coords):
                        print(f"[范围] {name}:{li} 坐标越界")
                        print(coords)
                        ok = False
                    # 简易面积检查
                    pts = np.array(coords, dtype=float).reshape(4,2)
                    area = cv2.contourArea(pts.astype(np.float32))
                    if area <= 1e-8:
                        print(f"[几何] {name}:{li} 面积≈0（点序/坐标异常）")
                        ok = False
                except Exception as e:
                    print(f"[解析] {name}:{li} {e}"); ok = False
    return ok


# ========== 4. 可视化旋转框 ==========
def visualize_predictions(results, class_names, save_dir="runs/detect/vis"):
    os.makedirs(save_dir, exist_ok=True)
    for r in results:
        img = r.orig_img.copy()
        if hasattr(r, 'obb') and r.obb is not None:
            boxes = r.obb.xyxyxyxy.cpu().numpy()
            confs = r.obb.conf.cpu().numpy()
            clss = r.obb.cls.cpu().numpy().astype(int)

            for pts, conf, cls_id in zip(boxes, confs, clss):
                pts = pts.reshape(4, 2).astype(int)
                color = (0, 255, 0)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1)
                label = f"{class_names[cls_id]} {conf:.2f}"
                cv2.putText(img, label, (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        save_path = os.path.join(save_dir, os.path.basename(r.path))
        cv2.imwrite(save_path, img)
        print(f"可视化结果保存到: {save_path}")


# ========== 6. 训练 & 评估 ==========
def train_and_eval():
    class2id = {"ship": 0, "aircraft": 1, "car": 2, "tank": 3, "bridge": 4, "harbor": 5}
    id2class = {v: k for k, v in class2id.items()}

    print("=" * 50)
    print("SAR图像目标检测训练开始")
    print("图像尺寸: 256x256像素")
    print("=" * 50)

    # 检查数据目录
    print("检查数据目录...")
    required_dirs = [
        "train/labelTxt",
        "val/labelTxt",
        "train/images",
        "val/images"
    ]

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"错误：目录不存在 - {dir_path}")
            return
        print(f"✓ {dir_path}")

    # 创建labels目录
    os.makedirs("train/labels", exist_ok=True)
    os.makedirs("val/labels", exist_ok=True)

    # 检查图像文件
    train_images = [f for f in os.listdir("train/images") if f.endswith('.png')]
    val_images = [f for f in os.listdir("val/images") if f.endswith('.png')]

    print(f"训练图像数量: {len(train_images)}")
    print(f"验证图像数量: {len(val_images)}")

    if len(train_images) == 0 or len(val_images) == 0:
        print("错误：没有找到PNG图像文件！")
        return

    # 转换数据格式 - 使用正确的YOLOv8-OBB格式（归一化坐标）
    print("开始转换数据格式...")

    convert_dota_to_yoloobb("train/labelTxt", "train/labels", class2id,"train/images")
    convert_dota_to_yoloobb("val/labelTxt", "val/labels", class2id,"val/images" )

    # 检查转换后的标注文件格式
    if not check_label_format("train/labels"):
        print("标注文件格式检查失败！")
        return
    if not check_label_format("val/labels"):
        print("验证集标注文件格式检查失败！")
        return
    # 生成配置文件
    write_yaml("rsar.yaml")

    # 加载模型
    print("加载YOLOv8-OBB模型...")
    try:
        model = YOLO("yolov8m-obb.pt")
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 训练配置
    print("开始训练...")
    model.train(
        data="rsar.yaml",
        epochs=50,  # 减少epoch数以便快速测试
        imgsz=256,
        batch=16,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        workers=4,
        verbose=True
    )

    # 验证
    print("开始验证...")
    try:
        metrics = model.val()
        print("评估结果：")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
    except Exception as e:
        print(f"验证失败: {e}")

    # 推理测试
    print("进行推理测试...")
    try:
        results = model.predict(
            source="val/images",
            save=True,
            imgsz=256,
            conf=0.3,
            device=0
        )
        print("推理完成")

        # 可视化
        visualize_predictions(results, id2class)

    except Exception as e:
        print(f"推理失败: {e}")


    print("=" * 50)
    print("训练完成！")
    print("=" * 50)


if __name__ == "__main__":
    train_and_eval()