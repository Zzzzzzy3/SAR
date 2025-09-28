import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


# ========== 1. 改进的数据转换：DOTA -> YOLOv8-OBB ==========
def safe_imread(img_path):
    """安全读取图像，支持多种格式和中文路径"""
    try:
        # 方法1: 直接使用OpenCV
        img = cv2.imread(img_path)
        if img is not None:
            return img

        # 方法2: 使用Ultralytics的imread（支持中文路径）
        from ultralytics.utils.ops import imread
        img = imread(img_path)
        if img is not None:
            return img

        # 方法3: 使用PIL作为备选
        try:
            from PIL import Image
            img = np.array(Image.open(img_path))
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except:
            pass

        return None

    except Exception as e:
        print(f"读取图像错误 {img_path}: {e}")
        return None


def polygon_area_xy(xy):
    """xy: [x1,y1,x2,y2,x3,y3,x4,y4]  计算四边形有向面积的绝对值"""
    assert len(xy) == 8
    pts = [(xy[i], xy[i + 1]) for i in range(0, 8, 2)]
    area = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def clamp_quad(xy, w, h, eps=1e-6):
    """将四边形顶点裁剪进图像边界 [0,w) x [0,h)"""
    clamped = []
    for i in range(0, 8, 2):
        x = min(max(xy[i], 0.0), w - eps)
        y = min(max(xy[i + 1], 0.0), h - eps)
        clamped += [x, y]
    return clamped


def convert_dota_to_yoloobb(txt_path, save_path, class2id, img_dir, img_suffix=".png",
                            allow_difficult=False,
                            max_outside_ratio=0.5,
                            min_area_ratio=0.2,
                            min_abs_area=10.0):
    """
    改进版数据转换函数，增强错误处理
    """
    os.makedirs(save_path, exist_ok=True)

    # 检查源目录是否存在
    if not os.path.exists(txt_path):
        print(f"错误：标签目录不存在 - {txt_path}")
        return

    if not os.path.exists(img_dir):
        print(f"错误：图像目录不存在 - {img_dir}")
        return

    processed_files = 0
    skipped_files = 0

    for file in os.listdir(txt_path):
        if not file.endswith('.txt'):
            continue

        base = os.path.splitext(file)[0]
        img_path = os.path.join(img_dir, base + img_suffix)

        # 如果默认后缀找不到，尝试其他常见图像格式
        if not os.path.exists(img_path):
            for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP', '.tif', '.TIF']:
                alt_path = os.path.join(img_dir, base + ext)
                if os.path.exists(alt_path):
                    img_path = alt_path
                    print(f"使用备用图像格式: {alt_path}")
                    break

        if not os.path.exists(img_path):
            print(f"[警告] 找不到对应图像: {base} (尝试了{img_suffix}和其他格式)")
            skipped_files += 1
            continue

        # 使用安全的图像读取函数
        img = safe_imread(img_path)
        if img is None:
            print(f"[警告] 图像读取失败: {img_path}")
            skipped_files += 1
            continue

        h, w = img.shape[:2]

        out_lines = []
        in_file = os.path.join(txt_path, file)

        try:
            with open(in_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            try:
                with open(in_file, 'r', encoding='gbk') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"[错误] 无法读取标签文件 {in_file}: {e}")
                skipped_files += 1
                continue

        valid_objects = 0
        for li, line in enumerate(lines, 1):
            vals = line.strip().split()
            if len(vals) < 9:  # 至少 8 个坐标 + 类别
                continue

            try:
                xy = list(map(float, vals[:8]))
            except ValueError:
                print(f"[警告] 非法坐标数值: {in_file}:{li} -> {vals[:8]}")
                continue

            cls = vals[8]
            difficult = None
            if len(vals) >= 10:
                difficult = vals[9]
                if not allow_difficult and str(difficult) in ("1", "true", "True"):
                    continue

            if cls not in class2id:
                continue

            # 统计顶点越界比例
            outside = 0
            for i in range(0, 8, 2):
                xi, yi = xy[i], xy[i + 1]
                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    outside += 1
            outside_ratio = outside / 4.0
            if outside_ratio > max_outside_ratio:
                continue

            # 面积检查（裁剪前）
            area0 = polygon_area_xy(xy)
            if area0 <= 0:
                continue

            # 裁剪到边界
            clamped = clamp_quad(xy, w, h)

            # 面积检查（裁剪后）
            area1 = polygon_area_xy(clamped)
            if area1 < min_abs_area or (area1 / area0) < min_area_ratio:
                continue

            # 归一化（确保不出现 1.0）
            eps = 1e-6
            normalized = []
            for i in range(0, 8, 2):
                nx = max(0.0, min(clamped[i] / w, 1.0 - eps))
                ny = max(0.0, min(clamped[i + 1] / h, 1.0 - eps))
                normalized += [nx, ny]

            out_lines.append(
                f"{class2id[cls]} " + " ".join(f"{v:.6f}" for v in normalized) + "\n"
            )
            valid_objects += 1

        # 保存转换后的标签文件
        if out_lines:
            output_file = os.path.join(save_path, file)
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.writelines(out_lines)
                processed_files += 1
                print(f"成功转换: {file} (包含{valid_objects}个对象)")
            except Exception as e:
                print(f"[错误] 保存文件失败 {output_file}: {e}")
                skipped_files += 1
        else:
            print(f"[警告] 文件 {file} 没有有效的标注对象")
            skipped_files += 1

    print(f"数据转换完成: {txt_path} -> {save_path}")
    print(f"成功处理: {processed_files} 个文件, 跳过: {skipped_files} 个文件")


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
    with open(save_path, "w", encoding='utf-8') as f:
        f.write(yaml_str)
    print(f"YAML配置文件已生成: {save_path}")
    print(f"数据集路径: {current_dir}")


# ========== 3. 检查标注文件格式 ==========
def check_label_format(label_dir):
    """检查标注文件格式，增强错误处理"""
    if not os.path.exists(label_dir):
        print(f"[错误] 标注目录不存在: {label_dir}")
        return False

    ok = True
    total_files = 0
    valid_files = 0

    for name in os.listdir(label_dir):
        if not name.endswith('.txt'):
            continue

        total_files += 1
        file_path = os.path.join(label_dir, name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"[错误] 无法读取文件 {name}: {e}")
                ok = False
                continue

        for li, line in enumerate(lines, 1):
            vals = line.strip().split()
            if len(vals) != 9:
                print(f"[格式] {name}:{li} 列数={len(vals)} != 9")
                ok = False
                continue

            try:
                cls = int(vals[0])
                coords = list(map(float, vals[1:]))

                if any(c < 0 or c > 1 for c in coords):
                    print(f"[范围] {name}:{li} 坐标越界: {coords}")
                    ok = False

                # 简易面积检查
                pts = np.array(coords, dtype=float).reshape(4, 2)
                area = cv2.contourArea(pts.astype(np.float32))
                if area <= 1e-8:
                    print(f"[几何] {name}:{li} 面积≈0（点序/坐标异常）")
                    ok = False

            except Exception as e:
                print(f"[解析] {name}:{li} {e}")
                ok = False

        if ok:
            valid_files += 1

    print(f"标注文件检查: 总共{total_files}个文件, 有效{valid_files}个文件")
    return ok


# ========== 4. 可视化旋转框 ==========
def visualize_predictions(results, class_names, save_dir="runs/detect/vis", max_images=50):
    """可视化预测结果，限制最大图像数量"""
    os.makedirs(save_dir, exist_ok=True)
    count = 0

    for r in results:
        if count >= max_images:
            break

        try:
            img = r.orig_img.copy()
            if hasattr(r, 'obb') and r.obb is not None:
                boxes = r.obb.xyxyxyxy.cpu().numpy()
                confs = r.obb.conf.cpu().numpy()
                clss = r.obb.cls.cpu().numpy().astype(int)

                for pts, conf, cls_id in zip(boxes, confs, clss):
                    pts = pts.reshape(4, 2).astype(int)
                    color = (0, 255, 0)
                    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
                    label = f"{class_names[cls_id]} {conf:.2f}"
                    cv2.putText(img, label, (pts[0][0], pts[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            save_path = os.path.join(save_dir, os.path.basename(r.path))
            cv2.imwrite(save_path, img)
            print(f"可视化结果保存到: {save_path}")
            count += 1

        except Exception as e:
            print(f"可视化失败 {r.path}: {e}")
            continue

    print(f"共可视化 {count} 张图像")


# ========== 5. 数据预处理检查 ==========
def check_data_structure():
    """检查数据目录结构"""
    print("检查数据目录结构...")

    required_dirs = [
        "train/labelTxt",
        "val/labelTxt",
        "train/images",
        "val/images"
    ]

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ 目录不存在: {dir_path}")
            return False
        else:
            file_count = len([f for f in os.listdir(dir_path) if not f.startswith('.')])
            print(f"✓ {dir_path} (包含{file_count}个文件)")

    return True


# ========== 6. 训练 & 评估 ==========
def train_and_eval():
    class2id = {"ship": 0, "aircraft": 1, "car": 2, "tank": 3, "bridge": 4, "harbor": 5}
    id2class = {v: k for k, v in class2id.items()}

    print("=" * 50)
    print("SAR图像目标检测训练开始")
    print("图像尺寸: 256x256像素")
    print("=" * 50)

    # 检查数据目录结构
    if not check_data_structure():
        print("数据目录结构检查失败！")
        return

    # 创建labels目录
    os.makedirs("train/labels", exist_ok=True)
    os.makedirs("val/labels", exist_ok=True)

    # 检查图像文件
    def get_image_files(img_dir):
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        files = []
        for ext in extensions:
            files.extend([f for f in os.listdir(img_dir) if f.lower().endswith(ext)])
        return files

    train_images = get_image_files("train/images")
    val_images = get_image_files("val/images")

    print(f"训练图像数量: {len(train_images)}")
    print(f"验证图像数量: {len(val_images)}")

    if len(train_images) == 0 or len(val_images) == 0:
        print("错误：没有找到图像文件！")
        print("支持的格式: .png, .jpg, .jpeg, .bmp, .tif")
        return

    # 转换数据格式
    print("开始转换数据格式...")

    try:
        convert_dota_to_yoloobb("train/labelTxt", "train/labels", class2id, "train/images")
        convert_dota_to_yoloobb("val/labelTxt", "val/labels", class2id, "val/images")
    except Exception as e:
        print(f"数据转换失败: {e}")
        return

    # 检查转换后的标注文件格式
    print("检查训练集标注格式...")
    if not check_label_format("train/labels"):
        print("训练集标注文件格式检查失败！")
        # 继续执行，但给出警告

    print("检查验证集标注格式...")
    if not check_label_format("val/labels"):
        print("验证集标注文件格式检查失败！")
        # 继续执行，但给出警告

    # 生成配置文件
    write_yaml("rsar.yaml")

    # 加载模型
    print("加载YOLOv8-OBB模型...")
    try:
        model = YOLO("yolov8m-obb.pt")
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试下载模型...")
        try:
            model = YOLO("yolov8m-obb.pt", task='obb')
        except:
            print("请确保有可用的YOLOv8-OBB模型文件")
            return

    # 训练配置
    print("开始训练...")
    try:
        model.train(
            data="rsar.yaml",
            epochs=80,  # 多给一些轮次让少数类收敛
            imgsz=640,  # 提升输入分辨率，利于小目标/细节（SAR中常见）
            batch=16,
            device=0,
            optimizer="AdamW",
            lr0=0.001,
            cos_lr=True,  # 余弦退火，后期更稳
            weight_decay=0.0005,

            # == 数据增强（对 SAR 友好，色彩增强基本关掉） ==
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,  # SAR 无需 HSV
            degrees=0.0,  # SAR 中旋转常用 90° 翻转替代
            fliplr=0.5, flipud=0.5,  # 水平/垂直翻转
            translate=0.1, scale=0.5, shear=0.0,
            mosaic=0.7,  # 适度 mosaic 提高小目标可见性
            mixup=0.15,  # 少量 mixup 防过拟合
            erasing=0.2,  # 随机擦除，增强鲁棒性

            # == 损失比例 ==
            cls=1.5,  # 提高分类损失占比（默认一般是 0.5 左右）
            box=7.5,  # 框回归权重（与默认相近，按显存/收敛情况微调）
            dfl=1.5,  # 分布式回归损失权重（与默认相近）
            workers=4,
            verbose=True,
            patience=20
        )

    except Exception as e:
        print(f"训练失败: {e}")
        return

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