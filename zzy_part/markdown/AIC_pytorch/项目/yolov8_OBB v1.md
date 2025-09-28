```python
import os                           # 标准库：文件/目录、路径拼接、创建目录等  
import cv2                          # OpenCV：图像读写、绘制可视化（在这里用来画旋转框、多边形）  
import numpy as np                  # 数值计算库（这里基本用于数组类型转换）  
import matplotlib.pyplot as plt     # 可视化：绘制训练曲线（Loss / mAP）  
import pandas as pd                 # 读取results.csv并处理数据帧  
from ultralytics import YOLO        # Ultralytics YOLO 类（v8系列，含OBB）  
  
# ========== 1. 数据转换：DOTA -> YOLOv8-OBB ==========  
def convert_dota_to_yoloobb(txt_path, save_path, class2id, img_width=256, img_height=256):  
    os.makedirs(save_path, exist_ok=True)                    # 如果输出目录不存在则创建  
    for file in os.listdir(txt_path):                        # 遍历标注目录中的所有文件  
        if not file.endswith('.txt'):                        # 只处理txt标注文件  
            continue  
        with open(os.path.join(txt_path, file), 'r') as f:   # 读入单个标注文件  
            lines = f.readlines()  
        out = []                                             # 准备输出的YOLO-OBB行（字符串列表）  
        for line in lines:                                   # 遍历每一行目标标注  
            vals = line.strip().split()                      # 去空白并按空格分列  
            if len(vals) < 9:                                # DOTA格式至少应为8个点坐标+类别名  
                continue  
            x = list(map(float, vals[:8]))                   # 前8列是四个点(x1,y1,...,x4,y4)  
            cls = vals[8]                                    # 第9列是类别字符串  
            if cls not in class2id:                          # 若类别不在映射字典中则跳过  
                continue  
  
            # 归一化坐标到0-1范围（偶数索引是x，用宽度除；奇数索引是y，用高度除）  
            normalized_x = [coord / img_width if i % 2 == 0 else coord / img_height  
                            for i, coord in enumerate(x)]  
  
            # YOLOv8-OBB格式：class_id x1 y1 x2 y2 x3 y3 x4 y4（全部0~1）  
            out.append(  
                f"{class2id[cls]} {normalized_x[0]:.6f} {normalized_x[1]:.6f} {normalized_x[2]:.6f} {normalized_x[3]:.6f} {normalized_x[4]:.6f} {normalized_x[5]:.6f} {normalized_x[6]:.6f} {normalized_x[7]:.6f}\n")  
  
        with open(os.path.join(save_path, file), 'w') as f:  # 将该标注文件转换后的所有行写回新目录  
            f.writelines(out)  
    print(f"数据转换完成: {txt_path} -> {save_path}")        # 打印转换完成信息  
  
# ========== 2. 生成数据配置文件 ==========def write_yaml(save_path="rsar.yaml"):  
    # 使用绝对路径确保YOLO能找到数据  
    current_dir = os.getcwd().replace('\\', '/')             # 获取当前工作目录并统一分隔符  
  
    yaml_str = f"""# RSAR SAR目标检测数据集 (YOLOv8-OBB)path: {current_dir}  
train: train/images  
val: val/images  
  
names:  
  0: ship  1: aircraft  2: car  3: tank  4: bridge  5: harbor"""  
    with open(save_path, "w") as f:                          # 将上面的多行字符串写为数据配置yaml  
        f.write(yaml_str)  
    print(f"YAML配置文件已生成: {save_path}")                 # 提示生成成功  
    print(f"数据集路径: {current_dir}")                      # 打印数据根路径  
  
# ========== 3. 检查标注文件格式 ==========def check_label_format(label_dir):  
    """检查标注文件格式是否正确"""  
    print("检查标注文件格式...")  
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]  # 找到所有txt标注  
  
    if not label_files:                                      # 若目录下没有标注，直接返回失败  
        print("没有找到标注文件！")  
        return False  
  
    for label_file in label_files[:3]:                        # 只抽查前三个文件  
        with open(os.path.join(label_dir, label_file), 'r') as f:  
            lines = f.readlines()  
            if lines:                                        # 若文件非空  
                first_line = lines[0].strip()                # 取第一行做检查  
                values = first_line.split()  
                print(f"文件: {label_file}, 第一行: {first_line}")  
                print(f"列数: {len(values)}")  
  
                # 检查是否为9列（class + 8个坐标）  
                if len(values) != 9:  
                    print(f"错误：应该有9列，实际有{len(values)}列")  
                    return False  
  
                # 检查坐标值是否在0-1范围内（此处默认已是YOLO归一化格式）  
                try:  
                    coords = list(map(float, values[1:]))    # 跳过class_id，仅检查8个坐标值  
                    for i, coord in enumerate(coords):  
                        if coord < 0 or coord > 1:  
                            print(f"警告：坐标 {coord} 不在0-1范围内 (位置 {i + 1})")  
                    print("坐标范围检查完成")  
                except ValueError:                           # 若转换浮点失败，则格式不正确  
                    print("错误：坐标值不是有效的浮点数")  
                    return False  
  
    return True                                              # 通过抽查则认为基本正确  
  
# ========== 4. 可视化旋转框 ==========def visualize_predictions(results, class_names, save_dir="runs/detect/vis"):  
    os.makedirs(save_dir, exist_ok=True)                     # 确保输出目录存在  
    for r in results:                                        # 遍历预测结果列表（Ultralytics返回的对象）  
        img = r.orig_img.copy()                              # 拿到原图（numpy数组）  
        if hasattr(r, 'obb') and r.obb is not None:          # 确认存在OBB结果（旋转框）  
            boxes = r.obb.xyxyxyxy.cpu().numpy()             # 取四点坐标（x1y1...x4y4），形状(N,8)  
            confs = r.obb.conf.cpu().numpy()                 # 置信度 (N,)            clss = r.obb.cls.cpu().numpy().astype(int)       # 类别索引 (N,)  
            for pts, conf, cls_id in zip(boxes, confs, clss):# 逐个目标绘制  
                pts = pts.reshape(4, 2).astype(int)          # 8值->(4,2)四个点，并转为整型像素坐标  
                color = (0, 255, 0)                          # 绿色轮廓  
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1)   # 画四边形  
                label = f"{class_names[cls_id]} {conf:.2f}"  # 文字标签：类别名+置信度  
                cv2.putText(img, label, (pts[0][0], pts[0][1] - 5),  
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)  # 写字到图上  
  
        save_path = os.path.join(save_dir, os.path.basename(r.path))  # 输出图像路径（沿用原文件名）  
        cv2.imwrite(save_path, img)                         # 保存可视化图  
        print(f"可视化结果保存到: {save_path}")             # 打印保存位置  
  
# ========== 5. 绘制 Loss 和 mAP 曲线 ==========def plot_training_curves(csv_path="runs/detect/train/weights/results.csv", save_path="runs/detect/train/weights/curves.png"):  
    if not os.path.exists(csv_path):                         # 若找不到结果CSV，直接返回  
        print(f"未找到 {csv_path}，无法绘制曲线")  
        return  
    df = pd.read_csv(csv_path)                               # 读入训练过程的metrics CSV  
  
    plt.figure(figsize=(12, 5))                              # 新建画布  
  
    # Loss 曲线  
    plt.subplot(1, 2, 1)                                     # 左子图：Loss  
    if "train/box_loss" in df.columns:  
        plt.plot(df["epoch"], df["train/box_loss"], label="box_loss")  
    if "train/cls_loss" in df.columns:  
        plt.plot(df["epoch"], df["train/cls_loss"], label="cls_loss")  
    if "train/dfl_loss" in df.columns:  
        plt.plot(df["epoch"], df["train/dfl_loss"], label="dfl_loss")  
    if "train/obb_loss" in df.columns:  
        plt.plot(df["epoch"], df["train/obb_loss"], label="obb_loss")  
    plt.xlabel("Epoch")  
    plt.ylabel("Loss")  
    plt.title("Training Loss")  
    plt.legend()  
    plt.grid(True)  
  
    # mAP 曲线  
    plt.subplot(1, 2, 2)                                     # 右子图：mAP  
    if "metrics/mAP50" in df.columns:  
        plt.plot(df["epoch"], df["metrics/mAP50"], label="mAP@0.5")  
    if "metrics/mAP50-95" in df.columns:  
        plt.plot(df["epoch"], df["metrics/mAP50-95"], label="mAP@0.5:0.95")  
    plt.xlabel("Epoch")  
    plt.ylabel("mAP")  
    plt.title("Validation mAP")  
    plt.legend()  
    plt.grid(True)  
  
    plt.tight_layout()                                       # 子图布局优化  
    plt.savefig(save_path)                                   # 保存曲线图  
    plt.show()                                               # 显示图像（交互环
```

## `vals[:8]`

- 假设 `vals` 是一个列表，例如：
    
    `vals = ["377", "181", "463", "177", "465", "223", "379", "228", "ship", "0"]`
    
- `vals[:8]` 表示取列表的前 **8 个元素**（不包括第 9 个），结果是：
    
    `["377", "181", "463", "177", "465", "223", "379", "228"]`
    
## 2. `map(float, ...)`

- `map()` 会把某个函数应用到列表中的每个元素。
    
- `map(float, ["377", "181", ...])` 的意思是：把列表里的字符串依次转成 `float` 类型。
    
- 结果是一个 **map对象**（迭代器），里面的元素类似：

    `[377.0, 181.0, 463.0, 177.0, 465.0, 223.0, 379.0, 228.0]`


`labels.cache` 是在一些深度学习框架（例如 YOLO 或其他目标检测模型）训练过程中生成的一个缓存文件，通常是为了加速数据加载过程。它的作用是缓存标注信息，避免每次训练时都重新读取和解析标注文件。

### 作用和用途

1. **加速数据加载**：
    
    - 在目标检测任务中，每个图像通常有一个与之对应的标注文件（例如，`txt` 文件）。这些标注文件包含了物体的边界框（Bounding Boxes）以及类别标签等信息。每次训练时，如果每个图像的标注文件都要重新读取、解析、转换成模型可以理解的格式，会消耗很多时间。
        
    - `labels.cache` 文件的生成，可以将这些标注信息缓存起来，这样在后续的训练中，程序就不需要每次都重新解析标注文件，而是直接从缓存中加载，从而加快数据加载速度。
        
2. **减少 I/O 操作**：
    
    - 读取大量的小文件会增加磁盘的 I/O 操作，尤其是当数据集非常大时。使用缓存文件可以减少对硬盘的访问，提高训练效率。
        
3. **避免重复计算**：
    
    - 如果数据集或者标注信息没有发生变化，那么缓存文件可以在多个训练过程中复用，避免每次都进行重复计算。
        

### 生成的方式

`labels.cache` 通常在以下情况下生成：

- **首次读取数据集时**：如果你使用的是像 YOLOv5 等模型框架，它们会在第一次加载数据时创建这个缓存文件。
    
- **数据集标注变化时**：如果你更新了数据集或标注，缓存文件可能会重新生成。
    

### 文件内容

- `labels.cache` 文件中存储的是已解析的标注信息，通常包括每张图片的标注数据，例如边界框坐标、类别标签、图像路径等。具体格式可能会因框架不同而有所差异，但通常是一个经过序列化的数据格式（例如 `pickle`、`json` 或者自定义的二进制格式）。
    

### 示例：

假设你使用 **YOLOv5**，第一次加载数据时，它会根据数据集中的标注文件生成一个 `labels.cache` 文件，用来加速后续的数据加载。

### 是否可以删除？

- **可以删除**：删除 `labels.cache` 文件通常不会对训练造成太大影响。删除之后，系统会在下一次加载数据时重新生成该文件。
    
- **可能会影响加载速度**：如果删除了缓存文件，第一次重新训练时，数据加载可能会变慢，因为要重新解析标注信息。
    

### 总结：

`labels.cache` 是用来缓存数据集标注信息的文件，目的是加速后续的训练过程。它通常在第一次训练时生成，并且可以在后续训练中复用。如果你觉得它不再需要或者希望重新生成缓存，完全可以删除它，系统会自动重新创建。
![[Pasted image 20250921103438.png]]

2000-->map50 : 0.42
### 1. 核心概念：多任务损失函数

YOLOv8 是一个多任务学习模型，它的总损失函数由三部分组成：

1. **边界框回归损失**：衡量预测的边界框（Bounding Box）与真实框（Ground Truth）的位置和尺寸差异。
    
2. **分类损失**：衡量预测的物体类别与真实类别的差异。
    
3. **分布焦点损失（DFL）**：一种更精细的边界框回归损失，用于提高边界框坐标的准确性（主要针对 YOLOv8 的边框表示方法）。
    

总损失函数大致可以表示为：  
`Total_Loss = w_box * L_box + w_cls * L_cls + w_dfl * L_dfl`

这里的 `w_box`， `w_cls`， `w_dfl` 就是各个损失部分的权重。

### 2. `cls=1.0` 的含义

因此，`cls=1.0` 意味着在计算总损失时，**分类损失部分（L_cls）的权重是 1.0**。

- **权重的作用**：这个权重决定了分类损失在总损失中的“重要性”或“贡献度”。
    
- **默认值**：在 YOLOv8 的默认配置中，`cls` 权重通常被设置为 `1.0`，边界框损失 `box` 的权重通常为 `7.5`，DFL损失 `dfl` 的权重通常为 `1.5`。
    

### 3. 为什么权重不同？

从默认的权重（`box=7.5`, `cls=1.0`, `dfl=1.5`）可以看出，**YOLO 模型更侧重于精确地定位物体（边界框回归），而不是仅仅正确地分类它**。

- **高 `box` 权重**：一个检测框位置不准但分类正确（例如，框住了狗的一半身体但正确识别为“狗”），通常比一个框位置很准但分类错误（例如，完美框住一只猫但识别为“狗”）要好。因此，模型通过提高 `box` 损失的权重，来优先学习如何画好框。
    
- **`cls=1.0`**：分类损失仍然非常重要，但它与定位任务处于一个相对平衡的水平。