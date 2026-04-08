#!/usr/bin/env python3
"""
快速诊断脚本
运行: python diagnose.py
检查: SAM版本、bag话题、首帧RGB+点云预览、内参
"""
import sys, os
import numpy as np
 
print("=" * 60)
print("  环境诊断")
print("=" * 60)
 
# 1. Python & 包版本
import importlib
for pkg in ["torch", "open3d", "cv2", "numpy", "PIL"]:
    try:
        m = importlib.import_module(pkg if pkg != "PIL" else "PIL.Image")
        ver = getattr(m, "__version__", "?")
        print(f"  ✅ {pkg}: {ver}")
    except ImportError:
        print(f"  ❌ {pkg}: 未安装")
 
# 2. SAM检测
print("\n--- SAM检测 ---")
sam2_ok = False
sam1_ok = False
try:
    from sam2.build_sam import build_sam2_video_predictor
    sam2_ok = True
    print("  ✅ SAM2: 可用")
except ImportError as e:
    print(f"  ❌ SAM2: {e}")
 
try:
    from segment_anything import sam_model_registry
    sam1_ok = True
    print("  ✅ SAM1: 可用")
    # 检查checkpoint
    ckpt = "sam_vit_h_4b8939.pth"
    if os.path.exists(ckpt):
        size_gb = os.path.getsize(ckpt) / 1e9
        print(f"  ✅ SAM1 checkpoint: {ckpt} ({size_gb:.1f}GB)")
    else:
        print(f"  ❌ SAM1 checkpoint 不存在: {ckpt}")
except ImportError as e:
    print(f"  ❌ SAM1: {e}")
 
# 3. CUDA
print("\n--- CUDA检测 ---")
try:
    import torch
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"  计算能力: sm_{cap[0]}{cap[1]}")
        if cap[0] >= 12:
            print("  ⚠️  sm_120 (RTX 50系) 与旧版CUDA扩展不兼容，建议用CPU")
except:
    pass
 
# 4. Bag读取测试
print("\n--- Bag诊断 ---")
bag_path = "duck_1_0.db3"
if not os.path.exists(bag_path):
    print(f"  ❌ bag文件不存在: {bag_path}")
else:
    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        import rosbag2_py
 
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions("", "")
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
 
        # 读取第一帧RGB和点云
        got_rgb = False
        got_pcd = False
        first_rgb = None
        first_pcd = None
        rgb_shape = None
        pcd_info = {}
 
        while reader.has_next() and not (got_rgb and got_pcd):
            topic, raw, ts = reader.read_next()
 
            if topic == "/camera/color/image_raw" and not got_rgb:
                msg = deserialize_message(raw, get_message(topic_types[topic]))
                print(f"  ✅ RGB: {msg.width}x{msg.height}, encoding={msg.encoding}")
                rgb_shape = (msg.height, msg.width)
                got_rgb = True
                # 解码
                data = np.frombuffer(msg.data, dtype=np.uint8)
                first_rgb = data.reshape(msg.height, msg.width, 3)
 
            elif topic == "/camera/depth/points" and not got_pcd:
                msg = deserialize_message(raw, get_message(topic_types[topic]))
                n = msg.width * msg.height
                fields = [f.name for f in msg.fields]
                print(f"  ✅ PointCloud2: {n}点, 字段={fields}, point_step={msg.point_step}")
                pcd_info = {"n": n, "fields": fields, "step": msg.point_step}
                got_pcd = True
 
        if not got_rgb:
            print("  ❌ 未找到 /camera/color/image_raw")
        if not got_pcd:
            print("  ❌ 未找到 /camera/depth/points")
            print("  💡 可用话题:")
            for t in topic_types:
                print(f"      {t}")
 
        # 5. 保存第一帧RGB预览
        if first_rgb is not None:
            import cv2
            out_path = "diagnose_first_frame.jpg"
            bgr = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, bgr)
            print(f"\n  💾 第一帧RGB已保存: {out_path}")
            print(f"     请查看图像，找到鸭子的大概像素坐标用于标注")
 
    except Exception as e:
        print(f"  ❌ Bag读取失败: {e}")
        import traceback; traceback.print_exc()
 
print("\n" + "=" * 60)
print("  诊断完成")
if sam2_ok:
    print("  推荐命令:")
    print("  python duck_sam2_pipeline.py \\")
    print("    --bag duck_1_0.db3 \\")
    print("    --sam2_checkpoint <your_sam2.pt> \\")
    print("    --output duck_clean_v2.pcd \\")
    print("    --skip_frames 2 --save_masks")
elif sam1_ok:
    print("  SAM2未安装，使用SAM1模式:")
    print("  python duck_sam2_pipeline.py \\")
    print("    --bag duck_1_0.db3 \\")
    print("    --sam1_checkpoint sam_vit_h_4b8939.pth \\")
    print("    --output duck_clean_v2.pcd \\")
    print("    --skip_frames 2 --save_masks")
    print("\n  安装SAM2 (强烈推荐):")
    print("  pip install git+https://github.com/facebookresearch/sam2.git")
    print("  # 下载权重: https://github.com/facebookresearch/sam2#model-checkpoints")
print("=" * 60)
