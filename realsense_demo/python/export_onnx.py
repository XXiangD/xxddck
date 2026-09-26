from ultralytics import YOLO

# 修改为实际的 best.pt 路径
model = YOLO("/home/humble/volleyball1/runs/detect/volleyball_m_project/5060_high_res-4/weights/best.pt")

output_path = model.export(
    format="onnx",
    imgsz=960,
    batch=1,
    dynamic=False,
    simplify=True,
    opset=12,
    half=False,
    nms=False,
)

print("导出文件位置：", output_path)