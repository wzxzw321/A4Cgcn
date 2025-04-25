import cv2
import os

# 配置输入输出路径
video_path = "/home/ubuntu/WZX/A4C_GCN/202504221449060009CARD.avi"
output_dir = "/home/ubuntu/WZX/output_frames"

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 打开视频文件
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    raise ValueError(f"无法打开视频文件 {video_path}")

# 逐帧处理
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break

    # 仅保存每100帧（第0帧开始）
    if frame_count % 50 == 0:
        output_path = os.path.join(output_dir, f"frame_{str(frame_count).zfill(5)}.png")

        if not cv2.imwrite(output_path, frame):
            print(f"警告：第 {frame_count} 帧保存失败")

    # 进度提示保持每100帧打印一次
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"已处理 {frame_count} 帧...")

# 释放资源
video.release()
print(f"处理完成！共保存 {frame_count} 帧到 {output_dir}")