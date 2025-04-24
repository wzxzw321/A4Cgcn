import cv2

# 打开AVI文件
video = cv2.VideoCapture("instance-segmentation.avi")

# 检查文件是否成功打开
if not video.isOpened():
    print("无法打开AVI文件")
    exit()

# 循环读取视频帧
while True:
    # 读取一帧视频
    ret, frame = video.read()

    # 检查是否读取到了帧
    if not ret:
        break

    # 显示帧
    cv2.imshow('Video', frame)

    # 按下ESC键退出
    if cv2.waitKey(1) == 27:
        break

# 释放资源
video.release()
cv2.destroyAllWindows()