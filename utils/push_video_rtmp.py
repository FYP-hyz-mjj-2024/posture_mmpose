import cv2
import subprocess
import time

# 推流器
class StreamPusher:
    def __init__(self, rtmp_url):
        # 创建FFmpeg命令行参数
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # 覆盖已存在的文件
            '-f', 'rawvideo',
            '-pixel_format', 'bgr24',
            '-video_size', '640x480',
            '-framerate', '30',  # 设置帧率
            '-i', '-',  # 从标准输入读取数据
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-f', 'flv',
            rtmp_url
        ]
        print('ffmpeg_cmd:', ffmpeg_cmd)
        # 启动 ffmpeg 并捕获输出日志
        self.ffmepg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8  # 设置较大的缓冲区
        )
        self.last_log_time = time.time()

    def streamPush(self, frame):
        try:
            # 检查帧的大小和格式
            if frame.shape != (480, 640, 3):
                raise ValueError(f"Frame size mismatch. Expected (480, 640, 3), got {frame.shape}")
            # 写入帧数据
            self.ffmepg_process.stdin.write(frame.tobytes())
            # self.ffmepg_process.stdin.flush()  # 刷新缓冲区
        except Exception as e:
            print(f"Error writing frame to stdin: {e}")

    def get_ffmpeg_logs(self):
        # 获取 ffmpeg 的输出日志
        current_time = time.time()
        if current_time - self.last_log_time > 5:  # 每5秒捕获一次日志
            self.last_log_time = current_time
            stdout, stderr = self.ffmepg_process.communicate(None, timeout=1)
            print("FFmpeg stdout:", stdout.decode('utf-8'))
            print("FFmpeg stderr:", stderr.decode('utf-8'))

rtmp_server = 'rtmp://152.42.198.96:1935/live'

# program entry
if __name__ == '__main__':
    pusher = StreamPusher(rtmp_server)
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率

    # 验证分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'Camera resolution set to {width}x{height}, FPS: {fps}')

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        # 显示帧
        cv2.imshow('frame', frame)
        # 如果按下Esc键，退出循环
        if cv2.waitKey(1) & 0xFF == 27:
            print("Press Esc, to exit.")
            break
        pusher.streamPush(frame)
        # 定期捕获 ffmpeg 的输出日志
        pusher.get_ffmpeg_logs()

    # 释放摄像头
    cap.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()
    # 获取 ffmpeg 的输出日志
    pusher.get_ffmpeg_logs()
