import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# 加载你上传的模型文件
model = YOLO('best.pt')

def generate_frames():
    # 打开摄像头 (0 通常是默认摄像头)
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 使用模型进行推理
            # stream=True 可以提高长视频流的处理效率
            results = model.predict(frame, conf=0.5, show=False)
            
            # 在帧上绘制检测结果（这是 YOLO 提供的便捷方法）
            annotated_frame = results[0].plot()

            # 将图像编码为 JPEG 格式
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            # 使用 multipart/x-mixed-replace 格式推送流
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # 渲染前端页面
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # 视频流路由
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 建议使用 threaded=True 处理多请求
    app.run(host='0.0.0.0', port=5000, debug=True)