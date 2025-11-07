import asyncio
import base64
import json
import time
from io import BytesIO
from multiprocessing import Process, Queue

import cv2
import numpy as np
import websockets
from PIL import Image

from lane_line_detection2 import *
from traffic_sign_detection import *

# Initialize traffic sign classifier
traffic_sign_model = cv2.dnn.readNetFromONNX(
    "traffic_sign_classifier_lenet_v3.onnx")

# Queue để lưu traffic signs phát hiện được
g_signs_queue = Queue(maxsize=5)

# Queue để lưu ảnh cho visualization
g_image_queue = Queue(maxsize=5)


def process_traffic_sign_loop(g_image_queue, g_signs_queue):
    """Process chạy song song để detect traffic signs"""
    while True:
        if g_image_queue.empty():
            time.sleep(0.1)
            continue
        image = g_image_queue.get()

        # Prepare visualization image
        draw = image.copy()

        # Detect traffic signs
        signs = detect_traffic_signs(image, traffic_sign_model, draw=draw)

        # Gửi kết quả vào queue để main process sử dụng
        if not g_signs_queue.full():
            g_signs_queue.put(signs)

        # Show the result to a window
        cv2.imshow("Traffic signs", draw)
        cv2.waitKey(1)


async def process_image(websocket):
    async for message in websocket:
        # Get image from simulation
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (640, 480))

        # Prepare visualization image
        draw = image.copy()

        # Lấy thông tin biển báo từ queue (nếu có)
        signs = []
        if not g_signs_queue.empty():
            signs = g_signs_queue.get()

        # Tính toán throttle và steering với thông tin biển báo
        throttle, steering_angle = calculate_control_signal(
            image, signs=signs, draw=draw
        )

        # Update image to g_image_queue - used to run sign detection
        if not g_image_queue.full():
            g_image_queue.put(image)

        # Show the result to a window
        cv2.imshow("Result", draw)
        cv2.waitKey(1)

        # Send back throttle and steering angle
        message = json.dumps(
            {"throttle": throttle, "steering": steering_angle})
        await websocket.send(message)


async def main():
    async with websockets.serve(process_image, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # run forever


if __name__ == '__main__':
    # Khởi động process phát hiện biển báo
    p = Process(target=process_traffic_sign_loop,
                args=(g_image_queue, g_signs_queue))
    p.start()

    # Chạy main process
    asyncio.run(main())