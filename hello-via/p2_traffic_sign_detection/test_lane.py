import cv2
from lane_line_detection2 import *

# Sử dụng đường dẫn tuyệt đối
image = cv2.imread(r"H:\AUTORACE\hello-via\p2_traffic_sign_detection\cua.jpg")

if image is None:
    print("Không tìm thấy ảnh! Kiểm tra lại đường dẫn.")
else:
    # Resize về đúng kích thước
    image = cv2.resize(image, (640, 480))
    
    # Tạo ảnh để vẽ kết quả
    draw = image.copy()
    
    # Tính toán
    throttle, steering_angle = calculate_control_signal(image, draw=draw)
    
    # In kết quả
    print(f"Throttle: {throttle}")
    print(f"Steering angle: {steering_angle}")
    
    # Thêm text lên ảnh
    cv2.putText(draw, f"Throttle: {throttle:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(draw, f"Steering: {steering_angle:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Hiển thị
    cv2.imshow("Original", image)
    cv2.imshow("Result", draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()