import cv2
import numpy as np
from collections import deque

# Lưu giá trị lane trước đó để dự đoán nếu mất lane
prev_left = deque(maxlen=5)
prev_right = deque(maxlen=5)

# Biến theo dõi trạng thái mất line
lost_line_counter = 0
LOST_LINE_THRESHOLD = 10  # Số frame liên tiếp mất line thì kích hoạt xử lý đặc biệt

# Lưu lịch sử góc lái để tính độ biến thiên
prev_steering_angles = deque(maxlen=5)


def find_lane_lines(img):
    """Phát hiện vạch kẻ đường bằng Canny"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 100, 200)
    return edges


def birdview_transform(img):
    """Chuyển đổi sang góc nhìn từ trên xuống (bird-view)"""
    IMAGE_H, IMAGE_W = img.shape[:2]
    src = np.float32([
        [IMAGE_W * 0.1, IMAGE_H * 0.95],
        [IMAGE_W * 0.9, IMAGE_H * 0.95],
        [IMAGE_W * 0.35, IMAGE_H * 0.6],
        [IMAGE_W * 0.65, IMAGE_H * 0.6]
    ])
    dst = np.float32([
        [200, IMAGE_H],
        [IMAGE_W - 200, IMAGE_H],
        [200, 0],
        [IMAGE_W - 200, 0]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
    return warped


def find_left_right_points(image, draw=None):
    """Tìm vị trí 2 vạch lane (dự đoán nếu chỉ có 1 vạch)"""
    im_height, im_width = image.shape[:2]
    interested_line_y = int(im_height * 0.9)
    interested_line = image[interested_line_y, :]

    # khởi tạo mặc định
    left_point, right_point = -1, -1
    lane_width_est = 250  # khoảng cách trung bình giữa 2 lane
    center = im_width // 2

    # Quét từ giữa sang hai bên
    for x in range(center, 0, -1):
        if interested_line[x] > 0:
            left_point = x
            break
    for x in range(center + 1, im_width):
        if interested_line[x] > 0:
            right_point = x
            break

    # --- Dự đoán lane còn lại ---
    # Chỉ thấy bên trái
    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width_est

    # Chỉ thấy bên phải
    if right_point != -1 and left_point == -1:
        left_point = right_point - lane_width_est

    # Nếu cả hai đều mất → dùng giá trị trước đó
    if left_point == -1 and len(prev_left) > 0:
        left_point = int(np.mean(prev_left))
    if right_point == -1 and len(prev_right) > 0:
        right_point = int(np.mean(prev_right))

    # Cập nhật lịch sử lane
    if left_point != -1:
        prev_left.append(left_point)
    if right_point != -1:
        prev_right.append(right_point)

    # Vẽ lên ảnh
    if draw is not None:
        cv2.line(draw, (0, interested_line_y),
                 (im_width, interested_line_y), (0, 0, 255), 2)
        if left_point != -1:
            cv2.circle(draw, (left_point, interested_line_y),
                       7, (255, 255, 0), -1)
        if right_point != -1:
            cv2.circle(draw, (right_point, interested_line_y),
                       7, (0, 255, 0), -1)

    return left_point, right_point


def calculate_control_signal(img, draw=None):
    """Tính steering & throttle để xe chạy giữa lane"""
    global lost_line_counter, prev_steering_angles

    img_lines = find_lane_lines(img)
    img_birdview = birdview_transform(img_lines)

    if draw is not None:
        draw[:, :] = birdview_transform(draw)

    left_point, right_point = find_left_right_points(img_birdview, draw=draw)

    im_center = img.shape[1] // 2
    throttle = 0.1
    steering_angle = 0
    status = "NORMAL"

    # --- Trường hợp tìm thấy cả 2 lane ---
    if left_point != -1 and right_point != -1:
        lost_line_counter = 0  # Reset bộ đếm
        center_lane = (left_point + right_point) // 2
        deviation = im_center - center_lane
        steering_angle = -float(deviation * 0.01)

        # === CẢI TIẾN: Điều chỉnh tốc độ dựa trên độ biến thiên góc lái ===

        # Tính độ biến thiên góc lái (steering change rate)
        if len(prev_steering_angles) > 0:
            steering_change = abs(steering_angle - prev_steering_angles[-1])
        else:
            steering_change = 0

        # Lưu góc lái hiện tại vào lịch sử
        prev_steering_angles.append(steering_angle)

        # Tính độ biến thiên trung bình trong 5 frame gần nhất
        if len(prev_steering_angles) >= 2:
            avg_steering_change = sum(
                abs(prev_steering_angles[i] - prev_steering_angles[i - 1])
                for i in range(1, len(prev_steering_angles))
            ) / (len(prev_steering_angles) - 1)
        else:
            avg_steering_change = 0

        # Tốc độ cơ bản - TĂNG LÊN để chạy nhanh hơn
        base_throttle = 1  # Tăng từ 0.25 → 0.45

        # GIẢM các hệ số penalty để ít bị giảm tốc
        # Giảm tốc theo góc lái hiện tại (giảm ảnh hưởng)
        angle_penalty = abs(steering_angle) * 0.6  # Giảm từ 0.15 → 0.12

        # Giảm tốc theo độ biến thiên tức thời (giảm ảnh hưởng)
        change_penalty = steering_change * 0.5  # Giảm từ 0.8 → 0.5

        # Giảm tốc theo độ biến thiên trung bình (giảm ảnh hưởng)
        avg_change_penalty = avg_steering_change * 0.8  # Giảm từ 0.6 → 0.4

        # Tính throttle cuối cùng
        throttle = base_throttle - angle_penalty - change_penalty - avg_change_penalty

        # Giới hạn throttle - TĂNG min và max
        throttle = max(0.15, min(1, throttle))  # Tăng từ [0.08, 0.25] → [0.15, 0.45]

        status = "TRACKING"

        if draw is not None:
            cv2.line(draw, (int(center_lane), 0),
                     (int(center_lane), img.shape[0]), (0, 255, 255), 2)

            # Hiển thị thêm thông tin về steering change
            cv2.putText(draw, f"Steering change: {steering_change:.3f}",
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
            cv2.putText(draw, f"Avg change: {avg_steering_change:.3f}",
                        (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    # --- Trường hợp MẤT CẢ 2 LANE ---
    else:
        lost_line_counter += 1

        # Reset lịch sử góc lái khi mất line lâu
        if lost_line_counter > LOST_LINE_THRESHOLD:
            prev_steering_angles.clear()

        if lost_line_counter > LOST_LINE_THRESHOLD:
            # Chế độ khẩn cấp: Đi thẳng rồi quẹo trái nhẹ
            status = "LOST LINE - EMERGENCY"

            # Đi thẳng trong 20 frame đầu
            if lost_line_counter <= LOST_LINE_THRESHOLD + 20:
                steering_angle = 0.0
                throttle = 0.25  # Tăng từ 0.15 → 0.25
                status = "LOST LINE - GO STRAIGHT"

            # Sau đó quẹo trái nhẹ để tìm line
            else:
                steering_angle = 0.3  # Quẹo trái (giá trị dương)
                throttle = 0.20  # Tăng từ 0.12 → 0.20
                status = "LOST LINE - TURN LEFT"

        else:
            # Vẫn còn trong ngưỡng an toàn, giữ hướng cũ
            status = "LOST LINE - KEEP DIRECTION"
            steering_angle = 0.0
            throttle = 0.20  # Tăng từ 0.1 → 0.20

    # Vẽ thông tin lên màn hình
    if draw is not None:
        # Hiển thị trạng thái
        color = (0, 255, 0) if status == "TRACKING" else (0, 0, 255)
        cv2.putText(draw, f"Status: {status}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Hiển thị steering và throttle
        cv2.putText(draw, f"Steering: {steering_angle:+.3f}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(draw, f"Throttle: {throttle:.2f}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Hiển thị bộ đếm mất line
        cv2.putText(draw, f"Lost frames: {lost_line_counter}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return throttle, steering_angle