import cv2
import numpy as np
from collections import deque

# Lưu giá trị lane trước đó để dự đoán nếu mất lane
prev_left = deque(maxlen=5)
prev_right = deque(maxlen=5)

# Biến theo dõi trạng thái mất line
lost_line_counter = 0
LOST_LINE_THRESHOLD = 10

# Lưu lịch sử góc lái để tính độ biến thiên
prev_steering_angles = deque(maxlen=5)

# ===== BIẾN XỬ LÝ BIỂN BÁO =====
sign_state = {
    'current_sign': None,
    'sign_distance': 0,
    'preparing_turn': False,
    'turning': False,
    'turn_counter': 0,
    'turn_direction': None,
    'sign_detected_frames': 0,
    'pre_turn_slowdown': False,
}


def reset_state():
    """Reset toàn bộ trạng thái về ban đầu - GỌI HÀM NÀY TRƯỚC MỖI LẦN CHẠY"""
    global prev_left, prev_right, lost_line_counter, prev_steering_angles, sign_state

    prev_left.clear()
    prev_right.clear()
    lost_line_counter = 0
    prev_steering_angles.clear()

    sign_state['current_sign'] = None
    sign_state['sign_distance'] = 0
    sign_state['preparing_turn'] = False
    sign_state['turning'] = False
    sign_state['turn_counter'] = 0
    sign_state['turn_direction'] = None
    sign_state['sign_detected_frames'] = 0
    sign_state['pre_turn_slowdown'] = False

    print("✅ Đã reset toàn bộ trạng thái về ban đầu")


# ===== CẤU HÌNH ĐÃ ĐƯỢC TỐI ƯU HÓA =====
TURN_CONFIG = {
    'sign_confirmation_frames': 3,  # Tăng số frame xác nhận để ổn định hơn
    'pre_turn_distance': 999999,  # BỎ NGƯỠNG - giảm tốc NGAY khi thấy biển
    'initial_slowdown_throttle': 0.15,  # Tốc độ giảm ban đầu (xa)
    'medium_slowdown_throttle': 0.10,  # Tốc độ giảm trung bình (gần hơn)
    'final_slowdown_throttle': 0.05,  # Tốc độ giảm cuối (rất gần) - GIẢM MẠNH HƠN
    'turn_throttle': 0.1,  # Tốc độ khi quẹo - CỰC CHẬM HƠN
    'turn_duration': 45,  # Thời gian quẹo dài hơn để hoàn thành 90 độ
    'turn_steering_angle': 1.0,  # Góc lái tối đa (đã là max)
    'post_turn_frames': 30,  # Thời gian ổn định dài hơn
    'start_turn_distance': 35,  # Ngưỡng bắt đầu quẹo SỚM HƠN (từ 35 -> 50)
    'distance_far': 150,  # Khoảng cách xa (tăng để detect sớm hơn)
    'distance_medium': 90,  # Khoảng cách trung bình (tăng)
    'distance_near': 50,  # Khoảng cách gần (tăng để quẹo sớm hơn)
}


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

    left_point, right_point = -1, -1
    lane_width_est = 250
    center = im_width // 2

    for x in range(center, 0, -1):
        if interested_line[x] > 0:
            left_point = x
            break
    for x in range(center + 1, im_width):
        if interested_line[x] > 0:
            right_point = x
            break

    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width_est
    if right_point != -1 and left_point == -1:
        left_point = right_point - lane_width_est

    if left_point == -1 and len(prev_left) > 0:
        left_point = int(np.mean(prev_left))
    if right_point == -1 and len(prev_right) > 0:
        right_point = int(np.mean(prev_right))

    if left_point != -1:
        prev_left.append(left_point)
    if right_point != -1:
        prev_right.append(right_point)

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


def estimate_sign_distance(bbox):
    """Ước tính khoảng cách đến biển báo dựa trên kích thước bbox - ĐÃ ỔN ĐỊNH HÓA"""
    x, y, w, h = bbox
    # Lấy trung bình kích thước và làm tròn để giảm dao động
    size_score = round((w + h) / 2)
    # Công thức ước tính với làm tròn
    estimated_distance = max(10, round(250 - size_score * 2.5))
    return estimated_distance


def process_traffic_signs(signs, img_height):
    """Xử lý thông tin biển báo và cập nhật trạng thái - ĐÃ GIẢM SAI SỐ"""
    global sign_state

    # Lọc các biển báo rẽ trái/phải
    turn_signs = [s for s in signs if s[0] in ['left', 'right']]

    if not turn_signs:
        sign_state['sign_detected_frames'] = 0
        if not sign_state['turning']:
            sign_state['current_sign'] = None
            sign_state['preparing_turn'] = False
            sign_state['pre_turn_slowdown'] = False
        return

    # Lấy biển báo gần nhất (ở vị trí thấp nhất trong ảnh)
    turn_signs.sort(key=lambda s: s[2] + s[4], reverse=True)
    closest_sign = turn_signs[0]

    sign_type = closest_sign[0]
    bbox = closest_sign[1:5]
    distance = estimate_sign_distance(bbox)

    # Xác nhận biển báo
    if sign_state['current_sign'] == sign_type:
        sign_state['sign_detected_frames'] += 1
    else:
        sign_state['current_sign'] = sign_type
        sign_state['sign_detected_frames'] = 1
        sign_state['sign_distance'] = distance

    # Cập nhật khoảng cách với trọng số ưu tiên giá trị cũ hơn (giảm dao động)
    sign_state['sign_distance'] = round(sign_state['sign_distance'] * 0.85 + distance * 0.15)

    if sign_state['sign_detected_frames'] < TURN_CONFIG['sign_confirmation_frames']:
        return

    # Bắt đầu giảm tốc NGAY KHI XÁC NHẬN BIỂN BÁO
    if (not sign_state['preparing_turn'] and
            not sign_state['turning']):
        sign_state['preparing_turn'] = True
        sign_state['pre_turn_slowdown'] = True
        sign_state['turn_direction'] = sign_type
        print(f"\n{'=' * 70}")
        print(f"🚦 PHÁT HIỆN BIỂN BÁO: {sign_type.upper()}")
        print(f"📏 Khoảng cách: {sign_state['sign_distance']:.0f}px")
        print(f"🐌 BẮT ĐẦU GIẢM TỐC NGAY LẬP TỨC!")
        print(f"{'=' * 70}\n")


def calculate_control_signal(img, signs=None, draw=None):
    """Tính steering & throttle với xử lý biển báo - ĐÃ GIẢM SAI SỐ"""
    global lost_line_counter, prev_steering_angles, sign_state

    # Xử lý biển báo nếu có
    if signs is not None and len(signs) > 0:
        process_traffic_signs(signs, img.shape[0])

    img_lines = find_lane_lines(img)
    img_birdview = birdview_transform(img_lines)

    if draw is not None:
        draw[:, :] = birdview_transform(draw)

    left_point, right_point = find_left_right_points(img_birdview, draw=draw)

    im_center = img.shape[1] // 2
    throttle = 0.1
    steering_angle = 0
    status = "NORMAL"

    # ===== XỬ LÝ BIỂN BÁO - ƯU TIÊN CAO NHẤT =====

    # 🔄 Trạng thái 1: ĐANG QUẸO (ƯU TIÊN TUYỆT ĐỐI)
    if sign_state['turning']:
        sign_state['turn_counter'] += 1

        # Phase 1: Thực hiện quẹo
        if sign_state['turn_counter'] <= TURN_CONFIG['turn_duration']:
            # Đảo ngược logic để quẹo đúng hướng
            if sign_state['turn_direction'] == 'left':
                steering_angle = -TURN_CONFIG['turn_steering_angle']  # ÂM = TRÁI
            else:  # right
                steering_angle = TURN_CONFIG['turn_steering_angle']  # DƯƠNG = PHẢI

            # ⚠️ FORCE THROTTLE THẤP - KHÔNG CHO LOGIC KHÁC GHI ĐÈ
            throttle = TURN_CONFIG['turn_throttle']
            status = f"🔄 TURNING {sign_state['turn_direction'].upper()} 90°"

            # Debug mỗi 5 frame
            if sign_state['turn_counter'] % 5 == 0:
                print(
                    f"⏳ Quẹo {sign_state['turn_direction'].upper()}: frame {sign_state['turn_counter']}/{TURN_CONFIG['turn_duration']} | steering={steering_angle:+.2f} | throttle={throttle:.3f}")

        # Phase 2: Ổn định sau quẹo
        elif sign_state['turn_counter'] <= TURN_CONFIG['turn_duration'] + TURN_CONFIG['post_turn_frames']:
            steering_angle = 0
            throttle = 0.15
            status = "✅ STABILIZING"

        # Phase 3: Hoàn thành
        else:
            sign_state['turning'] = False
            sign_state['preparing_turn'] = False
            sign_state['pre_turn_slowdown'] = False
            sign_state['turn_counter'] = 0
            sign_state['current_sign'] = None
            sign_state['turn_direction'] = None
            prev_steering_angles.clear()
            print(f"\n✅ HOÀN THÀNH QUẸO - Trở lại tracking bình thường\n")

    # 🐌 Trạng thái 2: ĐANG GIẢM TỐC (3 MỨC ĐỘ)
    elif sign_state['preparing_turn'] and sign_state['pre_turn_slowdown']:
        # Kiểm tra đủ gần để bắt đầu quẹo
        if (sign_state['sign_distance'] < TURN_CONFIG['start_turn_distance'] or
                left_point == -1 or right_point == -1):

            sign_state['turning'] = True
            sign_state['turn_counter'] = 0
            sign_state['pre_turn_slowdown'] = False
            print(f"\n🔄 BẮT ĐẦU QUẸO 90° {sign_state['turn_direction'].upper()}!\n")

            # ⚠️ SET THROTTLE NGAY KHI BẮT ĐẦU QUẸO
            throttle = TURN_CONFIG['turn_throttle']
            if sign_state['turn_direction'] == 'left':
                steering_angle = -TURN_CONFIG['turn_steering_angle']
            else:
                steering_angle = TURN_CONFIG['turn_steering_angle']
            status = f"🔄 TURNING {sign_state['turn_direction'].upper()} 90°"
        else:
            # XÁC ĐỊNH TỐC ĐỘ DỰA TRÊN KHOẢNG CÁCH
            distance = sign_state['sign_distance']

            if distance > TURN_CONFIG['distance_far']:
                throttle = TURN_CONFIG['initial_slowdown_throttle']
                slowdown_level = "NHẸ"
            elif distance > TURN_CONFIG['distance_medium']:
                throttle = TURN_CONFIG['medium_slowdown_throttle']
                slowdown_level = "VỪA"
            else:
                throttle = TURN_CONFIG['final_slowdown_throttle']
                slowdown_level = "MẠNH"

            # Tracking trong khi giảm tốc - GIẢM HỆ SỐ ĐỂ ỔN ĐỊNH HƠN
            if left_point != -1 and right_point != -1:
                lost_line_counter = 0
                center_lane = (left_point + right_point) // 2
                deviation = im_center - center_lane
                # GIẢM hệ số từ 0.007 -> 0.006 để ổn định hơn
                steering_angle = -float(deviation * 0.006)
                # Làm tròn góc lái để giảm dao động nhỏ
                steering_angle = round(steering_angle, 3)
                status = f"🐌 GIẢM TỐC {slowdown_level} [{throttle:.2f}] - {sign_state['turn_direction'].upper()} ({distance:.0f}px)"

                if draw is not None:
                    cv2.line(draw, (int(center_lane), 0),
                             (int(center_lane), img.shape[0]), (0, 255, 255), 2)
            else:
                steering_angle = 0
                status = f"🐌 GIẢM TỐC {slowdown_level} [{throttle:.2f}] - LOST LANE"

    # 🚗 Trạng thái 3: TRACKING BÌNH THƯỜNG
    elif left_point != -1 and right_point != -1:
        lost_line_counter = 0
        center_lane = (left_point + right_point) // 2
        deviation = im_center - center_lane
        # GIẢM hệ số từ 0.01 -> 0.008 để tracking ổn định hơn
        steering_angle = -float(deviation * 0.008)

        if len(prev_steering_angles) > 0:
            steering_change = abs(steering_angle - prev_steering_angles[-1])
        else:
            steering_change = 0

        prev_steering_angles.append(steering_angle)

        if len(prev_steering_angles) >= 2:
            avg_steering_change = sum(
                abs(prev_steering_angles[i] - prev_steering_angles[i - 1])
                for i in range(1, len(prev_steering_angles))
            ) / (len(prev_steering_angles) - 1)
        else:
            avg_steering_change = 0

        base_throttle = 1
        angle_penalty = abs(steering_angle) * 0.8
        change_penalty = steering_change * 0.5
        avg_change_penalty = avg_steering_change * 0.8

        throttle = base_throttle - angle_penalty - change_penalty - avg_change_penalty
        throttle = max(0.15, min(1, throttle))

        # Làm tròn góc lái để giảm dao động nhỏ
        steering_angle = round(steering_angle, 3)

        status = "🚗 TRACKING"

        if draw is not None:
            cv2.line(draw, (int(center_lane), 0),
                     (int(center_lane), img.shape[0]), (0, 255, 255), 2)

    # ⚠️ Trạng thái 4: MẤT LANE
    else:
        lost_line_counter += 1

        if lost_line_counter > LOST_LINE_THRESHOLD:
            prev_steering_angles.clear()

        if lost_line_counter > LOST_LINE_THRESHOLD:
            status = "⚠️ LOST LINE - EMERGENCY"
            if lost_line_counter <= LOST_LINE_THRESHOLD + 20:
                steering_angle = 0.0
                throttle = 0.25
                status = "⚠️ LOST - GO STRAIGHT"
            else:
                steering_angle = 0.3
                throttle = 0.20
                status = "⚠️ LOST - TURN LEFT"
        else:
            status = "⚠️ LOST - KEEP DIRECTION"
            steering_angle = 0.0
            throttle = 0.20

    # 📊 Vẽ thông tin lên màn hình
    if draw is not None:
        color = (0, 255, 0) if "TRACKING" in status else (0, 0, 255)
        cv2.putText(draw, f"Status: {status}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(draw, f"Steering: {steering_angle:+.3f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Highlight throttle với màu theo mức độ giảm tốc
        throttle_color = (0, 255, 255)  # Xanh dương - bình thường
        if sign_state['preparing_turn']:
            if throttle <= 0.06:
                throttle_color = (0, 0, 255)  # Đỏ - giảm cực mạnh
            elif throttle <= 0.10:
                throttle_color = (0, 100, 255)  # Cam - giảm mạnh
            elif throttle <= 0.15:
                throttle_color = (0, 200, 255)  # Vàng - giảm nhẹ

        cv2.putText(draw, f"Throttle: {throttle:.3f}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, throttle_color, 2)

        # Thông tin biển báo
        if sign_state['current_sign']:
            sign_color = (255, 100, 255)
            if sign_state['turning']:
                sign_color = (0, 150, 255)
            elif sign_state['preparing_turn']:
                sign_color = (0, 255, 255)

            sign_text = f"Sign: {sign_state['current_sign'].upper()} | Dist: {sign_state['sign_distance']:.0f}px"
            cv2.putText(draw, sign_text,
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sign_color, 2)

            # Phase info
            if sign_state['preparing_turn']:
                phase_text = "Phase: SLOWING DOWN"
                cv2.putText(draw, phase_text,
                            (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
            elif sign_state['turning']:
                phase_text = f"Phase: TURNING ({sign_state['turn_counter']}/{TURN_CONFIG['turn_duration']})"
                cv2.putText(draw, phase_text,
                            (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2)

        cv2.putText(draw, f"Lost: {lost_line_counter}",
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    return throttle, steering_angle