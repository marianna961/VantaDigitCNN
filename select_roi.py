import cv2
import json
import os
import glob
import numpy as np

all_roi_coords = []
drawing = False
current_roi_index = 0
current_video_path = ""
current_cap = None
all_camera_configs = {}
scale_factor = 3  # Коэффициент масштабирования для окна выделения цифр

def select_digit_callback(event, x, y, flags, param):
    global all_roi_coords, drawing, img, current_digit_index, digit_coords, scaled_roi_img
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            orig_x, orig_y = int(x / scale_factor), int(y / scale_factor)
            digit_coords.append([(orig_x, orig_y)])
            drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            orig_x, orig_y = int(x / scale_factor), int(y / scale_factor)
            digit_coords[current_digit_index].append((orig_x, orig_y))
            drawing = False
            p1 = digit_coords[current_digit_index][0]
            p2 = digit_coords[current_digit_index][1]
            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
            scaled_x1, scaled_y1 = int(x1 * scale_factor), int(y1 * scale_factor)
            scaled_x2, scaled_y2 = int(x2 * scale_factor), int(y2 * scale_factor)
            cv2.rectangle(scaled_roi_img, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), (0, 255, 0), 2)
            cv2.imshow("Select Digit", scaled_roi_img)
            print(f"Цифра {current_digit_index + 1}: x1={x1}, y1={y1}, width={x2-x1}, height={y2-y1}")
            current_digit_index += 1

def select_roi_callback(event, x, y, flags, param):
    global all_roi_coords, drawing, img, current_video_path, current_cap, all_camera_configs, current_roi_index
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            all_roi_coords.append([(x, y)])
            drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            all_roi_coords[current_roi_index].append((x, y))
            drawing = False
            p1 = all_roi_coords[current_roi_index][0]
            p2 = all_roi_coords[current_roi_index][1]
            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Select ROI", img)
            crop_width = x2 - x1
            crop_height = y2 - y1
            print(f"ROI {current_roi_index + 1}: x1={x1}, y1={y1}, width={crop_width}, height={crop_height}")

            video_filename = os.path.splitext(os.path.basename(current_video_path))[0]
            video_width = int(current_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(current_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if video_filename not in all_camera_configs:
                all_camera_configs[video_filename] = {
                    "video_resolution": {"width": video_width, "height": video_height}
                }

            global num_digits, digit_coords, current_digit_index, scaled_roi_img
            if current_roi_index == 1:  # Для времени
                print("Формат времени: 'HH:MM:SS'")
                num_digits = int(input(f"Количество цифр для ROI {current_roi_index + 1} (должно быть 6): ") or 6)
            else:
                print("Формат даты: 'DD/MM/YYYY'")
                num_digits = int(input(f"Количество цифр для ROI {current_roi_index + 1} (должно быть 8): ") or 8)

            roi_img = img[y1:y2, x1:x2]
            scaled_roi_img = cv2.resize(roi_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            
            digit_coords = []
            current_digit_index = 0
            cv2.namedWindow("Select Digit")
            cv2.setMouseCallback("Select Digit", select_digit_callback)
            print(f"Выделите {num_digits} цифр в ROI {current_roi_index + 1} по очереди. Нажмите Enter для завершения или перехода к следующему этапу.")
            cv2.imshow("Select Digit", scaled_roi_img)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter
                    break
            cv2.destroyWindow("Select Digit")

            # Сохраняем координаты цифр в относительных единицах
            digits = []
            for (x1, y1), (x2, y2) in digit_coords:
                digit = {
                    "x1": round(x1 / crop_width, 4),  # Относительно ширины ROI
                    "y1": round(y1 / crop_height, 4),  # Относительно высоты ROI
                    "width": round((x2 - x1) / crop_width, 4),
                    "height": round((y2 - y1) / crop_height, 4)
                }
                digits.append(digit)

            region_params = {
                "x_start": round(x1 / video_width, 4),  # Относительно ширины видео
                "y_start": round(y1 / video_height, 4),  # Относительно высоты видео
                "width": round(crop_width / video_width, 4),
                "height": round(crop_height / video_height, 4)
            }

            digit_params = {
                "num_digits": num_digits,
                "digits": digits
            }

            if current_roi_index == 0:
                all_camera_configs[video_filename]["data_crop_region"] = region_params
                all_camera_configs[video_filename]["data_digit_coords"] = digit_params
            else:
                all_camera_configs[video_filename]["time_crop_region"] = region_params
                all_camera_configs[video_filename]["time_digit_coords"] = digit_params

            current_roi_index += 1
            if current_roi_index < 2:
                display_img = img.copy()
                if len(all_roi_coords) > 0:
                    for i in range(len(all_roi_coords)):
                        p1 = all_roi_coords[i][0]
                        p2 = all_roi_coords[i][1]
                        x1_roi, y1_roi = min(p1[0], p2[0]), min(p1[1], p2[1])
                        x2_roi, y2_roi = max(p1[0], p2[0]), max(p1[1], p2[1])
                        cv2.rectangle(display_img, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)
                print(f"Выберите ROI {current_roi_index + 1}")
                cv2.imshow("Select ROI", display_img)
                cv2.waitKey(0)
            else:
                cv2.destroyAllWindows()

video_folder = "sip_12022025/test"
output_config_dir = "configs"
output_config_file = os.path.join(output_config_dir, "camera_params.json")

video_files = glob.glob(os.path.join(video_folder, "*.mp4"))

if os.path.exists(output_config_file):
    with open(output_config_file, 'r') as f:
        all_camera_configs = json.load(f)

os.makedirs(output_config_dir, exist_ok=True)

for i, video_path in enumerate(video_files):
    video_filename_base = os.path.splitext(os.path.basename(video_path))[0]
    
    current_video_path = video_path
    all_roi_coords = []
    current_roi_index = 0

    current_cap = cv2.VideoCapture(video_path)
    total_frames = int(current_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_to_open = int(total_frames * 0.10)
    current_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_open)

    ret, img = current_cap.read()

    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", select_roi_callback)
    
    print("Первый ROI (дата)")
    cv2.imshow("Select ROI", img)
    cv2.waitKey(0)

    current_cap.release()

with open(output_config_file, 'w') as f:
    json.dump(all_camera_configs, f, indent=4)
print("Готово")