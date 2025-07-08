import os, json, cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from typing import Optional, Tuple, Dict, Any

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def preprocess_image(
    image: np.ndarray, 
    img_size: Tuple[int, int]=(24, 24), 
    method: Optional[str] = 'adaptive_binary'
) -> Optional[np.ndarray]:

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'adaptive_binary':
        proc_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif method == 'Nothing':
        proc_img = gray_img.copy()
    else:
        raise ValueError("adaptive_binary | Nothing")

    img_resized = cv2.resize(proc_img, img_size)
    img_norm = (img_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.expand_dims(img_norm[None, ...], axis=0).astype(np.float32)  # [1, 1, H, W]

def predict_digit(session, img):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # if img.shape[1] == 2:  # (B, 2, H, W)
    #     third_channel = np.zeros_like(img[:, :1, :, :])  # shape: (B, 1, H, W)
    #     img = np.concatenate([img, third_channel], axis=1)  # (B, 3, H, W)

    pred = session.run([output_name], {input_name: img})[0]
    return np.argmax(pred, axis=1)[0]  # возвращает метку класса

def validate_roi(x, y, w, h, max_width, max_height):
    """Проверка, что ROI находится в допустимых границах."""
    x = max(0, min(x, max_width))
    y = max(0, min(y, max_height))
    w = max(1, min(w, max_width - x))
    h = max(1, min(h, max_height - y))
    return x, y, w, h

def evaluate_metrics(true_labels, pred_labels, position):
    """Вычисляет метрики для указанной позиции цифры."""
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    print(f"\n Позиция {position}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1: {f1:.4f}")
    return {'Accuracy': accuracy, 'F1': f1}

def process_video(video_path, camera_params, metrics_data, onnx_session, output_dir, method=None, video_name=None):
    """Обрабатывает видео, предсказывает цифры и возвращает метрики"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    camera_key = video_name.replace('.mp4', '')

    params = camera_params[camera_key]
    resolution = params['video_resolution']
    video_width, video_height = resolution['width'], resolution['height']

    # Извлечение параметров даты
    date_crop = params.get('data_crop_region', {})
    date_digits = params.get('data_digit_coords', {}).get('digits', [])
    date_x_start = int(date_crop.get('x_start', 0) * video_width) if date_crop else 0
    date_y_start = int(date_crop.get('y_start', 0) * video_height) if date_crop else 0
    date_width = int(date_crop.get('width', 0) * video_width) if date_crop else 0
    date_height = int(date_crop.get('height', 0) * video_height) if date_crop else 0
    date_x_start, date_y_start, date_width, date_height = validate_roi(
        date_x_start, date_y_start, date_width, date_height, video_width, video_height)

    # Извлечение параметров времени
    time_crop = params['time_crop_region']
    time_digits = params['time_digit_coords']['digits']
    time_x_start = int(time_crop['x_start'] * video_width)
    time_y_start = int(time_crop['y_start'] * video_height)
    time_width = int(time_crop['width'] * video_width)
    time_height = int(time_crop['height'] * video_height)
    time_x_start, time_y_start, time_width, time_height = validate_roi(
        time_x_start, time_y_start, time_width, time_height, video_width, video_height)

    # секунды в номера кадров
    frames_to_check = [int(float(frame_sec) * fps) for frame_sec in metrics_data[video_name]['date'].keys()]

    true_data_digits = {frame: metrics_data[video_name]['date'][str(int(frame / fps))] for frame in frames_to_check}
    true_time_digits = {frame: metrics_data[video_name]['time'][str(int(frame / fps))] for frame in frames_to_check}

    # хранение меток
    data_predictions = {i: {'true': [], 'pred': []} for i in range(8)}
    time_predictions = {i: {'true': [], 'pred': []} for i in range(6)}

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count in frames_to_check:
            if date_crop and date_digits:
                date_roi = frame[date_y_start:date_y_start + date_height, date_x_start:date_x_start + date_width]
                if date_roi.size == 0:
                    print(f"Ошибка: пустой date_roi для кадра {frame_count}")
                    for i in range(8):
                        data_predictions[i]['true'].append(int(true_data_digits[frame_count][str(i)]))
                        data_predictions[i]['pred'].append(-1)
                    continue
                for i, digit in enumerate(date_digits):
                    x1 = int(digit['x1'] * date_width)
                    y1 = int(digit['y1'] * date_height)
                    w = int(digit['width'] * date_width)
                    h = int(digit['height'] * date_height)
                    x1, y1, w, h = validate_roi(x1, y1, w, h, date_width, date_height)
                    digit_img = date_roi[y1:y1 + h, x1:x1 + w]
                    processed_img = preprocess_image(digit_img, method=method)
                    if processed_img is None:
                        data_predictions[i]['true'].append(int(true_data_digits[frame_count][str(i)]))
                        data_predictions[i]['pred'].append(-1)
                        continue
                    pred_digit = predict_digit(onnx_session, processed_img)
                    true_digit = int(true_data_digits[frame_count][str(i)])
                    data_predictions[i]['true'].append(true_digit)
                    data_predictions[i]['pred'].append(pred_digit)

            # Обработка цифр времени
            time_roi = frame[time_y_start:time_y_start + time_height, time_x_start:time_x_start + time_width]
            if time_roi.size == 0:
                print(f"Ошибка: пустой time_roi для кадра {frame_count}")
                for i in range(6):
                    time_predictions[i]['true'].append(int(true_time_digits[frame_count][str(i)]))
                    time_predictions[i]['pred'].append(-1)
                continue
            for i, digit in enumerate(time_digits):
                x1 = int(digit['x1'] * time_width)
                y1 = int(digit['y1'] * time_height)
                w = int(digit['width'] * time_width)
                h = int(digit['height'] * time_height)
                x1, y1, w, h = validate_roi(x1, y1, w, h, time_width, time_height)
                digit_img = time_roi[y1:y1 + h, x1:x1 + w]
                if digit_img.size > 0:
                    processed_img = preprocess_image(digit_img, method=method)
                    if processed_img is None:
                        time_predictions[i]['true'].append(int(true_time_digits[frame_count][str(i)]))
                        time_predictions[i]['pred'].append(-1)
                        continue
                    pred_digit = predict_digit(onnx_session, processed_img)
                    true_digit = int(true_time_digits[frame_count][str(i)])
                    time_predictions[i]['true'].append(true_digit)
                    time_predictions[i]['pred'].append(pred_digit)

        frame_count += 1

    cap.release()

    # Вычисление метрик и подготовка данных для CSV
    rows = []
    for i in range(8):
        if data_predictions[i]['true']:
            valid_indices = [idx for idx, pred in enumerate(data_predictions[i]['pred']) if pred != -1]
            true_labels = [data_predictions[i]['true'][idx] for idx in valid_indices]
            pred_labels = [data_predictions[i]['pred'][idx] for idx in valid_indices]
            if true_labels:
                metrics = evaluate_metrics(true_labels, pred_labels, f"Дата, позиция {i+1}")
                rows.append({
                    "video": video_name,
                    "type": "date",
                    "position": i + 1,
                    "accuracy": metrics['accuracy'],
                    "f1_score": metrics['f1']})
    for i in range(6):
        if time_predictions[i]['true']:
            valid_indices = [idx for idx, pred in enumerate(time_predictions[i]['pred']) if pred != -1]
            true_labels = [time_predictions[i]['true'][idx] for idx in valid_indices]
            pred_labels = [time_predictions[i]['pred'][idx] for idx in valid_indices]
            if true_labels:
                metrics = evaluate_metrics(true_labels, pred_labels, f"Время, позиция {i+1}")
                rows.append({
                    "video": video_name,
                    "type": "time",
                    "position": i + 1,
                    "accuracy": metrics['accuracy'],
                    "f1_score": metrics['f1']})
    return rows

def main():
    onnx_model_path = "main/24x24train_CNN9_80e/best.onnx"
    video_dir = "sip_12022025/sip_12022025"
    metrics_dir = "metrics_params"
    camera_params_path = "configs/camera_params_Rcopy.json"
    output_dir = "metrics"
    method = "Nothing"

    csv_folder = os.path.join(output_dir, "24x24train_CNN9_80e_metrics10")
    os.makedirs(csv_folder, exist_ok=True)

    session = ort.InferenceSession(onnx_model_path)
    input_details = session.get_inputs()
    for input in input_details:
        print(f"Input name: {input.name}, Shape: {input.shape}, Type: {input.type}")

    camera_params = load_json(camera_params_path)
    all_metrics = []

    for metrics_file in os.listdir(metrics_dir):
        if metrics_file.endswith('.json'):
            metrics_path = os.path.join(metrics_dir, metrics_file)
            metrics_data = load_json(metrics_path)

            video_name = metrics_file.replace('.json', '.mp4')
            print("Video", video_name)
            video_path = os.path.join(video_dir, video_name)

            rows = process_video(video_path, camera_params, metrics_data, session, csv_folder, method=method, video_name=video_name)
            if rows:
                df = pd.DataFrame(rows)
                out_path = os.path.join(csv_folder, f"{video_name}_metrics.csv")
                df.to_csv(out_path, index=False)
                all_metrics.extend(rows)


    # summary_metrics
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_avg = summary_df.groupby(['type', 'position']).mean(numeric_only=True).reset_index()
        summary_csv_path = os.path.join(csv_folder, "summary_metrics.csv")
        summary_avg.to_csv(summary_csv_path, index=False)
        print("=======summary_metrics DONE==========")

if __name__ == "__main__":
    main()
