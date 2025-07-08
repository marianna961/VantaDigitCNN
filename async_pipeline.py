# python async_pipeline.py --onnx chk/24x24_train_CNN_80e/best.onnx --video sip_12022025/sip_12022025 --out output/24x24_CNN9 --cam_cfg configs/camera_params_Rcopy.json --threads 8
"""
Асинхронный pipeline
3 потока + две неблокирующие очереди
Декодирование, обработка и кодирование происходят одновременно.
"""

import os, cv2, json, time, threading, argparse, pathlib, sys
import numpy as np
import onnxruntime as ort
import queue

def load_session(path: str, num_threads: int = 8):
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])

def _prep_single(roi: np.ndarray, size=(24, 24)) -> np.ndarray:
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    g = (g.astype(np.float32) / 255.0 - 0.5) / 0.5
    return g

def batch_preprocess(rois: list[np.ndarray]) -> np.ndarray:
    return np.stack([_prep_single(r) for r in rois])[:, None, :, :] #(N, H, W) в (N, 1, H, W)


STOP = object()

class AsyncPipeline:
    def __init__(self, onnx: str, cam_cfg: dict, threads: int, write: bool):
        self.sess = load_session(onnx, threads)
        self.cfg_all = cam_cfg
        self.write = write
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name


    @staticmethod
    def _crop_rect(r, W, H): 
        """Преобразует относительные координаты в пиксели"""
        return (int(r["x_start"] * W), int(r["y_start"] * H), int(r["width"] * W), int(r["height"] * H))

    @staticmethod
    def _safe(x, y, w, h, mw, mh):
        x = max(0, min(x, mw)); y = max(0, min(y, mh))
        w = max(1, min(w, mw - x)); h = max(1, min(h, mh - y))
        return x, y, w, h

    # 1 decode
    def _decode(self, cap: cv2.VideoCapture, q_out: queue.Queue):
        """Чтение кадров из видео и отправка их в очередь"""
        while True:
            ok, frame = cap.read()
            if not ok:
                q_out.put(STOP); break
            q_out.put(frame)

    # 2 inference
    def _infer(self, q_in: queue.Queue, q_out: queue.Queue, cfg: dict):
        W, H = cfg["video_resolution"]["width"], cfg["video_resolution"]["height"]
        box_t = self._crop_rect(cfg["time_crop_region"], W, H)
        box_d = self._crop_rect(cfg["data_crop_region"], W, H)
        dig_t = cfg["time_digit_coords"]["digits"]
        dig_d = cfg["data_digit_coords"]["digits"]

        def crop_digits(frame, box, digits):
            x, y, w, h = box; sub = frame[y:y + h, x:x + w]
            res = []
            for d in digits:
                cx = int(d["x1"] * w); cy = int(d["y1"] * h)
                cw = int(d["width"] * w); ch = int(d["height"] * h)
                cx, cy, cw, ch = self._safe(cx, cy, cw, ch, w, h)
                res.append(sub[cy:cy + ch, cx:cx + cw])
            return res

        # FPS counters
        local_frames = 0
        t0 = time.perf_counter(); t_mark = t0

        while True:
            frm = q_in.get()
            if frm is STOP:
                q_out.put(STOP); break
            rois = crop_digits(frm, box_t, dig_t) + crop_digits(frm, box_d, dig_d)
            logits = self.sess.run([self.out_name], {self.in_name: batch_preprocess(rois)})[0]
            d = logits.argmax(1).tolist()
            t_str = f"{d[0]}{d[1]}:{d[2]}{d[3]}:{d[4]}{d[5]}"
            dat = f"{d[6]}{d[7]}/{d[8]}{d[9]}/{d[10]}{d[11]}{d[12]}{d[13]}"
            cv2.putText(frm, t_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frm, dat, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            q_out.put(frm)

            # --- FPS live ---
            local_frames += 1
            if time.perf_counter() - t_mark >= 2.0:  # каждые 2 с
                inst_fps = local_frames / (time.perf_counter() - t0)
                print(f"[FPS] {inst_fps:.1f} frame/s  (~{inst_fps*14:.0f} ROI/s)", end="\r", flush=True)
                t_mark = time.perf_counter()

    # 3 encode
    @staticmethod
    def _encode(writer: cv2.VideoWriter, q_in: queue.Queue):
        while True:
            frm = q_in.get()
            if frm is STOP: break
            writer.write(frm)

    def run_single(self, v_in: str, v_out: str | None):
        key = pathlib.Path(v_in).stem
        cfg = self.cfg_all[key]

        cap = cv2.VideoCapture(v_in)
        fps_src, W, H = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))

        writer = None
        if self.write and v_out:
            four = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(v_out, four, fps_src, (W, H))

        q_d2i = queue.Queue(maxsize=4)
        q_i2e = queue.Queue(maxsize=4)

        threads = [
            threading.Thread(target=self._decode, args=(cap, q_d2i), daemon=True),
            threading.Thread(target=self._infer, args=(q_d2i, q_i2e, cfg), daemon=True)]
        if writer:
            threads.append(threading.Thread(target=self._encode, args=(writer, q_i2e), daemon=True))

        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        dur = time.perf_counter() - t0
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release(); writer.release() if writer else None

        fps_proc = nframes / dur if dur else 0
        roi_fps = fps_proc * 14
        print(f"\n{os.path.basename(v_in)} → {fps_proc:.1f} frame/s | {roi_fps:.0f} ROI/s")

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--onnx", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cam_cfg", required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--no_write", action="store_true")
    return p.parse_args()


def main():
    args = get_args()

    with open(args.cam_cfg, "r", encoding="utf-8") as f:
        cam_cfg = json.load(f)

    pipe = AsyncPipeline(args.onnx, cam_cfg, args.threads, write=not args.no_write)
    v_path = pathlib.Path(args.video)
    o_path = pathlib.Path(args.out)

    o_path.mkdir(parents=True, exist_ok=True)
    vids = [p for p in v_path.iterdir() if p.suffix.lower() == ".mp4"]
    print(f"[i] Found {len(vids)} videos in '{v_path}'")
    for v in vids:
        out_file = str(o_path / v.name) if pipe.write else None
        pipe.run_single(str(v), out_file)

    #     if pipe.write:
    #         if o_path.suffix.lower() == ".mp4":
    #             out_file = str(o_path)
    #         else:
    #             o_path.mkdir(parents=True, exist_ok=True)
    #             out_file = str(o_path / v_path.name)
    #     else:
    #         out_file = None
    #     pipe.run_single(str(v_path), out_file)

if __name__ == "__main__":
    main()
