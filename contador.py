"""
Cuenta personas por cámara usando el HLS que ya genera main.js (sin tocar RTSP).

Cada INTERVAL_S segundos, por cada cámara de list.json:
  1. baja stream.m3u8 (HD) y el último segmento .ts completo
  2. toma su primer frame (keyframe)
  3. corre YOLO (solo clase "person") y cuenta las detecciones

Salidas en OUT_DIR:
  counts.json      {"ts": timestamp global, "counts": {mac: cantidad}}
  frames/<mac>.jpg último frame con las cajas dibujadas

Uso:
  pip install ultralytics requests
  python contador.py
"""
import json
import os
import socket
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import urljoin

import cv2
import requests
from ultralytics import YOLO


# ============================================================
# PC QUE EJECUTA EL PROGRAMA
# ============================================================
def get_pc():
    name = socket.gethostname()

    if name == "DESKTOP-TDCII3R":
        return "DEV-LAUTARO"

    if name == "DESKTOP-PHI93EU":
        return "PROD-CRIANZA"

    if name == "DESKTOP-2O19MEO":
        return "DEV-ADMINISTRACION"

    return name


PC = get_pc()

# URL según la PC (PROD-CRIANZA corre en el mismo server que las cámaras)
URLS = {
    "DEV-LAUTARO": "https://camaras.cimbrapp.work",
    "PROD-CRIANZA": "http://localhost:8083",
}
BASE_URL = URLS.get(PC, "https://camaras.cimbrapp.work")  # PC desconocida -> acceso remoto

# ============================================================
# CONFIG
# ============================================================
INTERVAL_S = 15
CONF = 0.35              # umbral de confianza (original en 0.4)
MODEL = "yolo11n.pt"    # en el server con GPU probar yolo11s.pt / yolo11m.pt

# Carpeta de salida según la PC
CARPETA = "contador_personas_streams"
BASES = {
    "DEV-LAUTARO": "E:/Descargas",
    "PROD-CRIANZA": "R:/",
    "DEV-ADMINISTRACION": os.path.join(os.path.expanduser("~"), "Downloads"),
}
OUT_DIR = os.path.join(BASES.get(PC, os.path.join(os.path.expanduser("~"), "Downloads")), CARPETA)

BOX_COLOR = (0, 200, 0)  # BGR

http = requests.Session()
http.headers["User-Agent"] = "Mozilla/5.0"  # por si Cloudflare filtra el UA por defecto de requests


# ============================================================
# FRAME DESDE HLS
# ============================================================
def grab_frame(cam):
    """Primer frame del último segmento completo del HLS (HD). None si algo falla."""
    try:
        m3u8_url = f"{BASE_URL}/streams/{cam['mac']}/hd/stream.m3u8"
        r = http.get(m3u8_url, timeout=10)
        r.raise_for_status()
        segs = [l.strip() for l in r.text.splitlines() if l.strip() and not l.startswith("#")]
        if not segs:
            return None

        seg = http.get(urljoin(m3u8_url, segs[-1]), timeout=15)
        seg.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as f:
            f.write(seg.content)
        try:
            cap = cv2.VideoCapture(f.name)
            ok, frame = cap.read()
            cap.release()
        finally:
            os.remove(f.name)
        return frame if ok else None
    except Exception as e:
        print(f"[{cam['name']}] error: {e}")
        return None


# ============================================================
# DIBUJO DE CAJAS (solo confianza, sin el nombre de la clase)
# ============================================================
def draw_boxes(frame, res):
    img = frame.copy()
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = f"{float(box.conf[0]):.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)

        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_top = max(y1 - th - base - 2, 0)
        cv2.rectangle(img, (x1, y_top), (x1 + tw + 4, y_top + th + base + 2), BOX_COLOR, -1)
        cv2.putText(img, label, (x1 + 2, y_top + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ============================================================
# ESCRITURA ATÓMICA (Windows falla con "Acceso denegado" si otro proceso
# tiene abierto el destino justo en ese momento)
# ============================================================
def replace_con_reintentos(tmp, dst, intentos=5, espera=0.2):
    for i in range(intentos):
        try:
            os.replace(tmp, dst)
            return True
        except PermissionError:
            time.sleep(espera)
    print(f"No pude reemplazar {dst} (en uso por otro proceso); se reintenta en el próximo ciclo")
    return False


def guardar_frame(img, dst):
    """Guarda el JPG de forma atómica y avisa si falla (cv2.imwrite falla en silencio)."""
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        print(f"No pude codificar el frame para {dst}")
        return
    tmp = dst + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
    except Exception as e:
        print(f"No pude escribir {tmp}: {e}")
        return
    replace_con_reintentos(tmp, dst)


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"PC: {PC} | URL: {BASE_URL} | Salida: {OUT_DIR}")
    os.makedirs(f"{OUT_DIR}/frames", exist_ok=True)
    model = YOLO(MODEL)
    counts = {}  # mac -> cantidad; si una cámara falla conserva su último valor

    while True:
        t0 = time.time()
        try:
            cams = http.get(f"{BASE_URL}/list.json", timeout=10).json()
        except Exception as e:
            print(f"No pude leer list.json: {e}")
            time.sleep(INTERVAL_S)
            continue

        with ThreadPoolExecutor(max_workers=10) as ex:
            frames = list(ex.map(grab_frame, cams))
        ok = [(c, f) for c, f in zip(cams, frames) if f is not None]

        if ok:
            results = model.predict([f for _, f in ok], classes=[0], conf=CONF, verbose=False)
            now = datetime.now().isoformat(timespec="seconds")

            for (cam, frame), res in zip(ok, results):
                counts[cam["mac"]] = len(res.boxes)
                guardar_frame(draw_boxes(frame, res), f"{OUT_DIR}/frames/{cam['mac']}.jpg")

            tmp = f"{OUT_DIR}/counts.json.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"ts": now, "counts": counts}, f, ensure_ascii=False, indent=2)
            replace_con_reintentos(tmp, f"{OUT_DIR}/counts.json")

        print(f"{len(ok)}/{len(cams)} cámaras OK en {time.time() - t0:.1f}s")
        time.sleep(max(0, INTERVAL_S - (time.time() - t0)))


if __name__ == "__main__":
    main()
