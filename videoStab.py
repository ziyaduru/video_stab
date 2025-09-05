import cv2
import numpy as np
from kuyruk_kutuphane.kuyruk import Kuyruk
import matplotlib.pyplot as plt
import time
import csv
from dataclasses import dataclass
import pandas as pd

@dataclass
class LogRow:
    frame: int
    t_sec: float
    x_raw: float; y_raw: float; teta_raw: float; scale_raw: float
    x_s: float; y_s: float; teta_s: float; scale_s: float
    dx: float; dy: float; d_teta: float
    inliers: int

class stabLogger:
    def __init__(self, path:str):
        self.f = open(path, mode='w', newline='')
        self.w = csv.writer(self.f)
        self.w.writerow([
            'frame', 't_sec',
            'x_raw', 'y_raw', 'teta_raw', 'scale_raw',
            'x_s', 'y_s', 'teta_s', 'scale_s',
            'dx', 'dy', 'd_teta',
            'inliers'
        ])
        self.n = 0

    def write(self, row:LogRow):

        self.w.writerow([
            row.frame, row.t_sec,
            row.x_raw, row.y_raw, row.teta_raw, row.scale_raw,
            row.x_s, row.y_s, row.teta_s, row.scale_s,
            row.dx, row.dy, row.d_teta,
            row.inliers
            ])
        self.n += 1
        if self.n % 60 == 0:
            self.f.flush()

    def close(self):
        self.f.flush()
        self.f.close()

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0 or np.isnan(fps):
    fps = 30.0
frame_period = 1.0 / fps
    

gftt_params = dict (
    maxCorners = 400,
    qualityLevel = 0.01,
    minDistance = 7,
    blockSize = 7)

lk_params = dict(
    winSize = (21,21),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# TAKIP EDİLECEK İLK NOKTALARI BELİRLEME

ret, old_frame  = cap.read()
old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **gftt_params)

kuyruk_dx = Kuyruk()
kuyruk_dy = Kuyruk()
kuyruk_dteta = Kuyruk()
kuyruk_s = Kuyruk()
ham_kare_kuyruk = Kuyruk()
transform_kuyruk = Kuyruk()



logger = stabLogger("stab_log.csv")
start_t = time.perf_counter()

frame_idx = -1



x = 0
y = 0
aci_toplam = 0
s_toplam = 1.0


while True:
    frame_idx += 1
    t0 = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Kamera kare okunamadı.")
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    h, w = frame.shape[:2]
    
    ## OPTICAL FLOW hesaplaması
    p1,st,err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # sadece durumu 1 olan kareleri seç
    if p1 is not None and st is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
        if good_new.shape[0] < 20:
           logger.write(LogRow(
                        frame=frame_idx
                        t_sec=time.perf_counter() - start_t,
                        x_raw=x, y_raw=y, teta_raw=aci_toplam, scale_raw=s_toplam,
                        x_s=np.nan, y_s=np.nan, teta_s=np.nan, scale_s=np.nan,
                        dx=np.nan, dy=np.nan, d_teta=np.nan,
                        inliers=0
                        ))
           
           p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **gftt_params)
           old_gray = frame_gray.copy() 
           cv2.imshow("STAB",frame)
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
           continue
    else:
        logger.write(LogRow(
                           frame=frame_idx+1,
                            t_sec=time.perf_counter() - start_t,
                            x_raw=x, y_raw=y, teta_raw=aci_toplam, scale_raw=s_toplam,
                            x_s=np.nan, y_s=np.nan, teta_s=np.nan, scale_s=np.nan,
                            dx=np.nan, dy=np.nan, d_teta=np.nan,
                            inliers=0
                            ))
        
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **gftt_params)
        old_gray = frame_gray.copy() 
        cv2.imshow("STAB",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    #bfill rolling mean
    # ORTALAMA HAREKETİ HESAPLA
    
    #M,_ = cv2.estimateAffinePartial2D(good_old, good_new)

    M, inliers = cv2.estimateAffinePartial2D(
        good_old, good_new,
        method = cv2.RANSAC,
        ransacReprojThreshold = 3,
        maxIters = 2000,
        confidence = 0.99,
        refineIters = 10
    )

    if M is None or inliers is None or inliers.sum() < 20:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **gftt_params)
        cv2.imshow("STAB",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        old_gray = frame_gray.copy()
        continue    
    # M 2x3 dönüşüm matrisidir

    # matrisin yapısı
    # M = [[cos(teta), -sin(teta), dx],
    #      [sin(teta), cos(teta), dy]]

    dx = M[0,2]
    dy = M[1,2]
    d_teta = np.arctan2(M[1,0],M[0,0]) ## arctan ile mevcut açı değerini hesaplar
    s = np.hypot(M[0,0], M[1,0])  # ölçek faktörü hesaplama
    #cv2.getRotationMatrix()

    
    
    x += dx
    y += dy
    aci_toplam += d_teta
    s_toplam *= s
    
    

    

    kuyruk_dx.ekle(x)
    kuyruk_dy.ekle(y)
    kuyruk_dteta.ekle(aci_toplam)
    kuyruk_s.ekle(s_toplam)
    


    ##    -----YUMUŞATMA HESAPLAMALARI-----
    pencere_boyutu = 30

    
    pencere_dx = kuyruk_dx.son_N_eleman(pencere_boyutu)
    pencere_dy = kuyruk_dy.son_N_eleman(pencere_boyutu)
    pencere_dteta = kuyruk_dteta.son_N_eleman(pencere_boyutu)
    pencere_s = kuyruk_s.son_N_eleman(pencere_boyutu)
    
    yumusak_x = np.mean(pencere_dx)
    yumusak_y = np.mean(pencere_dy)
    yumusak_aci = np.mean(pencere_dteta)
    yumusak_s   = np.mean(pencere_s) 
    


    inliers_count = int(inliers.sum()) if inliers is not None else 0
    t_sec = time.perf_counter() - start_t

    logger.write(LogRow(
        frame=frame_idx,
        t_sec=t_sec,
        x_raw=x, y_raw=y, teta_raw=aci_toplam, scale_raw=s_toplam,
        x_s=yumusak_x, y_s=yumusak_y, teta_s=yumusak_aci, scale_s=yumusak_s,
        dx=dx, dy=dy, d_teta=d_teta,
        inliers=inliers_count
    ))


    ## ----YUMUŞATMA HESAPLAMALARI SON ----



    ## ---- YENİ DÖNÜŞÜM MATRİSİ OLUŞTURMA ----
    fark_x = yumusak_x - x
    fark_y = yumusak_y - y
    fark_aci = yumusak_aci - aci_toplam
     
    tx = fark_x
    ty = fark_y
    aci = fark_aci
    
    cos_aci = np.cos(aci)
    sin_aci = np.sin(aci)

    fark_s = yumusak_s / s_toplam
    
    M_yumusak = np.float32([[fark_s*cos_aci, -fark_s*sin_aci, tx],
                            [fark_s*sin_aci, fark_s*cos_aci,  ty]])
    transform_kuyruk.ekle(M_yumusak)
    ham_kare_kuyruk.ekle(frame)
    
    stabilize_edilmis_frame = None

    
# Savitzky–Golay filtresi ile gerçek zamanlı faz kayması olmadan yumuşatma yapılabilir
# veya aynı şekilde kalman filtresi de kullanılabilir


    if ham_kare_kuyruk.boyut() >= pencere_boyutu:
        stabilize_edilecek_kare = ham_kare_kuyruk.cikar()
        eski_M_yumusak = transform_kuyruk.cikar()

        stabilize_edilmis_frame = cv2.warpAffine(stabilize_edilecek_kare, eski_M_yumusak, (w,h))
    else: 
        stabilize_edilmis_frame = np.zeros_like(frame)


    #karsilastirma_frame = np.hstack((frame,stabilize_edilmis_frame))
    #cv2.imshow("KARŞILAŞTIRMA",karsilastirma_frame) 
    cv2.imshow("STAB",stabilize_edilmis_frame)

    remain = frame_period - (time.perf_counter() - t0)
    if remain > 0:
        time.sleep(remain)
    
    
    #cv2.estimateAffinePartial2D
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    p0 = good_new.reshape(-1,1,2).astype(np.float32)
    old_gray = frame_gray.copy()    
    if p0.shape[0] < 200:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **gftt_params)
logger.close()    
# --- GRAFİK ÇİZİMİ ----
def plot_from_log(path="stab_log.csv"):
    df = pd.read_csv(path)

    with plt.style.context("ggplot"):
        fig, axs = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
        fig.suptitle("Video Stabilizasyon Log Analizi", fontsize=18)

        # --- 1. X-Y Trajektori ---
        axs[0].plot(df["frame"], df["x_raw"], label="x_raw", alpha=0.6)
        axs[0].plot(df["frame"], df["x_s"], label="x_s (smooth)", linewidth=2)
        axs[0].plot(df["frame"], df["y_raw"], label="y_raw", alpha=0.6)
        axs[0].plot(df["frame"], df["y_s"], label="y_s (smooth)", linewidth=2)
        axs[0].set_ylabel("Piksel Konumu")
        axs[0].legend()
        axs[0].set_title("Ham vs Yumuşatılmış X/Y Trajektori")

        # --- 2. Açı (theta) ---
        axs[1].plot(df["frame"], np.degrees(df["teta_raw"]), label="θ_raw (deg)", alpha=0.6)
        axs[1].plot(df["frame"], np.degrees(df["teta_s"]), label="θ_s (smooth, deg)", linewidth=2)
        axs[1].set_ylabel("Derece")
        axs[1].legend()
        axs[1].set_title("Ham vs Yumuşatılmış Açı (θ)")

        # --- 3. Scale ---
        axs[2].plot(df["frame"], df["scale_raw"], label="scale_raw", alpha=0.6)
        axs[2].plot(df["frame"], df["scale_s"], label="scale_s (smooth)", linewidth=2)
        axs[2].axhline(1.0, linestyle="--", color="gray", linewidth=1)
        axs[2].set_ylabel("Ölçek Faktörü")
        axs[2].legend()
        axs[2].set_title("Ham vs Yumuşatılmış Ölçek")

        # --- 4. Frame-to-frame transform ---
        axs[3].plot(df["frame"], df["dx"], label="dx", alpha=0.7)
        axs[3].plot(df["frame"], df["dy"], label="dy", alpha=0.7)
        axs[3].plot(df["frame"], np.degrees(df["d_teta"]), label="dθ (deg)", alpha=0.7)
        axs[3].set_ylabel("Değişim")
        axs[3].legend()
        axs[3].set_title("Kareler Arası Dönüşümler (dx, dy, dθ)")

        # --- 5. İnliers ---
        axs[4].plot(df["frame"], df["inliers"], label="inliers", color="tab:green")
        axs[4].set_ylabel("Eşleşme Sayısı")
        axs[4].set_xlabel("Frame")
        axs[4].set_title("İnliers Sayısı (RANSAC güvenilirlik)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


plot_from_log("stab_log.csv")

cap.release()
cv2.destroyAllWindows()