import cv2
import numpy as np
from kuyruk_kutuphane.kuyruk import Kuyruk
import matplotlib.pyplot as plt

def plot_trajectory(titrek_yol, purusuz_yol):
   if titrek_yol is None:
       raise AttributeError("çizilecek bir yörünge verisi yok") 
   with plt.style.context('ggplot'):
       fig, (ax1, ax2) = plt.subplots(nrows=2, sharex='all')
       fig.suptitle("Kamera Yörüngesi",fontsize=16)

       ax1.plot(titrek_yol[:, 0], label='Titrek Yol (X)')
       ax1.plot(purusuz_yol[:, 0], label='Pürüzsüz Yol (X)')
       ax1.set_ylabel("X piksel konumu")
       ax1.legend()

       ax2.plot(titrek_yol[:, 1], label='Titrek Yol (Y)')
       ax2.plot(purusuz_yol[:, 1], label='Pürüzsüz Yol (Y)')
       ax2.set_xlabel("Kare Sayısı")
       ax2.set_ylabel("Y Piksel Konumu")
       ax2.legend()
       fig.canvas.manager.set_window_title('Yörünge Grafiği')

def plot_transforms(donusumler, radians=False):
    if donusumler is None:
        raise AttributeError("Çizilecek bir dönüşüm verisi bulunamadı.")
    
    with plt.style.context('ggplot'):
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex='all')
        fig.suptitle('Kareler Arası Dönüşümler', fontsize=16)

        # X ve Y Dönüşüm Grafiği
        ax1.plot(donusumler[:, 0], label='dx (Piksel)')
        ax1.plot(donusumler[:, 1], label='dy (Piksel)')
        ax1.set_ylabel("Piksel Değişimi")
        ax1.legend()
    
        # Açısal Dönüşüm Grafiği
        if radians:
            ax2.plot(donusumler[:, 2], label='d_aci (Radyan)')
            ax2.set_ylabel("Radyan Değişimi")
        else:
            # Radyanı dereceye çevirmek daha anlaşılır olur
            ax2.plot(np.degrees(donusumler[:, 2]), label='d_aci (Derece)')
            ax2.set_ylabel("Derece Değişimi")
            
        ax2.set_xlabel("Kare Sayısı")
        ax2.legend()

        fig.canvas.manager.set_window_title('Dönüşüm Grafiği')  
    



cap = cv2.VideoCapture(0)


gftt_params = dict (
    maxCorners = 400,
    qualityLevel = 0.1,
    minDistance = 7,
    blockSize = 7)

lk_params = dict(
    winSize = (15,15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
color = np.random.randint(0,255, (100,3))

# TAKIP EDİLECEK İLK NOKTALARI BELİRLEME

ret, old_frame  = cap.read()
old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **gftt_params)

mask = np.zeros_like(old_frame)

kuyruk_dx = Kuyruk()
kuyruk_dy = Kuyruk()
kuyruk_dteta = Kuyruk()

ham_kare_kuyruk = Kuyruk()
transform_kuyruk = Kuyruk()

transform_plot_data = []
trajectory_plot_data = []
smoothed_plot_data = []


x = 0
y = 0
aci_toplam = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    h, w = frame.shape[:2]
    
    ## OPTICAL FLOW hesaplaması
    p1,st,err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # sadece durumu 1 olan kareleri seç
    if p1 is not None and st is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    else:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **gftt_params)
        cv2.imshow("Stabilize görüntü",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    ## video Stabilizasyonu
    stabilize_frame = frame.copy()
    #bfill rolling mean
    # ORTALAMA HAREKETİ HESAPLA
    
    M,_ = cv2.estimateAffinePartial2D(good_old, good_new)

    # matrisin yapısı
    # M = [[cos(teta), -sin(teta), dx],
    #      [sin(teta), cos(teta), dy]]

    dx = M[0,2]
    dy = M[1,2]
    d_teta = np.arctan2(M[1,0],M[0,0]) ## arctan ile mevcut açı değerini hesaplar
    #cv2.getRotationMatrix()
    transform_plot_data.append([dx,dy,d_teta])
    x += dx
    y += dy
    aci_toplam += d_teta
    trajectory_plot_data.append([x,y,aci_toplam])
    

    kuyruk_dx.ekle(x)
    kuyruk_dy.ekle(y)
    kuyruk_dteta.ekle(aci_toplam)
    


    ##    -----YUMUŞATMA HESAPLAMALARI-----
    pencere_boyutu = 30
    yumusak_dx = 0
    yumusak_dy = 0
    yumusak_dteta = 0
    
    pencere_dx = kuyruk_dx.son_N_eleman(pencere_boyutu)
    pencere_dy = kuyruk_dy.son_N_eleman(pencere_boyutu)
    pencere_dteta = kuyruk_dteta.son_N_eleman(pencere_boyutu)
    
    yumusak_x = np.mean(pencere_dx)
    yumusak_y = np.mean(pencere_dy)
    yumusak_aci = np.mean(pencere_dteta)
    smoothed_plot_data.append([yumusak_x, yumusak_y, yumusak_aci])


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
    
    M_yumusak = np.float32([[cos_aci, -sin_aci, tx],
                            [sin_aci, cos_aci,   ty]])
    transform_kuyruk.ekle(M_yumusak)
    ham_kare_kuyruk.ekle(frame)
    
    stabilize_edilmis_frame = None

    if kuyruk_dx.boyut() >= pencere_boyutu:
        stabilize_edilecek_kare = ham_kare_kuyruk.cikar()
        eski_M_yumusak = transform_kuyruk.cikar()

        stabilize_edilmis_frame = cv2.warpAffine(stabilize_edilecek_kare, eski_M_yumusak, (w,h))
    else: 
        stabilize_edilmis_frame = np.zeros_like(frame)


    karsilastirma_frame = np.hstack((frame,stabilize_edilmis_frame))
    #cv2.imshow("KARŞILAŞTIRMA",karsilastirma_frame) 
    cv2.imshow("STAB",stabilize_edilmis_frame)
    
    #cv2.estimateAffinePartial2D
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    
    old_gray = frame_gray.copy()    
    p0 = good_new.reshape(-1,1,2)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **gftt_params)
    
# --- GRAFİK ÇİZİMİ ----

titrek_yol_dizisi = np.array(trajectory_plot_data)
puruzsuz_yol_dizisi = np.array(smoothed_plot_data)
donusumer_dizisi = np.array(transform_plot_data)

plot_trajectory(titrek_yol_dizisi, puruzsuz_yol_dizisi)

plot_transforms(donusumer_dizisi)
plt.show()        

cap.release()
cv2.destroyAllWindows()
