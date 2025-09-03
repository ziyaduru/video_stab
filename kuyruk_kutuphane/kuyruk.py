class Kuyruk:
    # kuyruk() yazıldıgında otomatik olarak çağrılacak.
    def __init__(self):
        self.elemanlar = []  
        
    def ekle(self,yeni_eleman):
        self.elemanlar.append(yeni_eleman)
        
        
    def cikar(self):
        
       if self.elemanlar:
           cikan_eleman = self.elemanlar.pop(0) 
           return cikan_eleman
       else:
           print("Liste boş")
           return None
    def bos_mu(self):
        return not self.elemanlar # bos ise true donecek.
    def boyut(self):
        kuyruk_boyutu = len(self.elemanlar)
        return kuyruk_boyutu
    def gozat(self):
        
        if self.elemanlar:
            return self.elemanlar
        else:
            print("Uyarı: kuyruk boş, burada bakabileceğin bir şey yok gardaş ")
            return None
    
    def __str__(self):
        kuyruk_icerik = ", ".join(str(e) for e in self.elemanlar)
        return f"Kuyruk : [{kuyruk_icerik}]"
    
    def son_N_eleman(self, N):
        return self.elemanlar[-N:]
    
    
        
    
    
        
       
        
        
    