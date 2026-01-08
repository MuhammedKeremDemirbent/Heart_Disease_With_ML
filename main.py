from p_tuning_and_graphs import p_tuning_and_graphs

"""
author: Muhammed Kerem Demirbent
        Mustafa Berat Yavaş

Bu projenin amaci kalp krizi riskini ölçmektir. Alinan 9 parametrenin bir kismi kullanilarak modeller eğitilmiştir 
ve regresyon yapilmiştir.
Dataset unbalance bir yapiya sahiptir ve performans sonuçlari biraz olsun düşüktür. (0.84)
"""

if __name__ =="__main__":
    print("Program Başlatiliyor...")
    PTAG = p_tuning_and_graphs()
    
    print("\nProgram Başariyla Tamamlandi. Grafikler gösterildi...")

    """
        İlk olarak data_manipulating.py dosyasindaki dataManipulating sinifi çağrilir. 
        Bu sinif veri setini yükler, eksik verileri doldurur ve kategorik değişkenleri kodlar.

        İkinci olarak test_train.py dosyasindaki testtrain sinifi çağrilir.
        Bu sinif, veri setini eğitim ve test setlerine böler ve SMOTE kullanarak eğitim setindeki dengesizliği giderir.

        Üçüncü olarak choosing_model.py dosyasindaki choosing_model sinifi çağrilir.
        Bu sinif, çeşitli makine öğrenimi modellerini eğitir ve performanslarini karşilaştirir.

        Son olarak p_tuning_and_graphs.py dosyasindaki p_tuning_and_graphs sinifi çağrilir.
        Bu sinif, modellerin hiperparametrelerini ayarlar ve performans sonuçlarini grafiklerle görselleştirir.

        Bu yapiyla, kalp krizi riskini tahmin etmek için kapsamli bir makine öğrenimi süreci gerçekleştirilir.
    
    """