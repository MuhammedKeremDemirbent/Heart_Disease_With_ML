#from data_manipulating import dataManipulating
#from test_train import testtrain
#from choosing_model import choosing_model
from p_tuning_and_graphs import p_tuning_and_graphs
from deep_learning import deep_learning
"""
author: Muhammed Kerem Demirbent
        Mustafa Berat Yavaş

Bu projenin amaci kalp krizi riskini ölçmektir. Alinan 9 parametrenin bir kismi kullanilarak modeller eğitilmiştir 
ve regresyon yapilmiştir.
Dataset unbalance bir yapiya sahiptir ve performans sonuçlari biraz olsun düşüktür. (0.84)
"""

if __name__ =="__main__":
    print("Program Başlatiliyor...")
    #DM = dataManipulating()
    #TT = testtrain()
    #CM = choosing_model()
    PTAG = p_tuning_and_graphs()
    print("\nProgram Başariyla Tamamlandi. Grafikler gösterildi...")
    print("Deep Learning başlatiliyor")

    DL = deep_learning()
    print("Deep Learning bitti")


    #Yorum satırı olan yerlere gerek yok PTAG hepsini miras aldı

    