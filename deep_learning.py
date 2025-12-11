from p_tuning_and_graphs import p_tuning_and_graphs
from test_train import testtrain
import keras
from sklearn.metrics import confusion_matrix

from keras.models import Sequential #YSA yı açıklıyoruz oluşturmak için
from keras.layers import Dense      #Katmanlar

#En iyi epoch batch değeri bulmak için 
from keras.callbacks import EarlyStopping


class deep_learning(testtrain):
    def __init__(self):
        super().__init__()
    
        classifier = Sequential() #YSA tanımladık
        input_sayisi = self.X_train.shape[1]

        """
        es = EarlyStopping(
            monitor='val_loss', # İzlenecek metrik (Genellikle doğrulama kaybı)
            mode='min',         # Kaybın minimuma inmesini bekleriz
            verbose=1,          # Durduğunda bilgi yazdırır
            patience=500         # 20 epoch boyunca iyileşme olmazsa eğitimi durdur.
        )"""
        #Nöron network

        #from keras.regularizers import l2

        classifier.add(Dense(5, kernel_initializer= 'uniform', activation= 'relu' , input_dim = input_sayisi)) #giriş katmanı
        classifier.add(Dense(5, kernel_initializer= 'uniform', activation= 'relu')) #gizli katmanlar,        
        classifier.add(Dense(1, kernel_initializer= 'uniform', activation= 'sigmoid')) # çıkış katmanı

        classifier.compile(optimizer='adam' , loss= 'binary_crossentropy' , metrics=['accuracy']) #Farklı optimizerler var categorical_crossentropy ******

        classifier.fit(
            self.X_train, 
            self.y_train_smote, 
            batch_size=32, 
            epochs=1000,           
            #validation_split=0.1, 
            #callbacks=[es]        
        )
        #print(f"Eğitim Verisi Boyutu: {self.X_train.shape}")
      
        y_pred_DL = classifier.predict(self.X_test)
        y_pred = (y_pred_DL > 0.5)

        cm_DL = confusion_matrix(self.y_test,y_pred)
        print(cm_DL)

"""
Skor .75 olmasina rağmen az olan sinifi iyi biliyor

"""