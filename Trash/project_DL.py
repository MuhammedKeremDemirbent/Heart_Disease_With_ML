import project_ML as ml
import keras

from keras.models import Sequential #YSA yı açıklıyoruz oluşturmak için
from keras.layers import Dense      #Katmanlar

def DL():

    #dm_nesnesi = ml.dataManipulating()
    tt_nesnesi = ml.testtrain()
    ptag_nesnesi = ml.p_tuning_and_graphs()
    
    classifier = Sequential() #YSA tanımladık
    input_sayisi = tt_nesnesi.X_train.shape[1]
    #Nöron network
    classifier.add(Dense(5, kernel_initializer= 'uniform', activation= 'relu' , input_dim = input_sayisi)) #giriş katmanı
    classifier.add(Dense(5, kernel_initializer= 'uniform', activation= 'relu')) #gizli katmanlar
    classifier.add(Dense(1, kernel_initializer= 'uniform', activation= 'sigmoid')) # çıkış katmanı

    classifier.compile(optimizer='adam' , loss= 'binary_crossentropy' , metrics=['accuracy']) #Farklı optimizerler var categorical_crossentropy ******

    classifier.fit(tt_nesnesi.X_train,tt_nesnesi.y_train_smote , epochs=500)
    #print(f"Eğitim Verisi Boyutu: {tt_nesnesi.X_train.shape}")

    from sklearn.metrics import confusion_matrix

    y_pred_DL = classifier.predict(tt_nesnesi.X_test)
    y_pred = (y_pred_DL > 0.5)

    cm_DL = confusion_matrix(tt_nesnesi.y_test,y_pred)
    print(cm_DL)


if __name__ == "__main__":
    DL()    



    """
    [[215  58]
    [ 11  28]]
    """