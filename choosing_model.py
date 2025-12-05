from test_train import testtrain
import matplotlib.pyplot as plt
import numpy as np

#3.0.0 Model Algoritmalarnı Deneme ve Karşılaştırma En İyi Parametreyi Bulma

#En İyi Hyperparametreyi Bulma Kütüphanesi
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

"""
Bu kodda bütün modellerin performans sonuçlari correlasyon matrisi ile kontrol edilmiştir.
GridSearch ile en iyi hiperparametreleri bulacağiz.
Burada ek olarak modelleri ONNX formatinda kaydetme yapilmiştir

"""

class choosing_model(testtrain):

    def __init__(self):

        super().__init__()

        #Konfüsyon matrisini kullanmak için gerekli kütüphane
        from sklearn.metrics import confusion_matrix

        #3.1.0 Logistic Regresyon
        from sklearn.linear_model import LogisticRegression 

        logr = LogisticRegression(random_state=0)
        logr.fit(self.X_train,self.y_train_smote)

        self.y_pred_logr = logr.predict(self.X_test)
        self.cm_logreg = confusion_matrix(self.y_test,self.y_pred_logr)

        print("Logistic regression")
        print(self.cm_logreg)

        #3.2.0 K-Nearest Neighbors
        from sklearn.neighbors import KNeighborsClassifier 

        knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')

        knn.fit(self.X_train,self.y_train_smote)
        self.y_pred_knn =knn.predict(self.X_test)

        
        knn_Score = knn.score(self.X_test,self.y_test)
        print(knn_Score)
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler

        pipe = Pipeline([
            ('std',StandardScaler()),
            ('sel',VarianceThreshold()),
            ('knn',KNeighborsClassifier())
        ])
        pipe.fit(self.X_train,self.y_train_smote)
        pipe_score = pipe.score(self.X_test,self.y_test)
        print(pipe_score)
        
        parameters = {
    'std': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
    'sel__threshold': [0.0, 0.001, 0.01, 0.1],
    'knn__n_neighbors': [3,4, 5,6, 7,8, 9,10, 11,12,13,14,15]
}

        grid = GridSearchCV(pipe, param_grid=parameters, cv=2).fit(self.X_train, self.y_train_smote)

        grid.fit(self.X_train, self.y_train_smote)

        grid_score=grid.score(self.X_test, self.y_test)

        print(grid_score)
"""

        self.cm_knn = confusion_matrix(self.y_test,self.y_pred_knn)

        print("KNN")
        print(self.cm_knn)

        #3.3.0 SVM
        from sklearn.svm import SVC 

        svc= SVC(kernel="rbf") 
        svc.fit(self.X_train,self.y_train_smote)

        self.y_pred_svc = svc.predict(self.X_test)
        self.cm_svc =confusion_matrix(self.y_test,self.y_pred_svc) 

        print("svc")
        print(self.cm_svc)

        #3.4.0 Naive Bayes
        from sklearn.naive_bayes import GaussianNB #Sürekli bir değer ise 
        from sklearn.naive_bayes import MultinomialNB #integer değerler
        from sklearn.naive_bayes import BernoulliNB #ikili var ise

        nb = BernoulliNB()
        nb.fit(self.X_train,self.y_train_smote)

        self.y_pred_nb=nb.predict(self.X_test)
        self.cm_nb =confusion_matrix(self.y_test,self.y_pred_nb) 

        print("Naive Bayes")
        print(self.cm_nb)

        #3.5.0 Decision Tree Classifier 
        from sklearn.tree import DecisionTreeClassifier

        dtc = DecisionTreeClassifier(criterion='gini',class_weight='balanced')
        dtc.fit(self.X_train,self.y_train_smote)

        self.y_pred_dtc = dtc.predict(self.X_test)

        self.cm_dtc =confusion_matrix(self.y_test,self.y_pred_dtc) 

        print("Decision Tree")
        print(self.cm_dtc)

        #3.6.0 Random Forest Classifier
        from sklearn.ensemble import RandomForestClassifier

        rfc = RandomForestClassifier(n_estimators=100, criterion='entropy')
        rfc.fit(self.X_train,self.y_train_smote)

        self.y_pred_rfc= rfc.predict(self.X_test)
        self.cm_rf =confusion_matrix(self.y_test,self.y_pred_rfc)

        print("Rassal Ağaçlar")
        print(self.cm_rf)

        #3.7.0
        from xgboost import XGBClassifier

        self.xgb_model = XGBClassifier(
            objective='binary:logistic', # objective='multi:softmax'  0 1 2 3 4
            n_estimators=50, 
            eval_metric='logloss',       # Binary için 'logloss' veya 'error' daha yaygındır
            random_state=10,
        )

        y_train_flat = np.ravel(self.y_train)
        self.xgb_model.fit(self.X_train, self.y_train_smote)

        self.y_pred_xgb = self.xgb_model.predict(self.X_test)
        self.cm_xgb = confusion_matrix(self.y_test, self.y_pred_xgb)

        print("XGBoost")
        print(self.cm_xgb)

        #3.7.1 Modeli uygulamaya uygun hale getirmek
        import xgboost as xgb
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        # ONNX'e dönüştürme
        initial_type = [('input', FloatTensorType([None, self.X_train.shape[1]]))]
        onnx_model = onnxmltools.convert_xgboost(self.xgb_model, initial_types=initial_type)

        # ONNX dosyasını kaydetme
        onnxmltools.utils.save_model(onnx_model, 'heart_disease_model.onnx')
        

        #3.8.0
        from sklearn.ensemble import GradientBoostingClassifier

        self.gbc = GradientBoostingClassifier(
            n_estimators=70,       # Ağaç sayısı
            learning_rate=0.4,     # Öğrenme hızı
            max_depth=3,           # Ağaç derinliği
            random_state=10
        )

        self.gbc.fit(self.X_train, self.y_train_smote)
        self.y_pred_gbc = self.gbc.predict(self.X_test)

        self.cm_gbc = confusion_matrix(self.y_test, self.y_pred_gbc)

        print("GRADİENT BOOST")
        print(self.cm_gbc)


#########################################################################################################################################################################

