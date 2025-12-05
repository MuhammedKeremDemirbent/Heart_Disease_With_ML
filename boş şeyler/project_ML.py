import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# C# için olan kütüphaneler
import onnx
import onnxruntime as ort
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

#1.0.0 Datanın projeye eklenmesi
#Toplam 10 farklı veri var istenen veri heart_disease.


class dataManipulating:

    def __init__(self):
    
        pure_data = pd.read_csv('heart_disease_datas.csv')

        #1.1.0 Datayı ayırma işlemi

        age= pure_data.iloc[:,1].values  
        age = age.reshape(-1, 1)

        gender= pure_data.iloc[:,2].values 
        gender = gender.reshape(-1, 1)

        dataset = pure_data.iloc[:,3].values 
        dataset = dataset.reshape(-1, 1)

        chest_pain_type = pure_data.iloc[:,4].values
        chest_pain_type = chest_pain_type.reshape(-1, 1)

        trestbps_missing = pure_data.iloc[:,5].values #Eksik var 
        trestbps_missing = trestbps_missing.reshape(-1, 1)

        cholesterol_missing= pure_data.iloc[:,6].values #Eksik var 
        cholesterol_missing = cholesterol_missing.reshape(-1, 1)

        fbs_missing = pure_data.iloc[:,7].values #Eksik var
        fbs_missing = fbs_missing.reshape(-1, 1)

        restecg_missing= pure_data.iloc[:,8].values #Eksik var
        restecg_missing = restecg_missing.reshape(-1, 1)

        thalc_missing=pure_data.iloc[:,9].values #Eksik var 
        thalc_missing = thalc_missing.reshape(-1, 1)

        heart_disease= pure_data.iloc[:,-1].values
        heart_disease = heart_disease.reshape(-1, 1)


        from sklearn.impute import SimpleImputer

        #1.2.0 Eksik verileri düzenleme


        #1.2.1 Nümerik olanlar 
        imputer_numerics = SimpleImputer(missing_values=np.nan, strategy='mean')

        imputer_numerics=imputer_numerics.fit(trestbps_missing)
        trestbps=imputer_numerics.transform(trestbps_missing)

        imputer_numerics=imputer_numerics.fit(cholesterol_missing)
        cholesterol=imputer_numerics.transform(cholesterol_missing)

        imputer_numerics=imputer_numerics.fit(thalc_missing)
        thalc=imputer_numerics.transform(thalc_missing)

        #1.2.2 Kategorik Olanlar
        imputer_categoricals = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

        imputer_categoricals=imputer_categoricals.fit(fbs_missing)
        fbs=imputer_categoricals.transform(fbs_missing)

        imputer_categoricals=imputer_categoricals.fit(restecg_missing)
        restecg=imputer_categoricals.transform(restecg_missing)


        #1.3.0 Kategorik verileri nümerik hale getirme ve sütün isimlerini belirleme
        #Birleştirme işlemini Label Encoder ve OneHotEncoder kullanarak yapacağız

        from sklearn import preprocessing

        #1.3.1 LabelEncoder
        labelEncoding =preprocessing.LabelEncoder()

        gender=labelEncoding.fit_transform(gender)
        gender = gender.reshape(-1, 1)

        dataset=labelEncoding.fit_transform(dataset)
        dataset = dataset.reshape(-1, 1)

        chest_pain_type=labelEncoding.fit_transform(chest_pain_type)
        chest_pain_type = chest_pain_type.reshape(-1, 1)

        fbs=labelEncoding.fit_transform(fbs)
        fbs = fbs.reshape(-1, 1)

        restecg=labelEncoding.fit_transform(restecg)
        restecg = restecg.reshape(-1, 1)

        #1.3.2 OneHotEncoder

        OneHotEncode = preprocessing.OneHotEncoder()

        gender =OneHotEncode.fit_transform(gender).toarray()

        dataset =OneHotEncode.fit_transform(dataset).toarray()

        chest_pain_type =OneHotEncode.fit_transform(chest_pain_type).toarray()

        fbs =OneHotEncode.fit_transform(fbs).toarray()

        restecg =OneHotEncode.fit_transform(restecg).toarray()


        #1.3.3 Sütunlara ad verme işlemi

        #1.3.3.1 Kategoric Olanlar
        gender= pd.DataFrame(data=gender, index = range(920), columns = ['Female','Male'])
        dataset= pd.DataFrame(data=dataset, index = range(920), columns = ['Cleveland','Hungary','Switzerland','VA Long Beach'])
        chest_pain_type= pd.DataFrame(data=chest_pain_type, index = range(920), columns = ['asymptomatic','atypical angina','non-anginal','typical angina'])
        fbs= pd.DataFrame(data=fbs, index = range(920), columns = ['FALSE','TRUE'])
        restecg= pd.DataFrame(data=restecg, index = range(920), columns = ['lv hypertrophy','normal','st-t abnormality'])

        heart_disease=pd.DataFrame(data=heart_disease,index=range(920),columns=["Heart Disease"])  #Hedef veri

        heart_disease['Heart Disease'] = heart_disease['Heart Disease'].apply(lambda x: 1 if x >= 3 else 0)


        #1.3.3.2 Nümerik Olanlar
        age= pd.DataFrame(data=age, index = range(920), columns = ['Age'])
        trestbps= pd.DataFrame(data=trestbps, index = range(920), columns = ['Trestbps'])
        cholesterol= pd.DataFrame(data=cholesterol, index = range(920), columns = ['Cholesterol'])
        thalc= pd.DataFrame(data=thalc, index = range(920), columns = ['Thalc'])


        #1.4.0 Bütün verileri tek set halinde toplama 
        #Makineden istediğimiz veriyi ayrı tutacağız teste sokarken anlaması için (heart_disease bağımlı değişken)

        categoric_datas = pd.concat([gender,dataset,chest_pain_type,fbs,restecg],axis=1)
        numeric_datas= pd.concat([age,trestbps,cholesterol,thalc],axis=1)


        self.last_data_without_target = pd.concat([categoric_datas,numeric_datas],axis=1)

        #1.5.0 Verisetinde olmasına gerek olmayan parametreleri çıkartıyoruz bazı parametreler gereksiz yük oluşturuyor ve modele yardımcı olmuyor.  
        #Gerekli düzeltmeler için P oran testi VIF testlerine gerek duyuldu ve testler sonucu bazı veriler setten çıkartıldı.

        self.last_data_without_target = self.last_data_without_target.drop(columns=[
            'Male',            #Dummy 
            'VA Long Beach',   #Dummy 
            'typical angina',  #Dummy 
            'TRUE',            #Dummy 
            'st-t abnormality',#Dummy 
            'atypical angina', #VIF değeri 4'ten büyük
            'non-anginal',     #VIF değeri 4'ten son
            'Cholesterol'      #VIF değeri 2'den büyük
        ])

        self.last_data=pd.concat([self.last_data_without_target,heart_disease],axis=1)


        covariance = self.last_data_without_target["Age"].cov(self.last_data["Heart Disease"])
        print(covariance)

        #1.5.1 Outlier özellikli olan verileri datasetin dışına çıkartıyoruz

        #1.5.1.1 Çeyrek değerleri ayarlama

        outlier_list = self.last_data_without_target[['Thalc', 'Trestbps', 'Age']]
        self.last_data_without_outliers = self.last_data_without_target.copy()

        for i in outlier_list:
            
            Q1 = outlier_list[i].quantile(0.25)
            Q3 = outlier_list[i].quantile(0.75)
            IQR = Q3 - Q1
            
            alt_sinir = Q1 - 1.5 * IQR
            ust_sinir = Q3 + 1.5 * IQR

            maske = (self.last_data_without_outliers[i] >= alt_sinir) & (self.last_data_without_outliers[i] <= ust_sinir)
            self.last_data_without_outliers = self.last_data_without_outliers[maske]
        
        clean_index = self.last_data_without_outliers.index
        self.heart_disease_without_outliers = heart_disease.loc[clean_index]

        print(len(self.heart_disease_without_outliers))

        #Bir veriden kaç tane olduğu gözüküyor
        print(self.last_data['Heart Disease'].value_counts(normalize=True))
        

        """
        Heart Disease
        0    0.853261
        1    0.146739
        """
       
######################################################################################################################################################################################################################################
#2.0.0 Model Eğitimi 

class testtrain(dataManipulating):
    def __init__(self):
        super().__init__()

        #2.1.0 Verileri train test diye bölme işlemi
        from sklearn.model_selection import train_test_split

        self.x_train, self.x_test,self.y_train, self.y_test = train_test_split(self.last_data_without_outliers,self.heart_disease_without_outliers,test_size=0.35, random_state=10)

        from imblearn.over_sampling import SMOTE
        from collections import Counter # Sınıf dağılımını kontrol etmek için

        #2.2.0 VIF ve SMOTE uygulamaları

        #2.2.1

        print("SMOTE öncesi eğitim verisi dağilimi:", Counter(self.y_train['Heart Disease'].values.ravel()))
        smote = SMOTE(sampling_strategy='auto', random_state=10)

        self.X_train_smote, self.y_train_smote = smote.fit_resample(self.x_train, self.y_train.values.ravel())
        print("SMOTE sonrasi eğitim verisi dağilimi:", Counter(self.y_train_smote))

        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        #2.2.2 VIF Analizi (Verilerin kendi arasındaki etkileşim oranlarını ölçer verilerin VIF değeri 5 ten yüksekse sıkıntı oluyor)

        X_vif_input = sm.add_constant(self.last_data_without_outliers) 

        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif_input.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif_input.values, i) 
                        for i in range(len(X_vif_input.columns))]

        print("\n--- Result of VIF Analysis ---")
        print(vif_data.sort_values(by='VIF', ascending=False).head(30).to_markdown(index=False))
        print("\n--- NOT: OLS (Lineer Regresyon) model analizi, sıralı sınıflama problemi için uygun olmadığı için çıkarılmıştır. ---")

        #2.3.0 Datayı scale edeceğiz bir verinin diğer verileri etkilemesi istemiyoruz 
        #Scale edilecekler nümerik olanlar

        from sklearn.preprocessing import StandardScaler

        StandardScale=StandardScaler()

        self.X_train =StandardScale.fit_transform(self.X_train_smote)
        self.X_test = StandardScale.transform(self.x_test)

    #########################################################################################################################################################################


#3.0.0 Model Algoritmalarnı Deneme ve Karşılaştırma En İyi Parametreyi Bulma

#En İyi Hyperparametreyi Bulma Kütüphanesi
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

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

#4.0.0 Performans sonuçları ve gerekli grafikler

class p_tuning_and_graphs(choosing_model):
    def __init__(self):
        super().__init__()

        import seaborn as sns
        
        #4.1.0 Gradient Boosting parametrelerin önem sırası
        self.X_features = self.x_train.columns 

        importance_df = pd.DataFrame({
            "Feature": self.x_train.columns, 
            "Importance": self.gbc.feature_importances_ 
        }).sort_values(by="Importance", ascending=False)

        print("\n# En önemli özellikler sirasi (Gradient Boosting)):")
        print(importance_df.head(10).to_markdown(index=False))

        plt.figure(figsize=(10,6))
        sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", palette="viridis")
        plt.title("Kalp Hastalığı Risk Faktörlerinin Önemi (Gradient Boosting)")
        plt.xlabel("Parametre Önem Sırası")
        plt.ylabel("Parametre")
        #plt.show()

        #4.2.0 Thalc verisinin dağalımı
        plt.figure(figsize=(12,8))
        sns.histplot(x='Thalc',data=self.last_data_without_outliers)
        plt.title('Distribution of Thalc')
        #plt.show()

        #4.3.0 Korrelasyon matrisi
        #Bu grafik parametreler arası ilişkiyi ölçer
        plt.figure(figsize=(20,30))
        sns.heatmap(self.last_data.corr(),annot=True)
        #plt.show()

        #4.4.0 Logistic Regresyonun konfüsyon matrisinini grafikleştirilmesi
        import seaborn as sns

        plt.figure(figsize=(5,4))
        sns.heatmap(self.cm_logreg, annot=True, cmap="Purples", fmt="d")
        plt.title("Lojistic Regression Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        #plt.show()

        #4.4.1 Gradient Boosting konfüsyon matrisi grafiği
        plt.figure(figsize=(5,4))
        sns.heatmap(self.cm_gbc, annot=True, cmap="Purples", fmt="d")
        plt.title("Gradient Boosting Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        #plt.show()

        #4.5.0 Performans skorları (F1)
        from sklearn.metrics import accuracy_score, f1_score
        from collections import OrderedDict

        #4.5.1 Tüm modelleri ve tahminlerini bir sözlükte toplayalım
        models = OrderedDict([
            ('Logistic Regression', self.y_pred_logr),
            ('KNN', self.y_pred_knn),
            ('SVM', self.y_pred_svc),
            ('Naive Bayes', self.y_pred_nb),
            ('Decision Tree', self.y_pred_dtc),
            ('Random Forest', self.y_pred_rfc),
            ('XGBoost', self.y_pred_xgb),
            ('Gradient Boosting', self.y_pred_gbc)
        ])

        # Sonuçları depolamak için boş bir liste
        performance_results = []

        #4.5.2 Performans metrikleri
        print("\n\n--- Performance Metrics of All Models ---")
        for model_name, y_pred in models.items():
            # Doğruluk (Accuracy)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # F1 Skoru (Çok sınıflı olduğu için 'weighted' ortalama kullanıldı)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            performance_results.append({
                'Model': model_name,
                'Accuracy': f'{accuracy:.4f}',
                'F1 Score': f'{f1:.4f}'
            })

        #4.5.3 Sonuçları DataFrame'e çevirip yazdırma
        sonuc_df = pd.DataFrame(performance_results)
        print(sonuc_df.sort_values(by='F1 Score', ascending=False).to_markdown(index=False))

        # En iyi modeli vurgulayalım
        best_model = sonuc_df.sort_values(by='F1 Score', ascending=False).iloc[0]
        print(f"\n The Best F1 Score: {best_model['Model']} ({best_model['F1 Score']})")


        

        # NOT: Bu kısım, performans sonuçlarının hesaplandığı kodunuzun devamıdır.

        # 4.5.3 Sonuçları DataFrame'e çevirip yazdırma (Mevcut kodunuzdan)
        sonuc_df = pd.DataFrame(performance_results)

        # Accuracy ve F1 Score sütunlarını float tipine çevirelim
        sonuc_df['Accuracy'] = sonuc_df['Accuracy'].astype(float)
        sonuc_df['F1 Score'] = sonuc_df['F1 Score'].astype(float)

        # F1 Skoru'na göre sıralayalım (grafikte düzenli görünmesi için)
        sonuc_df_sorted = sonuc_df.sort_values(by='F1 Score', ascending=False)

        # --- Görselleştirme Kodları ---

        # 1. Grafik Figürünü ve Alt Grafikleri Oluşturma
        # İki metrik (Accuracy ve F1 Score) için yan yana iki sütun grafiği çizelim
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        plt.style.use('ggplot') # Daha güzel bir stil kullanalım

        # 2. Accuracy Skoru Grafiği (Sol)
        sns.barplot(
            x='Accuracy', 
            y='Model', 
            data=sonuc_df_sorted, 
            ax=axes[0], 
            palette='viridis' # Renk paleti
        )
        axes[0].set_title('Modellerin Başarı Skorları (Accuracy)')
        axes[0].set_xlim(0.0, 1.0) # Skorlar 0 ile 1 arasında olduğu için limiti belirleyelim
        axes[0].set_xlabel('Accuracy Skoru')

        # Çubukların üzerine değerleri yazdıralım
        for index, value in enumerate(sonuc_df_sorted['Accuracy']):
            axes[0].text(value + 0.005, index, f'{value:.4f}', va='center')

        # 3. F1 Skoru Grafiği (Sağ)
        sns.barplot(
            x='F1 Score', 
            y='Model', 
            data=sonuc_df_sorted, 
            ax=axes[1], 
            palette='magma' # Farklı bir renk paleti
        )
        axes[1].set_title('Modellerin F1 Skorları')
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_xlabel('F1 Skoru')
        axes[1].set_ylabel('') # Y ekseni etiketi tekrar etmesin

        # Çubukların üzerine değerleri yazdıralım
        for index, value in enumerate(sonuc_df_sorted['F1 Score']):
            axes[1].text(value + 0.005, index, f'{value:.4f}', va='center')

        # Genel Başlık ve Düzenlemeler
        plt.suptitle('Makine Öğrenimi Modellerinin Performans Karşılaştırması', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Grafiklerin sıkışmasını önler
        plt.show()

#
        
if __name__ =="__main__":
    print("Program Başlatılıyor...")
    #DM = dataManipulating()
    #TT = testtrain()
    #CM = choosing_model()
    PTAG = p_tuning_and_graphs()
   
"""
BU SONUÇLAR BİNARY SINIF İÇİN GEÇERLİDİR
PERFORMANS SONUÇLARINA BAKINCA EN KARARLI SONUÇ VEREN
LOGİSTİC REGRESYON ANCAK EN YÜKSEK DORĞRULUK ORANINI
GRADİENT BOOSTİNG XGBOOST VE RANDOM FOREST.
NAİVE BAYES SINIFTA KALDI. ANCAK 
BERNOULLİ OLURSA İYİ SONUÇ VERİYOR HALA EN KÖTÜ AMA 
LOGİSTİC REGRESYONA YAKIN DEĞER VERİYOR



#Samsung İle Alakali Bölüm
        import pickle 

        with open('model.pkl', 'wb') as file:
            pickle.dump(self.xgb_model, file)

"""

#py -m pip install onnxmltools hummingbird-ml onnxruntime
