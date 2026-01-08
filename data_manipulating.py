import numpy as np
import pandas as pd

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