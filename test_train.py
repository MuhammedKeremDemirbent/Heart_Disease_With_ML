from data_manipulating import dataManipulating
import pandas as pd
import matplotlib.pyplot as plt
#2.0.0 Model Eğitimi 

"""
Bu kodda modelin eğitilmesi amaçlanmiştir. OLS model analizi uygun olmadiği kullailmamiştir. VİF analiz kullanilmiştir.
En sonda Standart Scaler ile ölçeklenmiştir.
"""

class testtrain(dataManipulating):
    def __init__(self):
        super().__init__()

        #2.1.0 Verileri train test diye bölme işlemi
        from sklearn.model_selection import train_test_split

        self.x_train, self.x_test,self.y_train, self.y_test = train_test_split(self.last_data_without_outliers,self.heart_disease_without_outliers,test_size=0.35, random_state=10)

        from imblearn.over_sampling import SMOTE
        from collections import Counter #Sınıf dağılımını kontrol etmek için

        #2.2.0 VIF ve SMOTE uygulamaları

        #2.2.1
        print("SMOTE öncesi eğitim verisi dağilimi:", Counter(self.y_train['Heart Disease'].values.ravel()))
        smote = SMOTE(sampling_strategy='auto', random_state=10)

        self.X_train_smote, self.y_train_smote = smote.fit_resample(self.x_train, self.y_train.values.ravel())
        print("SMOTE sonrasi eğitim verisi dağilimi:", Counter(self.y_train_smote))

        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        #2.2.2 VIF Analizi (Verilerin kendi arasındaki etkileşim oranlarını ölçer verilerin VIF değeri 5 ten yüksekse sıkıntı oluyor)

        
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import matplotlib.pyplot as plt

        X_vif_input = sm.add_constant(self.last_data_without_outliers) 
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif_input.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif_input.values, i) for i in range(len(X_vif_input.columns))]
        
        vif_data = vif_data[vif_data['feature'] != 'const']
        vif_data = vif_data.sort_values(by='VIF', ascending=True)

        plt.figure(figsize=(10, 8))
        colors = ['red' if x > 4 else 'mediumseagreen' for x in vif_data['VIF']]
        bars = plt.barh(vif_data['feature'], vif_data['VIF'], color=colors)
        plt.axvline(x=4, color='orange', linestyle='--', linewidth=2, label='Threshold (4.0)')
        
        plt.title('VIF (Variance Inflation Factor) Analysis', fontsize=14)
        plt.xlabel('VIF Value (> 4 High Multicollinearity Risk)', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.legend(loc='lower right')
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')

        plt.tight_layout()
        plt.show()
        
        print("\n--- VIF Grafiği Oluşturuldu ---")
        print("\n--- NOT: OLS (Lineer Regresyon) model analizi, sirali siniflama problemi için uygun olmadiği için çikarilmiştir. ---")

        #2.3.0 Datayı scale edeceğiz bir verinin diğer verileri etkilemesi istemiyoruz 
        #Scale edilecekler nümerik olanlar

        from sklearn.preprocessing import StandardScaler

        StandardScale=StandardScaler()

        self.X_train =StandardScale.fit_transform(self.X_train_smote)
        self.X_test = StandardScale.transform(self.x_test)

#########################################################################################################################################################################################
#stratify = y
