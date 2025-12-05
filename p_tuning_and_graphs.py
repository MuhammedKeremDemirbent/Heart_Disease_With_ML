from choosing_model import choosing_model
import matplotlib.pyplot as plt
import pandas as pd

#4.0.0 Performans sonuçları ve gerekli grafikler


"""
Bu kodda elde edilen bütün sonuçlarin görsellemesi yapilmiştir. Çiktilarin correlasyon matrisleri, correlasyon katsayilari,
parametrelerin(thalc) dağalimlari performans metrik skorlari ve sirasi ele alinmiştir.

"""

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
        plt.close()

        #4.2.0 Thalc verisinin dağalımı
        plt.figure(figsize=(12,8))
        sns.histplot(x='Thalc',data=self.last_data_without_outliers)
        plt.title('Distribution of Thalc')
        #plt.show()
        plt.close()

        #4.3.0 Korrelasyon matrisi
        #Bu grafik parametreler arası ilişkiyi ölçer
        plt.figure(figsize=(20,30))
        sns.heatmap(self.last_data.corr(),annot=True)
        #plt.show()
        plt.close()

        #4.4.0 Logistic Regresyonun konfüsyon matrisinini grafikleştirilmesi
        import seaborn as sns

        plt.figure(figsize=(5,4))
        sns.heatmap(self.cm_logreg, annot=True, cmap="Purples", fmt="d")
        plt.title("Lojistic Regression Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        #plt.show()
        plt.close()

        #4.4.1 Gradient Boosting konfüsyon matrisi grafiği
        plt.figure(figsize=(5,4))
        sns.heatmap(self.cm_gbc, annot=True, cmap="Purples", fmt="d")
        plt.title("Gradient Boosting Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        #plt.show()
        plt.close()

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

        sonuc_df = pd.DataFrame(performance_results)

        sonuc_df['Accuracy'] = sonuc_df['Accuracy'].astype(float)
        sonuc_df['F1 Score'] = sonuc_df['F1 Score'].astype(float)

        sonuc_df_sorted = sonuc_df.sort_values(by='F1 Score', ascending=False) #Büyükten küçüğe sıralama
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        plt.style.use('ggplot') # Daha güzel bir stil 

        
        sns.barplot(
            x='Accuracy', 
            y='Model', 
            data=sonuc_df_sorted, 
            ax=axes[0], 
            palette='viridis' # Renk paleti
        )
        axes[0].set_title('Modellerin Başarı Skorları (Accuracy)')
        axes[0].set_xlim(0.0, 1.0) 
        axes[0].set_xlabel('Accuracy Skoru')

        for index, value in enumerate(sonuc_df_sorted['Accuracy']):
            axes[0].text(value + 0.005, index, f'{value:.4f}', va='center')

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
        axes[1].set_ylabel('') 

        for index, value in enumerate(sonuc_df_sorted['F1 Score']): # Değer yazdırma
            axes[1].text(value + 0.005, index, f'{value:.4f}', va='center')

        # Genel Başlık ve Düzenlemeler
        plt.suptitle('Makine Öğrenimi Modellerinin Performans Karşılaştırması', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) #Grafiklerin sıkışmasını önlemek için 
        #plt.show()


########################################################################################################################################################