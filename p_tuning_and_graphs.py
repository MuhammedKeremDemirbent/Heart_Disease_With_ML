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

        #4.5.0 Performans skorları (Accuracy, F1, Recall, Precision)
        # GÜNCELLEME: Recall ve Precision kütüphaneleri eklendi
        from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
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
            #4.5.3 Modellerin performans metriklerini hesaplama
            accuracy = accuracy_score(self.y_test, y_pred)          
            f1 = f1_score(self.y_test, y_pred, average='weighted')      
            recall = recall_score(self.y_test, y_pred, average='weighted')
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            performance_results.append({
                'Model': model_name,
                'Accuracy': f'{accuracy:.4f}',
                'F1 Score': f'{f1:.4f}',
                'Recall': f'{recall:.4f}',     
                'Precision': f'{precision:.4f}' 
            })

        #4.5.3 Sonuçları DataFrame'e çevirip yazdırma
        sonuc_df = pd.DataFrame(performance_results)
        print(sonuc_df.sort_values(by='F1 Score', ascending=False).to_markdown(index=False))

        # En iyi modeli vurgulayalım
        best_model = sonuc_df.sort_values(by='F1 Score', ascending=False).iloc[0]
        print(f"\n The Best F1 Score: {best_model['Model']} ({best_model['F1 Score']})")

        sonuc_df = pd.DataFrame(performance_results)

        # String'den float'a çevirme (Grafik için)
        sonuc_df['Accuracy'] = sonuc_df['Accuracy'].astype(float)
        sonuc_df['F1 Score'] = sonuc_df['F1 Score'].astype(float)
        sonuc_df['Recall'] = sonuc_df['Recall'].astype(float)       
        sonuc_df['Precision'] = sonuc_df['Precision'].astype(float) 

        sonuc_df_sorted = sonuc_df.sort_values(by='F1 Score', ascending=False) # F1'e göre sıralama
        
        # GÜNCELLEME: Grafik alanı 2x2 (4'lü) olacak şekilde ayarlandı
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten() 
        
        plt.style.use('ggplot') 

        # Grafik çizdirmek için yardımcı fonksiyon 
        def draw_barplot(ax, x_col, title, palette):
            sns.barplot(
                x=x_col, 
                y='Model', 
                data=sonuc_df_sorted, 
                ax=ax, 
                palette=palette
            )
            ax.set_title(title)
            ax.set_xlim(0.0, 1.1) 
            ax.set_xlabel(title)
            ax.set_ylabel('')
            
            # Değerleri çubukların ucuna yazdırma
            for index, value in enumerate(sonuc_df_sorted[x_col]):
                ax.text(value + 0.005, index, f'{value:.4f}', va='center', fontsize=10, fontweight='bold')

        # Modelin toplam doğruluk oranı (Yanıltıcı)
        draw_barplot(axes[0], 'Accuracy', 'Accuracy Score', 'viridis')

        # Modelin F1 skoru RECALL İLE PRECİSİON ARASINDAKİ UYUM
        draw_barplot(axes[1], 'F1 Score', 'F1 Score', 'magma')

        
        draw_barplot(axes[2], 'Recall', 'Recall Score', 'rocket')

       
        draw_barplot(axes[3], 'Precision', 'Precision Score', 'mako')

        # Genel Başlık ve Düzenlemeler
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)
        plt.show()






        #Accuracy F1 Recall Precision

        """
        Accuracy: Modelin doğru tahminlerinin toplam tahminlere oranıdır. Ancak, dengesiz veri setlerinde yanıltıcı olabilir.
        F1 Score: Precision ve Recall'un harmonik ortalamasıdır. Dengesiz
        veri setlerinde daha güvenilir bir performans ölçütüdür.
        Recall: Modelin gerçek pozitifleri ne kadar iyi yakaladığını gösterir. Kritik durumlarda önemlidir.
        Precision: Modelin pozitif tahminlerinin ne kadarının doğru olduğunu gösterir. Yanlış pozitiflerin maliyetli olduğu durumlarda önemlidir.        
        """