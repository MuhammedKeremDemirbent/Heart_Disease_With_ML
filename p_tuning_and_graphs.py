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

        import pandas as pd # Series işlemi için gerekli olabilir

        # --- AYARLAR ---
        navy = '#000080'    # Lacivert
        red = '#FF0000'     # Kırmızı
        yellow = '#FFD700'  # Sarı
        green = '#008000'   # Yeşil
        
        hd_res = (19.2, 10.8)
        lbl_size = 20
        tick_size = 18
        val_size = 22

        total_count = len(self.last_data)

        # ==========================================
        # 1. CİNSİYET (GENDER)
        # ==========================================
        gender_counts = self.last_data['Female'].value_counts()
        male_count = gender_counts.get(0, 0)   
        female_count = gender_counts.get(1, 0) 

        plt.figure(figsize=hd_res)
        bars = plt.bar(['Male', 'Female'], [male_count, female_count], color=[navy, red])
        
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                     int(bar.get_height()), ha='center', va='bottom', fontsize=val_size, fontweight='bold')

        plt.ylabel('Number of Patients', fontsize=lbl_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.show()

        # ==========================================
        # 2. KAYNAK BÖLGE (DATASET SOURCE)
        # ==========================================
        c_cleveland = self.last_data['Cleveland'].sum()
        c_hungary = self.last_data['Hungary'].sum()
        c_switz = self.last_data['Switzerland'].sum()
        c_va = total_count - (c_cleveland + c_hungary + c_switz)

        regions = ['Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach']
        r_counts = [c_cleveland, c_hungary, c_switz, c_va]
        
        plt.figure(figsize=hd_res)
        # Renkler: Lacivert, Yeşil, Kırmızı, Sarı
        bars = plt.bar(regions, r_counts, color=[navy, green, red, yellow])

        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                     int(bar.get_height()), ha='center', va='bottom', fontsize=val_size, fontweight='bold')

        plt.xlabel('Regions', fontsize=lbl_size)
        plt.ylabel('Number of Patients', fontsize=lbl_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.show()

    
        # ==========================================
        # 6. KAN BASINCI (TRESTBPS)
        # ==========================================
        plt.figure(figsize=hd_res)
        plt.hist(self.last_data['Trestbps'], bins=20, color=navy, edgecolor='black')
        
        plt.xlabel('Resting Blood Pressure (mm Hg)', fontsize=lbl_size)
        plt.ylabel('Frequency', fontsize=lbl_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.show()

        # ==========================================
        # 7. MAKSİMUM NABIZ (THALC)
        # ==========================================
        plt.figure(figsize=hd_res)
        plt.hist(self.last_data['Thalc'], bins=20, color=navy, edgecolor='black')

        plt.xlabel('Max Heart Rate Achieved (Thalach)', fontsize=lbl_size)
        plt.ylabel('Frequency', fontsize=lbl_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.show()

        # ==========================================
        # 8. YAŞ (AGE)
        # ==========================================
        plt.figure(figsize=hd_res)
        plt.hist(self.last_data['Age'], bins=20, color=navy, edgecolor='black')

        plt.xlabel('Age', fontsize=lbl_size)
        plt.ylabel('Frequency', fontsize=lbl_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.show()

        # ==========================================
        # 9. HEDEF DEĞİŞKEN (HEART DISEASE)
        # ==========================================
        target_counts = self.last_data['Heart Disease'].value_counts()
        
        plt.figure(figsize=hd_res)
        bars = plt.bar(['Negative (0)', 'Positive (1)'], target_counts.values, color=[green, red])
        
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                     int(bar.get_height()), ha='center', va='bottom', fontsize=val_size, fontweight='bold')

        plt.xlabel('Prediction Target', fontsize=lbl_size)
        plt.ylabel('Number of Patients', fontsize=lbl_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.show()
        
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
        
        plt.style.use('ggplot') 


        plot_df = sonuc_df.sort_values(by='F1 Score', ascending=False).set_index('Model')

        plot_df = plot_df[['F1 Score', 'Accuracy', 'Recall', 'Precision']]

  
        plt.figure(figsize=(18, 8)) # Genişlik ve Yükseklik
        ax = plot_df.plot(kind='bar', 
                          figsize=(16, 8), 
                          width=0.8,           # Sütun genişliği
                          edgecolor='black',   # Kenar çizgileri
                          rot=0)               # Yazıları yatay tut (0 derece)

        plt.title("Comparison of ML Models - All Metrics", fontsize=16, fontweight='bold')
        plt.ylabel("Score", fontsize=14)
        plt.xlabel("Models", fontsize=14)
        
        plt.ylim(0, 1.15)

        plt.grid(axis='y', linestyle='--', alpha=0.5)

        plt.legend(loc='upper right', frameon=True, fontsize=11, title="Metrics")

        for container in ax.containers:
            ax.bar_label(container, 
                         fmt='%.2f',       
                         padding=3,       
                         fontsize=10,      
                         fontweight='bold') 

        plt.tight_layout()
        plt.show()

        def draw_single_barplot(x_col, title, palette):
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=x_col,
                y='Model',
                data=sonuc_df_sorted,
                palette=palette
            )
            plt.title(title)
            plt.xlim(0.0, 1.1)
            plt.xlabel("Score")
            plt.ylabel('Models')

            for index, value in enumerate(sonuc_df_sorted[x_col]):
                plt.text(value + 0.005, index, f'{value:.4f}', va='center', fontsize=10, fontweight='bold')

            plt.tight_layout()
            plt.show()   # 👈 BURASI ÖNEMLİ (sıralı çalışır)

    # 1️⃣ F1 Score (ilk olarak)
        draw_single_barplot('F1 Score', 'F1 Score', 'magma')

    # 2️⃣ Accuracy
        draw_single_barplot('Accuracy', 'Accuracy Score', 'viridis')

    # 3️⃣ Recall
        draw_single_barplot('Recall', 'Recall Score', 'rocket')

    # 4️⃣ Precision
        draw_single_barplot('Precision', 'Precision Score', 'mako')

    

        #Accuracy F1 Recall Precision

    """
    Accuracy: Modelin doğru tahminlerinin toplam tahminlere oranıdır. Ancak, dengesiz veri setlerinde yanıltıcı olabilir.
    F1 Score: Precision ve Recall'un harmonik ortalamasıdır. Dengesiz
    veri setlerinde daha güvenilir bir performans ölçütüdür.
    Recall: Modelin gerçek pozitifleri ne kadar iyi yakaladığını gösterir. Kritik durumlarda önemlidir.
    Precision: Modelin pozitif tahminlerinin ne kadarının doğru olduğunu gösterir. Yanlış pozitiflerin maliyetli olduğu durumlarda önemlidir.        
    """