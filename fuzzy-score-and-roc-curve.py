#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Veri setlerini yükle
errors_data = pd.read_csv('C:\\Users\Lenovo\Desktop\Fuzzy Ödev\Azure\PdM_errors.csv')
failures_data = pd.read_csv('C:\\Users\Lenovo\Desktop\Fuzzy Ödev\Azure\PdM_failures.csv')
machines_data = pd.read_csv('C:\\Users\Lenovo\Desktop\Fuzzy Ödev\Azure\PdM_machines.csv')
maintenance_data = pd.read_csv('C:\\Users\Lenovo\Desktop\Fuzzy Ödev\Azure\PdM_maint.csv')
telemetry_data = pd.read_csv('C:\\Users\Lenovo\Desktop\Fuzzy Ödev\Azure\PdM_telemetry.csv')

# Verilerin yüklendiğini kontrol et
print("Veri setleri başarıyla yüklendi.")


# In[2]:


# Tarihleri datetime formatına çevir
telemetry_data['datetime'] = pd.to_datetime(telemetry_data['datetime'])
errors_data['datetime'] = pd.to_datetime(errors_data['datetime'])
failures_data['datetime'] = pd.to_datetime(failures_data['datetime'])
maintenance_data['datetime'] = pd.to_datetime(maintenance_data['datetime'])

print("Tarih formatları dönüştürüldü.")


# In[3]:


# Telemetri ile hata kayıtlarını birleştir
merged_data = pd.merge_asof(
    telemetry_data.sort_values('datetime'),
    errors_data.sort_values('datetime'),
    on='datetime',
    by='machineID',
    direction='backward'
)

print("Telemetri ve hata verileri birleştirildi.")


# In[4]:


# Telemetri + hata verileri ile arıza kayıtlarını birleştir
merged_data = pd.merge_asof(
    merged_data.sort_values('datetime'),
    failures_data.sort_values('datetime'),
    on='datetime',
    by='machineID',
    direction='backward'
)

print("Arıza verileri eklendi.")


# In[5]:


# Makine bilgilerini ekle
merged_data = pd.merge(
    merged_data,
    machines_data,
    on='machineID',
    how='left'
)

print("Makine bilgileri eklendi.")


# In[6]:


# Bakım verilerini ekle
merged_data = pd.merge_asof(
    merged_data.sort_values('datetime'),
    maintenance_data.sort_values('datetime'),
    on='datetime',
    by='machineID',
    direction='backward'
)

print("Bakım verileri eklendi.")


# In[7]:


# Birleştirilmiş verinin ilk birkaç satırını göster
print(merged_data.head())


# In[8]:


# Eksik değerleri kontrol et
missing_summary = merged_data.isnull().sum()
print("Eksik değerlerin özeti:")
print(missing_summary)

# Eksik veri oranını hesapla
missing_percentage = (merged_data.isnull().sum() / len(merged_data)) * 100
print("\nEksik veri oranları (%):")
print(missing_percentage)

# Eksik değerleri doldurma veya silme stratejisi
# Sayısal değerleri ortalama ile doldur
numerical_columns = merged_data.select_dtypes(include=['float64', 'int64']).columns
merged_data[numerical_columns] = merged_data[numerical_columns].fillna(merged_data[numerical_columns].mean())

# Kategorik değerleri mod ile doldur
categorical_columns = merged_data.select_dtypes(include=['object']).columns
merged_data[categorical_columns] = merged_data[categorical_columns].fillna(merged_data[categorical_columns].mode().iloc[0])

# Düzeltmeler sonrası eksik değer kontrolü
print("\nEksik değerlerin düzeltme sonrası özeti:")
print(merged_data.isnull().sum())


# In[9]:


import numpy as np

# Gaussian üyelik fonksiyonu tanımı
def gaussian_membership(x, mean, std_dev):
    """
    Gaussian üyelik fonksiyonu.
    x: Değer
    mean: Ortalama
    std_dev: Standart sapma
    """
    return np.exp(-0.5 * ((x - mean) / std_dev) ** 2)


# In[10]:


# Telemetri verilerindeki özellikler
features = ['volt', 'rotate', 'pressure', 'vibration']


# In[11]:


for feature in features:
    # Her bir özellik için mean ve std_dev hesapla
    mean = merged_data[feature].mean()
    std_dev = merged_data[feature].std()

    # Low, Medium, High üyelik fonksiyonları
    merged_data[f'{feature}_low'] = merged_data[feature].apply(lambda x: gaussian_membership(x, mean - std_dev, std_dev))
    merged_data[f'{feature}_medium'] = merged_data[feature].apply(lambda x: gaussian_membership(x, mean, std_dev))
    merged_data[f'{feature}_high'] = merged_data[feature].apply(lambda x: gaussian_membership(x, mean + std_dev, std_dev))


# In[13]:


# Tüm özelliklerin low, medium ve high üyelik fonksiyonlarını görüntüle
membership_columns = [f'{feature}_{level}' for feature in features for level in ['low', 'medium', 'high']]
print(merged_data[membership_columns].head())



# In[22]:


# Üyelik derecelerini normalize ederek fuzzy skor hesaplama
def corrected_fuzzy_simulation(data, features, weights):
    """
    Calculates fuzzy scores with corrected normalization.

    Parameters:
        data (DataFrame): Data containing low, medium, and high membership values.
        features (list): List of feature names (e.g., 'volt', 'rotate').
        weights (list): Normalized weights for low, medium, and high.

    Returns:
        Series: Corrected fuzzy scores.
    """
    weights = np.array(weights) / np.sum(weights)  # Normalize weights
    fuzzy_scores = np.zeros(len(data))
    
    for feature in features:
        # Üyelik derecelerini al
        low_col = f'{feature}_low'
        medium_col = f'{feature}_medium'
        high_col = f'{feature}_high'

        # Üyelik derecelerini normalize et
        total_membership = (
            data[low_col] + 
            data[medium_col] + 
            data[high_col]
        )
        data[low_col] = data[low_col] / total_membership
        data[medium_col] = data[medium_col] / total_membership
        data[high_col] = data[high_col] / total_membership

        # Fuzzy skor hesapla
        fuzzy_scores += (
            data[low_col] * weights[0] +
            data[medium_col] * weights[1] +
            data[high_col] * weights[2]
        )
    
    return fuzzy_scores / fuzzy_scores.max()  # Toplamı normalize et

# Fuzzy skorları yeniden hesapla
weights = [0.3, 0.5, 0.2]
merged_data['corrected_fuzzy_score'] = corrected_fuzzy_simulation(merged_data, features, weights)

# İlk birkaç satırı kontrol et
print(merged_data[['volt', 'rotate', 'pressure', 'vibration', 'corrected_fuzzy_score']].head())


# In[23]:


# ROC analizi tüm fuzzy skorları birleştirerek genel bir tahmin performansı sunuyor.
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# ROC analizi için hedef değişkeni belirleme
# Varsayılan olarak, vibration özelliğinin ortalamasını "arızalı" eşik olarak alıyoruz
merged_data['failure_flag'] = (merged_data['vibration'] > merged_data['vibration'].mean()).astype(int)

# Hedef değişken ve tahmin değerleri
true_labels = merged_data['failure_flag']  # Gerçek etiketler (0 veya 1)
predicted_probabilities = merged_data['corrected_fuzzy_score']  # Tahmin skorları

# ROC eğrisi ve AUC skorunu hesapla
fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
auc_score = roc_auc_score(true_labels, predicted_probabilities)

# ROC eğrisini çiz
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'r--', label='Random Guessing')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.grid()
plt.show()

# AUC skorunu ekrana yazdır
print(f"AUC Score: {auc_score:.2f}")


# In[24]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def plot_roc_curve(feature, true_labels, data):
    """
    ROC eğrisini çizen ve AUC skorunu hesaplayan fonksiyon.
    
    Parameters:
        feature (str): Özellik adı (örn. 'volt').
        true_labels (Series): Gerçek etiketler (0 veya 1).
        data (DataFrame): Özelliğe ait low, medium, high üyelik değerleri.
    """
    # Özellik için low, medium, high fuzzy skorlarının ağırlıklı toplamını kullan
    predicted_probabilities = (
        data[f'{feature}_low'] * weights[0] +
        data[f'{feature}_medium'] * weights[1] +
        data[f'{feature}_high'] * weights[2]
    )

    # ROC eğrisi ve AUC skorunu hesapla
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
    auc_score = roc_auc_score(true_labels, predicted_probabilities)

    # ROC eğrisini çiz
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f}) for {feature}', color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guessing')
    plt.title(f'ROC Curve for {feature.capitalize()}')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend()
    plt.grid()
    plt.show()

    # AUC skorunu ekrana yazdır
    print(f"AUC Score for {feature.capitalize()}: {auc_score:.2f}")


# In[25]:


# Hedef değişkeni oluştur (örneğin vibration ortalamasının üzerini arıza olarak işaretle)
merged_data['failure_flag'] = (merged_data['vibration'] > merged_data['vibration'].mean()).astype(int)

# Gerçek etiketler
true_labels = merged_data['failure_flag']


# In[26]:


# ROC analizi için özellikler listesi
features = ['volt', 'rotate', 'pressure', 'vibration']

# Özellikler üzerinden döngüyle ROC analizi
for feature in features:
    plot_roc_curve(feature, true_labels, merged_data)


# In[ ]:




