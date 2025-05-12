```markdown
# IRIS Veriseti Sınıflandırma ve Öznitelik Katkısı Analizi

Arkadaşlar, bu çalışmada popüler IRIS verisetini kullanarak bir sınıflandırma problemi çözeceğiz.
Veri setindeki çiçek türlerini (setosa, versicolor, virginica) tahmin etmeye çalışacağız.
Bunun için 5 farklı makine öğrenmesi algoritması kullanacak ve performanslarını karşılaştıracağız.
Sonrasında ise hangi özniteliklerin (sepal length, sepal width, petal length, petal width) model kararlarına ne kadar etki ettiğini 2 farklı yöntemle inceleyeceğiz.

Hazırsanız, başlayalım!

## 1. Gerekli Kütüphanelerin İçe Aktarılması

Arkadaşlar, projemize başlamadan önce kullanacağımız kütüphaneleri Python ortamımıza dahil etmemiz gerekiyor.

```python
# Veri işleme ve sayısal operasyonlar için temel kütüphaneler arkadaşlar
import numpy as np  # Numpy, sayısal hesaplamalar için çok kuulanışlıdır
import pandas as pd # Pandas, veri çerçeveleri (DataFrame) ile çalışmak için harikadır

# Veri setlerini yüklemek ve bölmek için sklearn modülleri
from sklearn.datasets import load_iris # IRIS verisetini yüklemek için bu fonksiyonu kullanıcaz
from sklearn.model_selection import train_test_split # Verisetini eğitim ve test olarak ayırmak için gereklidir arkadaşlar

# Sınıflandırma algoritmalarımız
from sklearn.linear_model import LogisticRegression # Lojistik Regresyon, basit ama güçlü bir lineer modeldir
from sklearn.neighbors import KNeighborsClassifier # K-En Yakın Komşu, örnek tabanlı bir öğrenme algoritmasıdır
from sklearn.svm import SVC # Destek Vektör Makineleri, karmaşık sınırlar çizebilir
from sklearn.tree import DecisionTreeClassifier # Karar Ağaçları, anlaşılması kolay ve yorumlanabilir modellerdir
from sklearn.ensemble import RandomForestClassifier # Rastgele Orman, birden fazla karar ağacının birleşimidir, genelde daha iyidir

# Model performansını değerlendirmek için metrikler
from sklearn.metrics import accuracy_score, classification_report # Doğruluk ve detaylı sınıflandırma raporu için bunları kullanacaz

# Öznitelik katkısını göstermek için
from sklearn.inspection import permutation_importance # Permütasyon önemi, model agnostik bir yöntemdir arkadaşlar
import matplotlib.pyplot as plt # Görselleştirme için Matplotlib'i kullanıcaz
import seaborn as sns # Daha şık görseller için Seaborn kütüphanesi de iyidr

# Uyarıları bastırmak için (isteğe bağlı, daha temiz çıktı için)
import warnings # Bazen çıkan uyarıları gizlemek isteyebiliriz
warnings.filterwarnings('ignore') # Bu satır uyarıları kapatır arkadaşlarım
```

## 2. Veri Setinin Yüklenmesi ve Hazırlanması

Arkadaşlar, şimdi IRIS verisetini yükleyip özellikler (X) ve hedef değişken (y) olarak ayıralım.
Ayrıca, verisetini eğitim ve test alt kümelerine böleceğiz. Bu, modelimizin görmediği veriler üzerindeki performansını ölçmek için çok önemlidir.

```python
# IRIS verisetini yükleyelim arkadaşlar
iris = load_iris() # Scikit-learn içerisinden hazır verisetini çektik
X = iris.data # X, özniteliklerimizi (sepal length/width, petal length/width) içerir
y = iris.target # y, hedef değişkenimizdir (çiçek türleri: 0, 1, 2)
feature_names = iris.feature_names # Öznitelik isimlerini de alalım, lazım olacak

# Veri setini eğitim ve test olarak ayıralım
# test_size=0.3 demek, verinin %30'u test için, %70'i eğitim için ayrılacak demektr
# random_state, sonuçların tekrarlanabilir olması için kullanılır, bir nevi tohum değeridir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# stratify=y, sınıfların oranını hem eğitim hem de test setinde korumaya çalışır, önemlidir!

print(f"Eğitim seti boyutu arkadaşlar: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}") # Eğitim setinin boyutlarnı yazdıralım
print(f"Test seti boyutu arkadaşlar: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}") # Test setinin boyutlarını da görelim
```

## 3. Modellerin Tanımlanması, Eğitilmesi ve Değerlendirilmesi

Arkadaşlar, şimdi belirlediğimiz 5 farklı sınıflandırma algoritmasını kullanarak modellerimizi eğiteceğiz ve test verisi üzerinde performanslarını ölçeceğiz.
Her model için doğruluk (accuracy) skorunu ve detaylı sınıflandırma raporunu (precision, recall, f1-score) göreceğiz.

```python
# Kullanacağımız modelleri bir sözlük yapısında tanımlayalım
models = {
    "Lojistik Regresyon": LogisticRegression(max_iter=200), # max_iter, yakınsama için maksimum iterasyon sayısı
    "K-En Yakın Komşu (KNN)": KNeighborsClassifier(n_neighbors=5), # n_neighbors, komşu sayısıdır arkadaşlar
    "Destek Vektör Makinesi (SVM)": SVC(probability=True), # probability=True, olasılık tahmini için, permütasyon öneminde gerekebilir
    "Karar Ağacı": DecisionTreeClassifier(random_state=42), # random_state, ağacın dallanma kararlarında rastgeleliği kontrol eder
    "Rastgele Orman": RandomForestClassifier(n_estimators=100, random_state=42) # n_estimators, ormandaki ağaç sayısıdır
}

# Her bir modeli eğitelim ve değerlendirelim
for model_name, model in models.items(): # Sözlükteki her bir model için döngü başlatalım arkadaşlar
    print(f"--- {model_name} ---") # Hangi model üzerinde çalıştığımızı belirtelim

    # Modeli eğitim verisiyle eğitelim (fit edelim)
    model.fit(X_train, y_train) # fit metodu, modelin öğrenme işlemini yapar

    # Test verisi üzerinde tahmin yapalım
    y_pred = model.predict(X_test) # predict metodu, eğitilmiş modelle tahmin üretir

    # Performans metriklerini hesaplayalım ve yazdıralım
    accuracy = accuracy_score(y_test, y_pred) # Gerçek değerler (y_test) ile tahminleri (y_pred) karşılaştırıp doğruluğu bulur
    print(f"Doğruluk (Accuracy): {accuracy:.4f}") # Doğruluğu 4 ondalık basamakla gösterelim
    print("Sınıflandırma Raporu:") # Detaylı rapor için başlık
    print(classification_report(y_test, y_pred, target_names=iris.target_names)) # Raporu yazdıralım, target_names ile sınıf isimlerini de ekleyelim
    print("-" * 30) # Modeller arası ayırıcı bir çizgi ekleyelim
```

## 4. Öznitelik Katkısının Gösterilmesi

Arkadaşlar, modellerimizin hangi öznitelikleri daha önemli bulduğunu anlamak, model yorumlanabilirliği açısından çok değerlidir.
Burada iki farklı yöntem kullanacağız.

### 4.1. Yöntem 1: Ağaç Tabanlı Modellerin Öznitelik Önem Düzeyleri (`feature_importances_`)

Karar Ağaçları ve Rastgele Ormanlar gibi ağaç tabanlı modeller, eğitim sırasında özniteliklerin ne kadar "saf" düğümler oluşturduğuna bakarak bir önem skoru hesaplarlar.
Bu skorlara `feature_importances_` özelliği ile erişebiliriz. Genellikle Rastgele Orman'ın verdiği önem skorları daha stabildir.

```python
# Rastgele Orman modelimizin öznitelik önemlerini alalım
# Eğer diğer modeller için de varsa (mesela Karar Ağacı), onun için de yapabilirsiniz
rf_model = models["Rastgele Orman"] # Daha önce eğittiğimiz Rastgele Orman modelini alalım
importances_rf = rf_model.feature_importances_ # Modelin 'feature_importances_' özniteliğini çekelim

# Önem düzeylerini öznitelik isimleriyle birlikte bir Pandas Serisi olarak saklayalım
feature_importances_rf_series = pd.Series(importances_rf, index=feature_names) # Daha okunaklı olması için Pandas Serisi yapalım
feature_importances_rf_series = feature_importances_rf_series.sort_values(ascending=False) # Önem sırasına göre sıralayalım

print("\n--- Rastgele Orman: Öznitelik Önem Düzeyleri (feature_importances_) ---") # Başlığımızı atalım arkadaşlar
print(feature_importances_rf_series) # Önem düzeylerini yazdıralım

# Görselleştirelim
plt.figure(figsize=(10, 6)) # Grafik boyutunu ayarlayalım
sns.barplot(x=feature_importances_rf_series.values, y=feature_importances_rf_series.index, palette="viridis") # Seaborn ile çubuk grafik çizelim
plt.title('Rastgele Orman: Öznitelik Önem Düzeyleri', fontsize=15) # Grafiğe başlık ekleyelim
plt.xlabel('Önem Skoru', fontsize=12) # X ekseni etiketi
plt.ylabel('Öznitelikler', fontsize=12) # Y ekseni etiketi
plt.tight_layout() # Grafiğin düzgün sığmasını sağlar
plt.show() # Grafiği gösterelim arkadaşlarım
```
Arkadaşlar, yukarıdaki grafikte hangi özniteliğin Rastgele Orman modeli için ne kadar önemli olduğunu görebiliyoruz. Genelde petal length ve petal width daha ayırt edici oluyor IRIS verisetnde.

### 4.2. Yöntem 2: Permütasyon Önem Düzeyi (Permutation Importance)

Permütasyon önemi, model-agnostik bir tekniktir, yani hemen hemen her modelle kullanılabilir.
Bir özniteliğin değerlerini rastgele karıştırarak (permütasyon uygulayarak) modelin performansındaki düşüşü ölçer. Performans ne kadar çok düşerse, o öznitelik o kadar önemlidir arkadaşlar.
Bu işlemi genelde test seti üzerinde yaparız.

```python
# Rastgele Orman modeli için permütasyon önemini hesaplayalım
# Diğer modeller için de deneyebilirsiniz (örneğin SVM)
# n_repeats, her öznitelik için permütasyonun kaç kez tekrarlanacağını belirtir, ortalaması alınır
# random_state, tekrarlanabilirlik için
print("\n--- Rastgele Orman: Permütasyon Önem Düzeyi (Test Seti Üzerinde) ---") # Başlığımızı atalım

# permutation_importance fonksiyonunu kullanalım
# rf_model: Hangi modelin kullanılacağı
# X_test, y_test: Hangi veri üzerinde hesaplanacağı
# n_repeats: Kaç defa permütasyon yapılacağı
# random_state: Sonuçların tekrarlanabilirliği için
perm_importance_rf = permutation_importance(rf_model, X_test, y_test, n_repeats=30, random_state=42)

# Sonuçları daha okunaklı hale getirelim
perm_importances_rf_mean = perm_importance_rf.importances_mean # Ortalama önem skorlarını alalım
perm_importances_rf_std = perm_importance_rf.importances_std # Standart sapmalarını da alalım (isteğe bağlı)

# Önem düzeylerini öznitelik isimleriyle birlikte bir Pandas Serisi olarak saklayalım
perm_feature_importances_rf_series = pd.Series(perm_importances_rf_mean, index=feature_names) # Pandas Serisi oluşturalım
perm_feature_importances_rf_series = perm_feature_importances_rf_series.sort_values(ascending=False) # Önem sırasına göre sıralayalım

print(perm_feature_importances_rf_series) # Permütasyon önem skorlarını yazdıralım

# Görselleştirelim
plt.figure(figsize=(10, 6)) # Grafik boyutunu ayarlayalım
sns.barplot(x=perm_feature_importances_rf_series.values, y=perm_feature_importances_rf_series.index, palette="mako") # Seaborn ile çubuk grafik çizelim
plt.title('Rastgele Orman: Permütasyon Önem Düzeyleri (Test Seti)', fontsize=15) # Grafiğe başlık
plt.xlabel('Önem Skoru (Doğruluktaki Düşüş)', fontsize=12) # X ekseni etiketi
plt.ylabel('Öznitelikler', fontsize=12) # Y ekseni etiketi
plt.tight_layout() # Grafiğin düzgün sığmasını sağlar
plt.show() # Grafiği gösterelim
```
Arkadaşlar, permütasyon önemi bize bir özniteliği karıştırdığımızda modelin doğruluğunun ne kadar düştüğünü gösterir. Bu da özniteliğin ne kadar kritik olduğunu anlamamıza yardımcı olur. Genellikle `feature_importances_` ile benzer sonuçlar verse de, farklılıklar olabilir ve bu da bize ek bilgi sunar.

## 5. Sonuç

Arkadaşlar, bu çalışmada IRIS verisetini kullanarak 5 farklı sınıflandırma algoritmasının performansını karşılaştırdık.
Lojistik Regresyon, KNN, SVM, Karar Ağacı ve Rastgele Orman modellerini eğittik ve doğruluk skorlarını, sınıflandırma raporlarını inceledik.
Ayrıca, Rastgele Orman modelini örnek alarak, ağaç tabanlı modellerin sunduğu `feature_importances_` ve model-agnostik bir yöntem olan permütasyon önemini kullanarak özniteliklerin modele katkısını analiz ettik.


## Python Kodunun Tamamı

Aşağıda tüm adımları içeren Python kodunu toplu halde bulabilirsiniz :

```python
# Veri işleme ve sayısal operasyonlar için temel kütüphaneler 
import numpy as np
import pandas as pd

# Veri setlerini yüklemek ve bölmek için sklearn modülleri
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Sınıflandırma algoritmalarımız
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Model performansını değerlendirmek için metrikler
from sklearn.metrics import accuracy_score, classification_report

# Öznitelik katkısını göstermek için
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Uyarıları bastırmak için (isteğe bağlı, daha temiz çıktı için)
import warnings
warnings.filterwarnings('ignore')

# --- 1. Veri Setinin Yüklenmesi ve Hazırlanması ---
# IRIS verisetini yükleyelim arkadaşlar
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Veri setini eğitim ve test olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Eğitim seti boyutu arkadaşlar: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"Test seti boyutu arkadaşlar: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("-" * 30)

# --- 2. Modellerin Tanımlanması, Eğitilmesi ve Değerlendirilmesi ---
models = {
    "Lojistik Regresyon": LogisticRegression(max_iter=200),
    "K-En Yakın Komşu (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Destek Vektör Makinesi (SVM)": SVC(probability=True, random_state=42), # random_state ekledim SVM'e de
    "Karar Ağacı": DecisionTreeClassifier(random_state=42),
    "Rastgele Orman": RandomForestClassifier(n_estimators=100, random_state=42)
}

for model_name, model in models.items():
    print(f"--- {model_name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Doğruluk (Accuracy): {accuracy:.4f}")
    print("Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("-" * 30)

# --- 3. Öznitelik Katkısının Gösterilmesi ---

# 3.1. Yöntem 1: Ağaç Tabanlı Modellerin Öznitelik Önem Düzeyleri (feature_importances_)
rf_model = models["Rastgele Orman"]
importances_rf = rf_model.feature_importances_
feature_importances_rf_series = pd.Series(importances_rf, index=feature_names)
feature_importances_rf_series = feature_importances_rf_series.sort_values(ascending=False)

print("\n--- Rastgele Orman: Öznitelik Önem Düzeyleri (feature_importances_) ---")
print(feature_importances_rf_series)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_rf_series.values, y=feature_importances_rf_series.index, palette="viridis")
plt.title('Rastgele Orman: Öznitelik Önem Düzeyleri', fontsize=15)
plt.xlabel('Önem Skoru', fontsize=12)
plt.ylabel('Öznitelikler', fontsize=12)
plt.tight_layout()
plt.show()

# 3.2. Yöntem 2: Permütasyon Önem Düzeyi (Permutation Importance)
print("\n--- Rastgele Orman: Permütasyon Önem Düzeyi (Test Seti Üzerinde) ---")
perm_importance_rf = permutation_importance(rf_model, X_test, y_test, n_repeats=30, random_state=42)
perm_importances_rf_mean = perm_importance_rf.importances_mean
perm_feature_importances_rf_series = pd.Series(perm_importances_rf_mean, index=feature_names)
perm_feature_importances_rf_series = perm_feature_importances_rf_series.sort_values(ascending=False)

print(perm_feature_importances_rf_series)

plt.figure(figsize=(10, 6))
sns.barplot(x=perm_feature_importances_rf_series.values, y=perm_feature_importances_rf_series.index, palette="mako")
plt.title('Rastgele Orman: Permütasyon Önem Düzeyleri (Test Seti)', fontsize=15)
plt.xlabel('Önem Skoru (Doğruluktaki Düşüş)', fontsize=12)
plt.ylabel('Öznitelikler', fontsize=12)
plt.tight_layout()
plt.show()

print("\nArkadaşlar, analizimiz tamamlandı! Umarım her şey açıktır.")
```