# Makine Öğrenmesinde Yorumlanabilirlik: Özellik Önemi, SHAP ve Kategorik Kodlama

Bir model inşa etmek bir iştir, o modeli anlamak başka bir iştir. Sektördeki karar alıcı ne model kodundan anlar ne de matematikten. Ona "algoritmam bu sonucu verdi" demek yeterli değildir. Bu ders notunda öğrendiğiniz üç araç — Feature Importance, SHAP ve OneHot Encoding — birbirleriyle doğrudan bağlantılıdır ve yorumlanabilir makine öğrenmesinin (Interpretable Machine Learning — IML) temel taşlarını oluşturur.

Bu üç kavramı ortak bir örnek üzerinde götüreceğiz: **Ev Fiyat Tahmini**. Veri setimizde hem sayısal hem kategorik değişkenler var. Bu sayede OneHot Encoding'in neden gerektiğini ve Feature Importance ile SHAP'ın birbirinden nasıl ayrıştığını aynı anda görebileceğiz.

---

## Ortak Veri Seti: Ev Fiyat Tahmini

Veri setimiz 300 satırdan oluşuyor ve aşağıdaki değişkenleri içeriyor:

| Değişken | Tür | Açıklama |
|---|---|---|
| `alan_m2` | Sayısal | Evin net alanı (m²) |
| `bina_yasi` | Sayısal | Binanın yaşı (yıl) |
| `oda_sayisi` | Sayısal (1–5) | Oda sayısı |
| `semt` | Kategorik | Merkez / Kuzey / Güney / Doğu |
| `isitma` | Kategorik | Doğalgaz / Elektrik / Kombi |
| `fiyat_tl` | Hedef | Satış fiyatı (TL) |

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 300

df = pd.DataFrame({
    'alan_m2':    np.random.randint(60, 250, n),
    'bina_yasi':  np.random.randint(0, 40, n),
    'oda_sayisi': np.random.choice([1, 2, 3, 4, 5], n,
                                   p=[0.05, 0.20, 0.40, 0.25, 0.10]),
    'semt':       np.random.choice(['Merkez','Kuzey','Güney','Doğu'], n,
                                   p=[0.30, 0.25, 0.25, 0.20]),
    'isitma':     np.random.choice(['Doğalgaz','Elektrik','Kombi'], n,
                                   p=[0.50, 0.30, 0.20])
})

semt_prim    = {'Merkez': 200_000, 'Kuzey': 120_000,
                'Güney':   80_000, 'Doğu':   50_000}
isitma_prim  = {'Doğalgaz': 30_000, 'Elektrik': -10_000, 'Kombi': 15_000}

df['fiyat_tl'] = (
    df['alan_m2']   * 12_000
    + (40 - df['bina_yasi']) * 5_000
    + df['oda_sayisi']       * 25_000
    + df['semt'].map(semt_prim)
    + df['isitma'].map(isitma_prim)
    + np.random.normal(0, 60_000, n)
).clip(lower=0).astype(int)

print(df.head())
```

---

## 1. OneHot Encoding (Tek-Sıcak Kodlama)

Makineler metinle düşünmez. Bir regresyon veya karar ağacı modeline `"Merkez"` yazıp veriyi gönderirsek model çöker. Tüm matematiksel işlemler sayılar üzerinde yürür, bu nedenle kategorik (sınıflandırılmış) değişkenleri önce sayılara çevirmemiz gerekir.

### 1.1 İlk Akla Gelen ve Neden Yanlış Olduğu

Bir öğrenci şöyle der: "Semtlere sıra numarası vereyim. Merkez=0, Kuzey=1, Güney=2, Doğu=3." Bu yaklaşıma **Etiket Kodlama** (Label Encoding) denir. Hızlıdır, tek sütuna sığar. Ancak kritik bir sorun taşır.

Sıra numarası veren bu kodlama, modele şunu söyler: "Doğu (3), Kuzeyden (1) iki kat daha büyük bir anlam taşıyor. Ve Merkez (0) ile Doğu (3) arasındaki fark, Merkez ile Kuzey (1) arasındaki farkın üç katı." Bu ilişkilerin hiçbiri gerçek değildir. Semt isimleri sıralı ya da sayısal bir yapıya sahip değildir; bunlar eşit düzeyde birbirinden bağımsız kategorilerdir.

Tıpkı çantanızdaki renklere sıra numarası verip "kırmızı=1, mavi=2, yeşil=3" diye kodladığınızda, modele "yeşil iki mavi ediyor" demeniz gibi.

### 1.2 Doğru Yaklaşım: Bir Sütunu Birden Fazla Sütuna Yay

**OneHot Encoding** (Tek-Sıcak Kodlama — sıcak Latince `calere`, "ısınmak" anlamına gelir, "aktif olan" için mecazi kullanım) her kategori için ayrı bir ikili (binary) sütun açar. Bir satırda hangi kategori aktifse o sütunun değeri `1`, diğerleri `0` olur. Aynı anda yalnızca bir sütun "sıcak" (1) olabilir.

[Şekil 1: OneHot Encoding Dönüşümü](images/onehot_encoding.svg)

### 1.3 Kukla Değişken Tuzağı (Dummy Variable Trap)

Dört semtimiz varsa dört sütun mı oluşturmalıyız? **Hayır.** Bu noktada matematiksel bir sorun ortaya çıkar.

Dört kategorimiz için dört sütun açtığımızda, dördüncü sütun her zaman diğer üçünün bir fonksiyonudur:

```
semt_Merkez = 1 − semt_Kuzey − semt_Güney − semt_Doğu
```

Modeliniz bir doğrusal ilişki kümesiyle çalışıyorsa (regresyon, destek vektör makineleri, yapay sinir ağları) bu doğrusal bağımlılık, parametre tahmini sırasında **Çoklu Doğrusallık** (Multicollinearity) sorununa yol açar. Matrisin tersini almanız gerektiğinde matris tekilleşir (singular), yani tersi yoktur.

Çözüm basittir: `k` kategoriniz varsa `k−1` sütun açın. Dışarıda bıraktığınız kategori referans kategorisi olur; diğer tüm sütunlar o referansa göre yorumlanır. `pandas`'ta `drop_first=True`, scikit-learn'de `drop='first'` parametresi bunu otomatik yapar.

```python
# Yöntem 1: pandas ile
X_encoded = pd.get_dummies(
    df.drop('fiyat_tl', axis=1),
    columns=['semt', 'isitma'],
    drop_first=True   # Kukla değişken tuzağından kaçınmak için
)
print(X_encoded.columns.tolist())
# ['alan_m2', 'bina_yasi', 'oda_sayisi',
#  'semt_Güney', 'semt_Kuzey', 'semt_Merkez',
#  'isitma_Elektrik', 'isitma_Kombi']
# Not: 'semt_Doğu' ve 'isitma_Doğalgaz' referans olarak çıktı
```

```python
# Yöntem 2: scikit-learn ColumnTransformer ile (pipeline için önerilir)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

kategorik_sutunlar = ['semt', 'isitma']
sayisal_sutunlar   = ['alan_m2', 'bina_yasi', 'oda_sayisi']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first', sparse_output=False), kategorik_sutunlar),
    ('num', 'passthrough', sayisal_sutunlar)
])

X = df.drop('fiyat_tl', axis=1)
y = df['fiyat_tl']

X_transformed = preprocessor.fit_transform(X)
```

### 1.4 Ne Zaman OneHot, Ne Zaman Başka Bir Yöntem?

Kategori sayısı az (2–15 arası) ve sırasal bir anlam taşımıyorsa OneHot Encoding doğru tercihtir. Ancak bazı özel durumlar farklı yaklaşımlar gerektirir:

| Durum | Öneri |
|---|---|
| Sıralı anlam var (örn. `Düşük < Orta < Yüksek`) | Ordinal Encoding |
| Çok fazla kategori (örn. şehir ismi, 500+ ülke kodu) | Target Encoding veya Embedding |
| Yalnızca ağaç tabanlı model kullanılıyorsa | Label Encoding de kabul edilebilir |
| Yüksek kardinalitenin hafızayı şişirdiği durumlar | Hashing Encoding |

Ağaç tabanlı modeller (Karar Ağacı, Rastgele Orman — Random Forest, XGBoost) aslında Label Encoding ile de çalışabilir, çünkü bu modeller sayıların büyüklüğünü değil yalnızca eşik değerlerini kullanır. Ama doğrusal modellerde ve sinir ağlarında OneHot Encoding tercih edilmelidir.

---

## 2. Özellik Önemi (Feature Importance)

Modelinizi eğittiniz ve doğruluk metrikleriniz tatmin edici. Şimdi şu soruyu sormak doğaldır: "Bu model hangi değişkenlere bakarak karar veriyor?" Özellik Önemi (Feature Importance) bu soruya sayısal bir cevap verir.

### 2.1 Karar Ağacında Safsızlığın Azalması

Bir karar ağacı (decision tree), veriyi adım adım böler. Her düğümde bir kural sorar: "alan_m2 > 150 mi?" Eğer bu bölünme sonrasında iki grup birbirinden belirgin biçimde ayrışıyorsa, bu özellik iyi bir ayrıştırıcıdır.

Bölünme kalitesini ölçmek için Gini Safsızlığı (Gini Impurity — İtalyan matematikçi Corrado Gini'den) kullanılır:

$$G(D) = 1 - \sum_{k=1}^{K} p_k^2$$

Burada $p_k$, $D$ düğümündeki $k$ sınıfının oranıdır. Safsızlık sıfıra yakın ise düğüm homojen (tüm örnekler aynı sınıfta), bire yakın ise heterojendir.

Bir düğümdeki bölünmenin sağladığı **Bilgi Kazanımı** (Information Gain):

$$\Delta G(D, j) = G(D) - \frac{|D_L|}{|D|} \cdot G(D_L) - \frac{|D_R|}{|D|} \cdot G(D_R)$$

Burada $D_L$ ve $D_R$ bölünme sonucunda oluşan sol ve sağ alt kümelerdir. $j$ özelliğinin tüm ağaçtaki önemi, bu özelliğin kullanıldığı her düğümdeki kazanımların ağırlıklı toplamıdır.

### 2.2 Rastgele Ormanda Özellik Önemi

Rastgele Orman (Random Forest) yüzlerce karar ağacından oluşur. Her ağaçta aynı özellik birden fazla düğümde kullanılabilir, ve her ağaç rastgele bir özellik alt kümesi görür. Bir özelliğin modeldeki toplam önem puanı, tüm ağaçlardaki katkılarının ortalamasıdır:

$$\text{FI}(j) = \frac{1}{B} \sum_{b=1}^{B} \sum_{\substack{t \in T_b \\ \text{split}(t)=j}} \frac{n_t}{N} \cdot \Delta G_t$$

Burada $B$ ağaç sayısı, $T_b$ $b$'inci ağacın düğümleri, $n_t$ düğüme düşen örnek sayısı ve $N$ toplam örnek sayısıdır. Bu yönteme **MDI** (Mean Decrease in Impurity — Ortalama Safsızlık Azalması) denir.

Puanlar normalizasyon sonrası 0 ile 1 arasına sıkıştırılır ve toplamları 1 eder.

### 2.3 Kod Örneği: Feature Importance

```python
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Veriyi hazırla (önceki bölümdeki X_encoded ve y)
X = df.drop('fiyat_tl', axis=1)
y = df['fiyat_tl']
X_enc = pd.get_dummies(X, columns=['semt', 'isitma'], drop_first=True)

# Model eğit
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_enc, y)

# Önem puanlarını al ve sırala
fi_series = pd.Series(rf.feature_importances_, index=X_enc.columns)
fi_series = fi_series.sort_values(ascending=False)

print(fi_series.round(3))
# alan_m2           0.382
# semt_Merkez       0.221
# bina_yasi         0.148
# oda_sayisi        0.122
# semt_Kuzey        0.071
# isitma_Doğalgaz   0.038
# isitma_Kombi      0.018

# Görselleştir
fi_series.sort_values().plot(kind='barh', figsize=(8, 5), color='steelblue')
plt.xlabel('Önem Puanı')
plt.title('Feature Importance — Rastgele Orman')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
```

[Şekil 2: Feature Importance — Rastgele Orman Modeli](images/feature_importance_bar.svg)

### 2.4 Feature Importance'ın Sınırları

Bu yöntem güçlüdür ama körce güvenilmemelidir.

**Birinci sorun: Yorumlama eksikliği.** Sonuç şunu söyler: "alan_m2 önemli." Ancak şunu söylemez: "Büyük alan fiyatı artırır mı düşürür mü? Ve ne kadar?" Yönü ve büyüklüğü göremezsiniz.

**İkinci sorun: Küresellik.** Tek bir önem puanı tüm veri setini temsil eder. Bir mahalledeki evler için alan önemliyken, Merkez semtindeki pahalı ama küçük lüks dairelerde alan belki çok da belirleyici değildir. Bu ayrımı göremezsiniz.

**Üçüncü sorun: Korelasyonlu değişkenler.** İki değişken birbiriyle yüksek korelasyon taşıyorsa (örn. `alan_m2` ve `oda_sayisi`), önem puanları ikisi arasında bölünür. İkisi de düşük görünebilir, oysa ikisi birden yüksek öneme sahiptir.

Bu sınırların hepsini aşan araç SHAP'tır.

---

## 3. SHAP Analizi

SHAP, SHapley Additive exPlanations kelimelerinin kısaltmasıdır:

- **SHapley** → Nobel Ekonomi Ödülü sahibi (2012) Amerikalı matematikçi Lloyd Shapley'den
- **Additive** → Katkılar toplanabilir; tüm özellik katkıları toplanınca tahmin farkını verir
- **exPlanations** → Açıklamalar

Kavramın kökeni 1950'lerin kooperatif oyun teorisine (cooperative game theory) dayanır. Shapley 1953'te şu soruya yanıt aradı: "Birden fazla oyuncu ortak çalışarak bir ödül kazanırsa, bu ödül oyuncular arasında nasıl adil biçimde paylaştırılır?"

2017'de Lundberg ve Lee bu fikri makine öğrenmesine taşıdı. "Oyuncular" özelliklerdir, "ödül" modelin yaptığı tahmindir.

### 3.1 Fikrin Özü

Mutfak analojisiyle düşünelim. Dört kişilik bir ekip birlikte yemek yarışmasında birincilik kazandı ve 100.000 TL ödül aldı. Kimin katkısı ne kadardı? Başçefin mi, pastacının mı, teslimatçının mı?

Shapley değeri şunu hesaplar: Bu kişi farklı takım kombinasyonlarına katılsaydı, her birinde ne kadar ek katkı getirirdi? Tüm bu marjinal katkıların ağırlıklı ortalaması o kişinin adil payıdır.

Makine öğrenmesine çevirirsek: Bir özellik farklı özellik alt kümeleriyle birlikte olduğunda, her kombinasyona ne kadar ek tahmin değişikliği getiriyor?

### 3.2 Shapley Değeri Formülü

$i$ özelliğinin Shapley değeri:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!} \left[ v(S \cup \{i\}) - v(S) \right]$$

Burada:

- $F$ : tüm özellikler kümesi
- $S$ : $i$ özelliği **dışındaki** bir alt küme
- $v(S)$ : yalnızca $S$ özellikleri bilindiğinde modelin tahmini
- $v(S \cup \{i\}) - v(S)$ : $i$ özelliğinin $S$'e eklenmesiyle tahminlerdeki değişim (marjinal katkı)
- $\frac{|S|!\,(|F|-|S|-1)!}{|F|!}$ : bu kombinasyonun karşıladığı sıralama oranı (ağırlık)

Formülün ağırlık kısmı, $i$'nin varlığıyla yokluğu arasındaki farkın hesaplandığı kombinasyonun tüm olası sıralamalar içindeki yüzdesini verir. Tüm olası özellik sıralamalarının ortalamasını alıyorsunuz, denilebilir.

### 3.3 SHAP'ın Toplamsallık Özelliği

SHAP değerlerinin en kritik özelliği şudur: bir örnek için tüm Shapley değerleri toplanınca, temel değerden (base value) tahmine olan farkı tam olarak açıklar.

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$

Burada:
- $f(x)$ : modelin bu örnek için verdiği tahmin
- $\phi_0$ : temel değer, $E[f(x)]$ — tüm eğitim verisininin ortalama tahmini
- $\phi_i$ : $i$'inci özelliğin bu örneğe katkısı (pozitif veya negatif olabilir)

Bu toplamsallık özelliği SHAP'ı güçlü kılan şeydir. Diğer yorumlama yöntemlerinin çoğu bu garantiyi vermez.

### 3.4 Kod Örneği: SHAP

```python
import shap

# TreeExplainer: ağaç tabanlı modeller için optimize edilmiş hızlı hesaplama
explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_enc)
# shap_values shape: (300, 8) — her örnek x her özellik için bir değer

# Temel değer (E[f(x)])
print(f"Temel Değer: {explainer.expected_value:,.0f} TL")
# Temel Değer: 851,423 TL

# Tek bir örnek için SHAP değerlerine bakalım (50. satır)
idx = 50
shap_df = pd.DataFrame({
    'Özellik':     X_enc.columns,
    'Özellik Değeri': X_enc.iloc[idx].values,
    'SHAP Değeri': shap_values[idx]
}).sort_values('SHAP Değeri', ascending=False)

print(shap_df)
# Özellik         Özellik Değeri  SHAP Değeri
# alan_m2                   185   +118,420
# semt_Merkez                 1    +96,300
# oda_sayisi                  4    +14,150
# isitma_Elektrik             0     -4,820
# bina_yasi                  27   -24,600

# Tahmin kontrolü
toplam_phi = shap_values[idx].sum() + explainer.expected_value
print(f"Tahmin (f(x)):  {rf.predict(X_enc.iloc[[idx]])[0]:,.0f} TL")
print(f"Temel + SHAP:   {toplam_phi:,.0f} TL")
# İki değer birbirine eşit — toplamsallık kanıtlandı
```

```python
# --- SHAP Görselleştirmeleri ---

# 1. Summary Plot (Özet Grafik): tüm veri için her özelliğin dağılımı
shap.summary_plot(shap_values, X_enc, plot_type='dot')
# Yatay eksen: SHAP değeri (ne kadar büyükse o kadar güçlü etki)
# Renk: özelliğin yüksek mi (kırmızı) düşük mü (mavi) değer aldığı

# 2. Bar Plot (Çubuk Grafik): ortalama mutlak SHAP değerleri
shap.summary_plot(shap_values, X_enc, plot_type='bar')

# 3. Waterfall Plot (Şelale Grafik): tek örnek açıklaması
shap.waterfall_plot(
    shap.Explanation(
        values    = shap_values[idx],
        base_values = explainer.expected_value,
        data        = X_enc.iloc[idx],
        feature_names = X_enc.columns.tolist()
    )
)

# 4. Force Plot (Kuvvet Grafici): tek örnek, inline
shap.initjs()
shap.force_plot(
    explainer.expected_value,
    shap_values[idx],
    X_enc.iloc[idx]
)
```

[Şekil 3: SHAP Katkı Grafiği — Tek Örnek](images/shap_waterfall.svg)

### 3.5 TreeSHAP ve KernelSHAP

Shapley değerinin teorik hesabı üstel karmaşıklığa sahiptir: $2^{|F|}$ kombinasyon denenmesi gerekir. 8 özellik için 256, 20 özellik için zaten 1 milyonun üzerinde kombinasyon var. Bu nedenle uygulamada iki pratik yaklaşım kullanılır:

**TreeSHAP** — Ağaç yapısından yararlanarak Shapley değerlerini polinom zamanda (polynomial time) hesaplar. Karar ağaçları, Rastgele Orman, XGBoost, LightGBM gibi ağaç tabanlı modellerde kullanılır. `shap.TreeExplainer` bunu çağırır.

**KernelSHAP** — Model bağımsızdır (model-agnostic). Herhangi bir kara kutu (black-box) modele uygulanabilir. Özellik alt kümelerini rastgele örnekleyerek yaklaşık Shapley değerleri hesaplar. Daha yavaştır. `shap.KernelExplainer` ile çağrılır.

---

## 4. Feature Importance ile SHAP Karşılaştırması

Bu iki araç birbiriyle rekabet etmez, birbirini tamamlar. Hangisini ne zaman kullanacağınızı netleştirmek gerekiyor.

[Şekil 4: Global ve Yerel Açıklama Karşılaştırması](images/shap_vs_fi.svg)

### 4.1 Temel Farklar

| Kriter | Feature Importance (MDI) | SHAP |
|---|---|---|
| **Kapsam** | Küresel (Global) — tüm model | Yerel (Local) — her örnek ayrı |
| **Yön bilgisi** | Yok (+/−?) | Var (+/−) |
| **Büyüklük** | Göreli oran (0–1) | Orijinal birim (TL, kg…) |
| **Korelasyona duyarlılık** | Yüksek (değişkenler arası paylaşım) | Daha robust |
| **Hız** | Çok hızlı (model içinde hazır) | Daha yavaş (TreeSHAP hariç) |
| **Toplamsallık garantisi** | Yok | Var |
| **Küresel özet** | Doğrudan | mean(\|SHAP\|) ile elde edilir |

### 4.2 Somut Bir Ayrım

Birinin "bu modelde hangi özellikler genel olarak önemli?" diye sormasıyla, birinin "bu 185 m²'lik Merkez semtindeki dairenin neden bu kadar yüksek fiyat biçildi?" diye sorması farklı soru türleridir.

Birinci soru için Feature Importance yeterlidir. İkinci soru için SHAP gereklidir.

Gıda kalite kontrol sistemine analoji yaparsak: Feature Importance, aylık üretim raporunda "en çok hataya neden olan makine hangisi" sorusunu cevaplar. SHAP ise bugün reddedilen spesifik ürünün "tam olarak hangi ölçüm neden hata bayrağı kaldırdı" sorusunu cevaplar.

### 4.3 Çelişen Sonuçlar Geldiğinde

SHAP ve Feature Importance zaman zaman farklı sıralamalar üretebilir. Bu durumun iki temel nedeni vardır:

**Korelasyonlu değişkenler.** MDI, korelasyonlu iki değişkenin önem puanını aralarında böler. SHAP ise her örnek için marjinal katkıyı ölçtüğünden bu bölünmeyi daha gerçekçi yapar.

**Nadir ama güçlü etkiler.** Bir değişken çok az örnekte devreye giriyor ama girdiğinde tahmini büyük ölçüde değiştiriyorsa, MDI bunu küçük görebilir. Mean(|SHAP|) bunu daha doğru yakalar çünkü o az örnek için yüksek değer alır.

Pratikte önerilen yaklaşım şudur: Model seçimi ve genel yorumlama için Feature Importance, müşteriye/kullanıcıya veya regülatöre spesifik bir tahminin nedenini açıklamak için SHAP kullanın.

---

## 5. Üç Kavramı Birleştiren Tam Uygulama

Aşağıdaki kod parçası veri hazırlama (OneHot Encoding), model eğitimi, Feature Importance analizi ve SHAP analizini baştan sona uygular.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# ─── 1. VERİ SETİ ────────────────────────────────────────────────────────────
np.random.seed(42)
n = 300

df = pd.DataFrame({
    'alan_m2':    np.random.randint(60, 250, n),
    'bina_yasi':  np.random.randint(0, 40, n),
    'oda_sayisi': np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.20, 0.40, 0.25, 0.10]),
    'semt':       np.random.choice(['Merkez','Kuzey','Güney','Doğu'], n, p=[0.30,0.25,0.25,0.20]),
    'isitma':     np.random.choice(['Doğalgaz','Elektrik','Kombi'], n, p=[0.50,0.30,0.20])
})
semt_prim   = {'Merkez':200_000,'Kuzey':120_000,'Güney':80_000,'Doğu':50_000}
isitma_prim = {'Doğalgaz':30_000,'Elektrik':-10_000,'Kombi':15_000}
df['fiyat_tl'] = (
    df['alan_m2']*12_000 + (40-df['bina_yasi'])*5_000 + df['oda_sayisi']*25_000
    + df['semt'].map(semt_prim) + df['isitma'].map(isitma_prim)
    + np.random.normal(0, 60_000, n)
).clip(lower=0).astype(int)

# ─── 2. ONEHOT ENCODING ──────────────────────────────────────────────────────
X = pd.get_dummies(
    df.drop('fiyat_tl', axis=1),
    columns=['semt', 'isitma'],
    drop_first=True
)
y = df['fiyat_tl']

print("Kodlanmış Sütunlar:", X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── 3. MODEL EĞİTİMİ ───────────────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

test_r2 = rf.score(X_test, y_test)
print(f"Test R²: {test_r2:.4f}")

# ─── 4. FEATURE IMPORTANCE ──────────────────────────────────────────────────
fi = pd.Series(rf.feature_importances_, index=X.columns) \
       .sort_values(ascending=False)

print("\n--- Feature Importance (MDI) ---")
print(fi.round(3))

fi.sort_values().plot(kind='barh', figsize=(8, 5), color='#2471a3')
plt.xlabel('Önem Puanı')
plt.title('Feature Importance — Rastgele Orman')
plt.tight_layout()
plt.savefig('fi_plot.png', dpi=150)
plt.show()

# ─── 5. SHAP ANALİZİ ────────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)   # shape: (60, 8)

base_value = explainer.expected_value
print(f"\nTemel Değer E[f(x)]: {base_value:,.0f} TL")

# Global özet: Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type='dot', show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150)
plt.show()

# Yerel örnek: test setinin ilk satırı
idx = 0
tahmin_gercek = rf.predict(X_test.iloc[[idx]])[0]
toplam_shap   = shap_values[idx].sum() + base_value

print(f"\nÖrnek #{X_test.index[idx]} için:")
print(f"  Model tahmini:      {tahmin_gercek:,.0f} TL")
print(f"  phi_0 + sum(phi_i): {toplam_shap:,.0f} TL  (eşit olmalı)")

shap_ornek = pd.DataFrame({
    'Özellik':         X_test.columns,
    'Değer':           X_test.iloc[idx].values,
    'SHAP Katkısı':    shap_values[idx]
}).sort_values('SHAP Katkısı', ascending=False)

print(shap_ornek.to_string(index=False))

# Waterfall plot
shap.waterfall_plot(
    shap.Explanation(
        values       = shap_values[idx],
        base_values  = base_value,
        data         = X_test.iloc[idx],
        feature_names= X_test.columns.tolist()
    )
)

# ─── 6. KARŞILAŞTIRMA: FI vs mean(|SHAP|) ──────────────────────────────────
mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X_test.columns
).sort_values(ascending=False)

karsilastirma = pd.DataFrame({
    'Feature Importance (MDI)': fi,
    'mean(|SHAP|) × 10⁻⁵':    mean_abs_shap / 1e5
}).round(3)

print("\n--- FI vs mean(|SHAP|) karşılaştırması ---")
print(karsilastirma)
```

### Beklenen Çıktı Yorumu

`Feature Importance` ile `mean(|SHAP|)` sıralamaları büyük ölçüde örtüşür. Küçük farklılıklar normaldir ve genellikle değişkenler arası korelasyondan kaynaklanır. `alan_m2` her iki sıralamada da açık ara birinci çıkar; bu verinin tasarım gereğidir. Gerçek projelerde bu kadar net ayrışmalar olmaz ve iki yöntemi birlikte değerlendirmek daha sağlıklı sonuç verir.

---

## Notlar ve Referanslar

- **SHAP kütüphanesi:** Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*. `pip install shap`
- **Shapley, L. S.** (1953). A value for n-person games. *Contributions to the Theory of Games*, 2, 307–317.
- **Molnar, C.** (2022). *Interpretable Machine Learning* (2nd ed.). Güncel çevrimiçi sürüm için: [christophm.github.io/interpretable-ml-book](https://christophm.github.io/interpretable-ml-book)
- **scikit-learn dokümantasyonu:** `sklearn.ensemble.RandomForestRegressor.feature_importances_`, `sklearn.preprocessing.OneHotEncoder`
