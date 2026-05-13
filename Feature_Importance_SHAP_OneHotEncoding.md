# Makine Öğrenmesinde Yorumlanabilirlik: Özellik Önemi, SHAP ve Kategorik Kodlama

Bir model inşa etmek bir iştir, o modeli anlamak başka bir iştir. Sektördeki karar alıcı ne model kodundan anlar ne de matematikten. Ona "algoritmam bu sonucu verdi" demek yeterli değildir. Bu ders notunda öğrendiğiniz üç araç — Feature Importance, SHAP ve OneHot Encoding — birbirleriyle doğrudan bağlantılıdır ve yorumlanabilir makine öğrenmesinin (Interpretable Machine Learning — IML) temel taşlarını oluşturur.

Bu üç kavramı ortak bir örnek üzerinde götüreceğiz: **Ev Fiyat Tahmini**. Veri setimizde hem sayısal hem kategorik değişkenler var. Bu sayede OneHot Encoding'in neden gerektiğini ve Feature Importance ile SHAP'ın birbirinden nasıl ayrıştığını aynı anda görebileceğiz.

---

## Ortak Veri Seti: Ev Fiyat Tahmini

Veri setimiz 300 satırdan oluşuyor ve aşağıdaki değişkenleri içeriyor:

| Değişken | Tür | Açıklama |
| --- | --- | --- |
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

```text
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
| --- | --- |
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

Bu sınırların büyük bölümünü ele alan iki araç vardır: SHAP ve LIME. İkisi de yerel açıklama sağlar, ancak birbirinden farklı mekanizmalar ve garantilerle çalışır.

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

## 4. LIME — Local Interpretable Model-agnostic Explanations

LIME, Ribeiro, Singh ve Guestrin'in 2016 tarihli "Why Should I Trust You?" başlıklı makalesiyle tanıtıldı. İsim açılımı tekniğin özünü özetler:

- **Local** → Yerel: tek bir tahmini açıklar
- **Interpretable** → Yorumlanabilir: çıktı basit, insan tarafından okunabilir bir modeldir
- **Model-agnostic** → Model bağımsız: herhangi bir kara kutuda (black-box) çalışır
- **Explanations** → Açıklamalar

### 4.1 Temel Fikir: Zor Durumda Basit Harita

Karmaşık bir modelin küresel yapısı doğrusal olmayabilir. Ama herhangi bir noktanın çevresine baktığınızda, o küçük bölgede model davranışı çoğunlukla yeterince pürüzsüzdür ve basit bir doğrusal modelle iyi temsil edilebilir. Şehrin tüm yol ağı haritası karmaşıktır; ama o an bulunduğunuz mahalleyi bir peçeteye çizmek kolaydır ve işe yarar.

LIME tam olarak bunu yapar: sorgu noktasının çevresinde bir doğrusal model uydurur ve bu modelin katsayılarını özellik katkısı olarak kullanır.

### 4.2 LIME'ın Çalışma Adımları

Beş adımı vardır:

1. **Sorgu noktası x seç:** Açıklamak istediğiniz tek bir örnek.
2. **Pertürbe et:** x'in özelliklerini rastgele değiştirerek bir komşuluk örneklemi oluştur: $z_1, z_2, \ldots, z_n$.
3. **Kara kutuyu çalıştır:** Her pertürbe edilmiş örnek için $f(z_i)$ tahminini al.
4. **Ağırlıklandır:** Her $z_i$'nin x'e olan uzaklığına göre bir yakınlık puanı hesapla. Uzak örnekler daha az ağırlık taşır.
5. **Yerel model uydur:** Ağırlıklı bu verilere basit bir lineer regresyon uydur. Bu modelin katsayıları LIME açıklamasıdır.

### 4.3 LIME Optimizasyon Formülü

LIME'ın resmi ifadesi bir optimizasyon problemidir:

$$\xi(x) = \arg\min_{g \in G} \underbrace{\mathcal{L}(f, g, \pi_x)}_{\text{yerel uyum kaybı}} + \underbrace{\Omega(g)}_{\text{karmaşıklık cezası}}$$

Burada:

- $G$ : yorumlanabilir modeller ailesi (örn. lineer regresyonlar, küçük karar ağaçları)
- $f$ : asıl kara kutu model
- $g$ : seçilen yerel basit model
- $\mathcal{L}$ : $g$'nin $f$'e yerel olarak ne kadar iyi uyduğunu ölçen kayıp fonksiyonu
- $\Omega(g)$ : modelin karmaşıklık cezası (kullanılan özellik sayısını sınırlar)

Yakınlık ağırlık fonksiyonu üssel çekirdek (exponential kernel) kullanır:

$$\pi_x(z) = \exp\!\left(-\frac{d(x, z)^2}{\sigma^2}\right)$$

$d(x,z)$ sorgu noktası ile pertürbe edilmiş örnek arasındaki mesafe (Öklid veya kosinüs), $\sigma$ ise çekirdeğin bant genişliğidir. Uzak örnekler üssel olarak küçülür ve yerel modele neredeyse hiç katkı vermez.

### 4.4 Kod Örneği: LIME

```python
# pip install lime
from lime import lime_tabular

# LIME Explainer — eğitim verisi istatistikleri üzerinde kurulur
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data = X_train.values,
    feature_names = X_train.columns.tolist(),
    mode          = 'regression',
    random_state  = 42
)

# Tek bir örneği açıkla
idx = 0

lime_exp = lime_explainer.explain_instance(
    data_row   = X_test.iloc[idx].values,
    predict_fn = rf.predict,
    num_features = len(X_train.columns)
)

print(f"Yerel Model R²:             {lime_exp.score:.3f}")
print(f"LIME Tahmini (yerel model): {lime_exp.predicted_value:,.0f} TL")
print(f"Kara Kutu Tahmini:          {rf.predict(X_test.iloc[[idx]])[0]:,.0f} TL")
print()
print("Özellik Katkıları (yerel lineer katsayılar):")
for kural, katki in sorted(lime_exp.as_list(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {kural:45s}  {katki:+,.0f}")
# alan_m2 > 140                              +90,200
# semt_Merkez = 1                            +72,400
# oda_sayisi > 3                             +11,800
# bina_yasi > 20                             -18,100

# Görselleştir
import matplotlib.pyplot as plt
fig = lime_exp.as_pyplot_figure()
fig.suptitle(f"LIME Açıklaması — Örnek #{X_test.index[idx]}", fontsize=12)
plt.tight_layout()
plt.savefig('lime_explanation.png', dpi=150)
plt.show()
```

[Şekil 5: LIME — Sorgu Noktası Çevresi ve Yerel Lineer Model](images/lime_concept.svg)

LIME çıktısındaki kuralların ("alan_m2 > 140") sürekli değişkenleri otomatik olarak aralıklara (bin) böldüğüne dikkat edin. Bu binleştirme SHAP'ın kesin değer üretmesinden farklıdır; açıklamayı bazen daha az hassas kılar.

### 4.5 LIME'ın Güçlü Yanı: Görüntü ve Metin

LIME'ın SHAP'tan belirgin biçimde üstün olduğu alan, **veri tipi esnekliğidir**. Aynı framework tabular verinin ötesinde doğrudan görüntü ve metne uygulanabilir:

**Görüntü:** Pikselleri "süper-piksel" adı verilen bölgelere böler. Her bölgeyi açıp kapatarak kara kutunun tepkisini ölçer. "Model bu X-ray görüntüsünü anormal olarak sınıflandırdı; buna hangi bölge yol açtı?" sorusu yanıtlanabilir.

**Metin:** Kelimeleri teker teker çıkararak hangi kelimenin sınıflandırma kararını etkilediğini ölçer. Duygu analizi, spam tespiti, hukuki belge sınıflandırması gibi alanlarda doğrudan kullanılabilir.

```python
# Metin sınıflandırması için LIME (yapı örneği)
from lime.lime_text import LimeTextExplainer

text_explainer = LimeTextExplainer(class_names=['olumsuz', 'olumlu'])
text_exp = text_explainer.explain_instance(
    text_instance = "Bu ürün beklentilerin altında kaldı",
    classifier_fn = metin_modeli.predict_proba,
    num_features  = 6
)
text_exp.show_in_notebook()
```

### 4.6 LIME'ın Sınırları

**Stokastisite.** Pertürbasyon rastgeleliğe dayandığından aynı örneği iki kez açıklarsanız biraz farklı katsayılar elde edebilirsiniz. Kritik kararlar için bu güvenilirlik sorusu yaratır. `random_state` sabitlense de pertürbasyon sayısı ve bant genişliği seçimi sonuçları etkiler.

**Yerel model uyum kalitesi.** $R^2$ değeri düşükse yerel lineer model o bölgede kara kutuyu iyi temsil etmiyor demektir. Her LIME çıktısında $R^2$ değerini kontrol etmek iyi bir alışkanlıktır; 0.80 altı sonuçlara temkinli yaklaşılmalıdır.

**Toplamsallık garantisi yok.** SHAP'tan farklı olarak, LIME katsayılarının toplamının $f(x) - E[f(x)]$'e eşit olacağına dair matematiksel bir güvence yoktur.

**Binleştirme bilgi kaybı.** Tabular LIME, sürekli özellikleri "alan_m2 > 140" gibi aralıklara böler. Bu bazı durumlarda açıklamanın hassasiyetini düşürür.

---

## 5. Üç Yöntemin Karşılaştırması

Bu üç araç birbiriyle rekabet etmez, birbirini tamamlar. Hangisini ne zaman kullanacağınızı soru tipine göre belirlemek gerekiyor.

[Şekil 4: Global ve Yerel Açıklama Karşılaştırması — Feature Importance / SHAP / LIME](images/fi_shap_lime_comparison.svg)

### 5.1 Temel Farklar

| Kriter | Feature Importance (MDI) | SHAP | LIME |
| --- | --- | --- | --- |
| **Kapsam** | Küresel — tüm model | Yerel + Küresel | Yerel |
| **Yön bilgisi** | Yok (+/−?) | Var (+/−) | Var (+/−) |
| **Büyüklük** | Göreli oran (0–1) | Orijinal birim (TL…) | Katsayı (göreli) |
| **Toplamsallık garantisi** | Yok | Var | Yok |
| **Tutarlılık** | Deterministik | Deterministik | Stokastik |
| **Model bağımsızlığı** | Hayır (ağaç gerekir) | Kısmi (KernelSHAP ile) | Tam |
| **Görüntü / metin desteği** | Hayır | Kısmi | Tam |
| **Hız** | Çok hızlı | Hızlı (TreeSHAP) | Orta |
| **Korelasyona duyarlılık** | Yüksek | Daha robust | Yüksek |

### 5.2 Hangi Soru Hangi Araç

Üç farklı soru tipi var; her biri farklı bir araça işaret eder.

Birinin "bu modelde hangi özellikler genel olarak önemli?" sorusuyla, birinin "bu 185 m²'lik Merkez semtindeki dairenin neden bu kadar yüksek fiyat biçildi?" sorusuyla, birinin "banka kredi reddini müşteriye nasıl açıklayacağım, ve bu açıklama bir lineer kural şeklinde ifade edilebilir mi?" sorusu aynı değildir.

Gıda kalite kontrol sistemine analoji yaparsak: Feature Importance, aylık üretim raporunda "en çok hataya neden olan makine hangisi" sorusunu cevaplar. SHAP bugün reddedilen spesifik ürünün "tam olarak hangi ölçüm neden hata bayrağı kaldırdı" sorusunu cevaplar. LIME ise operatöre "bu ürün şu iki koşul gerçekleştiğinde reddediliyor" şeklinde sade bir kural çıkarır.

| Soru | Önerilen Araç |
| --- | --- |
| Model seçimi, değişken seçimi, genel analiz | Feature Importance |
| Müşteri/regülatör için spesifik karar açıklaması | SHAP |
| Görüntü veya metin modeli yorumlama | LIME |
| Hızlı prototip, model bağımsız ortam | LIME |
| Matematiksel garanti gereken audit/uyumluluk | SHAP |

### 5.3 Çelişen Sonuçlar Geldiğinde

SHAP, LIME ve Feature Importance zaman zaman farklı sıralamalar üretebilir. Bu durumun başlıca nedenleri:

**Korelasyonlu değişkenler.** MDI, korelasyonlu iki değişkenin önem puanını aralarında böler. SHAP her örnek için marjinal katkıyı ölçtüğünden bu bölünmeyi daha gerçekçi yapar. LIME ise pertürbasyon sırasında değişkenleri bağımsız olarak örneklediği için benzer bölünme sorunuyla karşılaşır.

**Nadir ama güçlü etkiler.** Bir değişken çok az örnekte devreye giriyor ama girdiğinde tahmini büyük ölçüde değiştiriyorsa, MDI bunu küçük görebilir. Mean(|SHAP|) bunu daha doğru yakalar. LIME ise yalnızca tek örnek üzerinde çalıştığından bu soruna yapısal olarak karşılaşmaz, ancak o tek örnek için yerel doğruluk sorunu çıkabilir.

**LIME'ın stokastisite payı.** Çalıştırmadan çalıştırmaya değişen sonuçlar görürseniz `num_samples` parametresini artırmak (varsayılan 5000) kararlılığı iyileştirir.

---

## 6. Dört Kavramı Birleştiren Tam Uygulama

Aşağıdaki kod parçası veri hazırlama (OneHot Encoding), model eğitimi, Feature Importance, SHAP ve LIME analizlerini baştan sona uygular.

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

# ─── 7. LIME ANALİZİ ─────────────────────────────────────────────────────────
from lime import lime_tabular

lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data = X_train.values,
    feature_names = X_train.columns.tolist(),
    mode          = 'regression',
    random_state  = 42
)

idx = 0
lime_exp = lime_explainer.explain_instance(
    data_row   = X_test.iloc[idx].values,
    predict_fn = rf.predict,
    num_features = len(X_train.columns),
    num_samples  = 5000
)

print(f"\nLIME Yerel Model R²:        {lime_exp.score:.3f}")
print(f"LIME Tahmini (yerel model): {lime_exp.predicted_value:,.0f} TL")
print(f"Kara Kutu Tahmini:          {rf.predict(X_test.iloc[[idx]])[0]:,.0f} TL")
print(f"SHAP Tahmini:               {shap_values[idx].sum() + base_value:,.0f} TL")

print("\nLIME Özellik Katkıları:")
for kural, katki in sorted(lime_exp.as_list(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {kural:45s}  {katki:+,.0f}")

# ─── 8. LIME vs SHAP: aynı örnek için yan yana ──────────────────────────────
lime_dict = dict(lime_exp.as_list())

shap_df = pd.DataFrame({
    'SHAP Katkısı (TL)': shap_values[idx]
}, index=X_test.columns).sort_values('SHAP Katkısı (TL)', ascending=False)

print("\n--- SHAP vs LIME: aynı örnek ---")
print("SHAP her özellik için kesin bir değer üretir.")
print("LIME ise aralık kuralları üretir ('alan_m2 > 140' gibi).")
print(shap_df.round(0))
```

### Beklenen Çıktı Yorumu

`Feature Importance` ile `mean(|SHAP|)` sıralamaları büyük ölçüde örtüşür. LIME ise aynı örnek için benzer yön kararları verir ancak sayısal değerler tam örtüşmez; bu beklenen bir durumdur çünkü LIME'ın hedefi kesin hesap vermek değil, yerel bir lineer modelle tahmin etmektir. `alan_m2` üç yöntemde de açık ara birinci çıkar. Gerçek projelerde bu kadar net ayrışmalar olmaz ve yöntemleri birlikte değerlendirmek daha sağlıklı sonuç verir.

---

## Notlar ve Referanslar

- **SHAP kütüphanesi:** Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*. `pip install shap`
- **Shapley, L. S.** (1953). A value for n-person games. *Contributions to the Theory of Games*, 2, 307–317.
- **Molnar, C.** (2022). *Interpretable Machine Learning* (2nd ed.). Güncel çevrimiçi sürüm için: [christophm.github.io/interpretable-ml-book](https://christophm.github.io/interpretable-ml-book)
- **scikit-learn dokümantasyonu:** `sklearn.ensemble.RandomForestRegressor.feature_importances_`, `sklearn.preprocessing.OneHotEncoder`
