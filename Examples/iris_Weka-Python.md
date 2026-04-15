# IRIS Veri Seti ile Sınıflandırma ve Kümeleme: Weka ve Python

IRIS veri seti küçük görünür; ama öğretimde çok işe yarar. Aynı dört ölçümden yola çıkıp iki ayrı soruyu çözebiliriz. Etiketli örnekler varsa bir çiçeğin türünü tahmin ederiz. Etiket yoksa birbirine benzeyen örnekleri gruplarız. Mutfakta üzerinde adı yazılı kavanozları rafa dizmek başka bir iştir; etiketi düşmüş kavanozları sadece renk, koku ve dokuya bakarak ayırmak başka bir iş. Birincisi sınıflandırma, ikincisi kümelemedir.

Bu notta önce kavramı sade bir dille kuracağız, ardından aynı veri setini hem Weka hem Python ile işleyeceğiz. Akış şemaları için [sınıflandırma çizimi](../images/iris_classification_flow.svg) ve [kümeleme çizimi](../images/iris_clustering_flow.svg) bağlantılarına bakabilirsiniz.

## IRIS veri setini tanıyalım

IRIS veri seti 150 gözlemden oluşur. Her gözlem bir çiçeği temsil eder. Üç tür vardır:

- `setosa`
- `versicolor`
- `virginica`

Her çiçek için dört sayısal özellik ölçülür:

| Özellik | Açıklama |
|---|---|
| `sepal length` | çanak yaprak uzunluğu |
| `sepal width` | çanak yaprak genişliği |
| `petal length` | taç yaprak uzunluğu |
| `petal width` | taç yaprak genişliği |

Bir örneği matematiksel olarak şöyle gösterebiliriz:

$$
\mathbf{x} = [x_1, x_2, x_3, x_4]
$$

Burada her bileşen bir ölçümdür. Eğer elimizde tür bilgisi de varsa buna etiket, yani sınıf etiketi deriz. Sınıf (class) sözcüğü Latince *classis* kökünden gelir; ayrılmış grup anlamı taşır. Etiketli veriyle öğrenme işine denetimli öğrenme (supervised learning), etiketsiz veriyle benzer örnekleri toplama işine denetimsiz öğrenme (unsupervised learning) denir.

## Aynı veriyle iki ayrı problem

| Soru | Kullanılan bilgi | Amaç | Tipik çıktı |
|---|---|---|---|
| Sınıflandırma (classification) | Özellikler + gerçek tür etiketi | Yeni bir örneğin türünü tahmin etmek | Tür adı, karışıklık matrisi, doğruluk |
| Kümeleme (clustering) | Yalnızca özellikler | Benzer örnekleri doğal gruplara ayırmak | Küme numarası, küme merkezleri, silüet skoru |

Sınıflandırmada öğretmen cevabı önceden biliyordur; öğrenciye örnek çözdürür. Kümelemede ise masaya karışık tohum torbaları dökülmüştür; benzer olanları bir araya getiririz, ama torbaların üstünde isim yoktur.

## Sınıflandırma

Sınıflandırma, yeni bir örneğe doğru etiketi verme problemidir. Burada model, daha önce gördüğü etiketli örneklerden bir karar kuralı çıkarır. Bir sera çalışanını düşünün: birkaç hafta sonra yeni gelen fidelerin yaprak ölçülerine bakıp hangi türe ait olduğunu tahmin etmeye başlar. Modelin yaptığı iş de buna benzer.

En çok kullanılan ölçütler şunlardır:

- Doğruluk (accuracy): toplam tahminlerin ne kadarının doğru olduğunu gösterir.
- Kesinlik (precision): bir sınıfa ait dediklerimizin ne kadarının gerçekten o sınıfa ait olduğunu gösterir.
- Duyarlılık (recall): o sınıfa ait gerçek örneklerin ne kadarını yakaladığımızı gösterir.
- F1 ölçütü (F1-score): kesinlik ile duyarlılığın dengeli ortalamasıdır.
- Cohen uyum katsayısı (Cohen's kappa): tesadüfî uyuşmayı dışarıda bırakarak uyumu ölçer.

Doğruluk için temel oran şöyledir:

$$
\text{Accuracy} = \frac{\text{Doğru Tahmin Sayısı}}{\text{Toplam Tahmin Sayısı}}
$$

Cohen uyum katsayısı ise şu fikirle çalışır: Gözlenen uyum yüksek olabilir, ama bunun bir kısmı şans eseri oluşmuş olabilir.

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

Burada $p_o$ gözlenen uyumu, $p_e$ ise tesadüfî beklenen uyumu gösterir.

### Weka ile sınıflandırma

Weka, görsel arayüzü sayesinde ilk temas için çok uygundur. Özellikle küçük veri setlerinde, adımların görünür olması öğrencinin kafasında süreci oturtur.

1. Weka'yı açın ve `Explorer` bölümüne girin.
2. `Open file` ile `iris.arff` dosyasını yükleyin.
3. `Preprocess` sekmesinde son sütunun `class` olduğunu doğrulayın.
4. `Classify` sekmesine geçin.
5. `Choose` düğmesinden `trees > J48` seçin. Bu, Weka içindeki C4.5 karar ağacı uygulamasıdır.
6. `Test options` altında `Cross-validation` seçin ve kat sayısını `10` yapın.
7. `Start` düğmesine basın.

Burada özellikle `10` katlı çapraz doğrulama (cross-validation) kullanıyoruz. Bunun nedeni veri setinin küçük olmasıdır. Tek bir `%80-%20` bölmesi bazen şanslı, bazen şanssız bir ayrım üretir. Çapraz doğrulama aynı veri üzerinde nöbetleşe eğitim ve sınama yapar; böylece daha dengeli bir performans resmi verir.

J48 sonucunda şu bölümlere dikkat edin:

- `Correctly Classified Instances`
- `Incorrectly Classified Instances`
- `Kappa statistic`
- sınıf bazlı `Precision`, `Recall`, `F-Measure`
- `Confusion Matrix`

Ardından aynı işlemi şu algoritmalarla tekrarlayın:

- `bayes > NaiveBayes`
- `trees > RandomForest`
- `functions > SMO`

`SMO`, Sequential Minimal Optimization ifadesinin kısaltmasıdır; Türkçede ardışık en küçük eniyileme olarak çevrilebilir. Weka'da destek vektör makinelerini eğitmek için sık kullanılır.

Karşılaştırma yaparken yalnızca tek bir sayıya takılmayın. İki modelin doğruluğu yakın olabilir; ama birinin `versicolor` ile `virginica` ayrımında daha dengeli çalıştığını karışıklık matrisinde görebilirsiniz. Bu, fabrikada üç farklı kutuyu ayıran işçinin toplam hatasının aynı olmasına rağmen belirli iki kutuda sürekli karışıklık yapmasına benzer.

### Python ile sınıflandırma

Burada önemli bir düzeltme yapıyoruz: `classification_report` çıktısı kendi başına Cohen uyum katsayısını vermez. Bu yüzden kodda bu ölçütü ayrıca hesaplıyoruz. Ayrıca tek bir eğitim-test ayrımına güvenmek yerine, önce eğitim verisi üzerinde çoklu model karşılaştırması yapıyoruz.

Örnek dosya: [iris_classification.py](/e:/Repolar/AI_ML_YL/AI_and_Machine_Learning_Tech/Examples/iris_classification.py)

Gerekli paket:

```bash
python -m pip install scikit-learn
```

Kod:

```python
from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_models() -> dict[str, object]:
    return {
        "Karar Agaci": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Destek Vektor Makinasi": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale"),
        ),
        "Rastgele Orman": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
    }


def print_table(rows: list[dict[str, float]]) -> None:
    header = (
        f"{'Model':<26}"
        f"{'Accuracy':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'Kappa':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['name']:<26}"
            f"{row['accuracy']:>10.3f}"
            f"{row['precision']:>12.3f}"
            f"{row['recall']:>10.3f}"
            f"{row['f1']:>10.3f}"
            f"{row['kappa']:>10.3f}"
        )


def main() -> None:
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = build_models()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
        "kappa": make_scorer(cohen_kappa_score),
    }

    rows: list[dict[str, float]] = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        rows.append(
            {
                "name": name,
                "accuracy": scores["test_accuracy"].mean(),
                "precision": scores["test_precision"].mean(),
                "recall": scores["test_recall"].mean(),
                "f1": scores["test_f1"].mean(),
                "kappa": scores["test_kappa"].mean(),
            }
        )

    rows.sort(key=lambda row: row["accuracy"], reverse=True)
    print("10 katli capraz dogrulama ortalamalari")
    print_table(rows)

    best_model_name = rows[0]["name"]
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("\nSecilen model:", best_model_name)
    print("Test kumesi Cohen kappa:", round(cohen_kappa_score(y_test, y_pred), 3))
    print("\nSiniflandirma raporu")
    print(classification_report(y_test, y_pred, target_names=iris.target_names, digits=3))
    print("Karisiklik matrisi")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
```

Bu kod üç önemli işi düzgün yapar:

1. Eğitim ve test kümelerini `stratify=y` ile dengeli ayırır.
2. Farklı algoritmaları aynı koşullarda karşılaştırır.
3. Metinde adı geçen ölçütleri gerçekten üretir; özellikle Cohen uyum katsayısını ayrıca hesaplar.

Burada `Destek Vektör Makinasi` için `StandardScaler` kullanmamızın nedeni mesafe ve ölçek duyarlılığıdır. Cetveli santimetre yerine milimetreye çevirdiğinizde bazı algoritmaların davranışı değişebilir. Ağaç tabanlı yöntemler bu değişimden daha az etkilenir; ama destek vektör makineleri ve en yakın komşu türü yöntemler doğrudan etkilenir.

Bu depoda yapılan doğrulama çalıştırmasında `Destek Vektör Makinasi` en iyi ortalama sonucu verdi. `10` katlı çapraz doğrulamada doğruluk yaklaşık `0.967`, Cohen uyum katsayısı yaklaşık `0.950` çıktı. Ayrı tutulan test kümesinde ise yalnızca bir `versicolor` örneği `virginica` olarak karıştı. Bu sonuç, IRIS veri setinde ayrımın büyük ölçüde güçlü olduğunu; kalan zorluğun ise çoğunlukla bu iki tür arasında toplandığını gösterir.

## Kümeleme

Kümeleme, etiket olmadan örnekleri benzerliklerine göre gruplama işidir. Pazarda karışık kasalar halinde gelen meyveleri düşünün. Kasaların üstünde isim yoksa, önce renk, boyut ve dokuya bakarak gruplar kurarsınız. Sonra bu grupların elma mı armut mu olduğuna ayrıca karar verirsiniz. Kümeleme tam olarak bu ilk aşamadır.

Bu bölümde `K-Means` algoritmasını kullanacağız. Adındaki `K`, kurulacak küme sayısını gösterir. Algoritma, her kümenin merkezini bulmaya ve örnekleri bu merkezlere göre ayırmaya çalışır.

Temel amaç fonksiyonu şöyledir:

$$
J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \lVert x_i - \mu_k \rVert^2
$$

Burada:

- $C_k$, `k` numaralı kümeyi,
- $\mu_k$, o kümenin merkezini,
- $\lVert x_i - \mu_k \rVert^2$, örneğin merkezden uzaklığının karesini gösterir.

Yani algoritma, her kümeyi kendi merkezine olabildiğince yakın örneklerle kurmaya çalışır.

### Weka ile kümeleme

1. `Explorer > Open file` ile yine `iris.arff` dosyasını açın.
2. `Cluster` sekmesine geçin.
3. `Choose` düğmesinden `SimpleKMeans` seçin.
4. Parametrelerde küme sayısını `3` yapın.
5. Uzaklık ölçüsü olarak `EuclideanDistance` bırakın.
6. Başlangıç tohumu için `seed = 42` kullanın.
7. Sonucu daha kolay incelemek için `Classes to clusters evaluation` seçeneğini etkinleştirin.
8. `Start` düğmesine basın.

Burada iki ayrı şeye bakın:

- `Clustered Instances`: kümelerin büyüklüğü dengeli mi?
- `Classes to Clusters`: bulunan kümeler gerçek türlerle ne ölçüde örtüşüyor?

IRIS veri setinde çoğu zaman `setosa` ayrı bir küme olarak kolay seçilir. Asıl karışma, `versicolor` ile `virginica` arasında olur. Bunun nedeni, bu iki türün bazı ölçülerde birbirine yaklaşmasıdır.

Saf denetimsiz inceleme yapmak isterseniz `class` sütununu geçici olarak kaldırabilirsiniz. Ama öğretimde sınıf sütununu koruyup kümelerin gerçek türlerle ne kadar örtüştüğünü görmek daha açıklayıcı olur.

### Python ile kümeleme

Örnek dosya: [iris_clustering.py](/e:/Repolar/AI_ML_YL/AI_and_Machine_Learning_Tech/Examples/iris_clustering.py)

Kod:

```python
from __future__ import annotations

import os
from collections import Counter

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler


def main() -> None:
    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=3, n_init=20, random_state=42)
    cluster_labels = model.fit_predict(X_scaled)

    print("Siluet skoru:", round(silhouette_score(X_scaled, cluster_labels), 3))
    print("Duzeltilmis Rand indeksi:", round(adjusted_rand_score(y, cluster_labels), 3))

    print("\nKume buyuklukleri")
    for cluster_id, size in sorted(Counter(cluster_labels).items()):
        print(f"Kume {cluster_id}: {size} ornek")

    print("\nKume merkezleri (orijinal olcekte)")
    original_centers = scaler.inverse_transform(model.cluster_centers_)
    for cluster_id, center in enumerate(original_centers):
        values = ", ".join(f"{value:.2f}" for value in center)
        print(f"Kume {cluster_id}: {values}")

    print("\nGercek sinif - kume karsilastirma matrisi")
    matrix = contingency_matrix(y, cluster_labels)
    print(matrix)

    print("\nSatirlar gercek turleri gosterir:")
    for class_id, class_name in enumerate(iris.target_names):
        print(f"{class_id}: {class_name}")


if __name__ == "__main__":
    main()
```

Bu kodda iki değerlendirme ölçütü özellikle önemlidir:

- Silüet skoru (silhouette score): kümelerin kendi içinde ne kadar sıkı, birbirine karşı ne kadar ayrık olduğunu ölçer. `1` değerine yaklaşması daha iyi ayrışmayı gösterir.
- Düzeltilmiş Rand indeksi (adjusted Rand index): bulunan kümelerin, yalnızca öğretim amacıyla elimizde tuttuğumuz gerçek etiketlerle ne kadar uyuştuğunu gösterir.

Burada yine `StandardScaler` kullanıyoruz. Çünkü `K-Means` uzaklıkla çalışır. Ölçeği büyük bir özellik, sırf sayısal aralığı daha geniş diye kümeyi gereğinden fazla etkileyebilir. Bahçede tohumları yalnızca boyuna göre dizerken renk bilgisini neredeyse yok saymak gibi bir hata doğar.

Bu depoda yapılan çalıştırmada silüet skoru yaklaşık `0.460`, düzeltilmiş Rand indeksi ise yaklaşık `0.620` bulundu. Kümelerden biri `setosa` sınıfını neredeyse tek başına yakalarken, diğer iki kümede `versicolor` ile `virginica` birbirine karıştı. Bu da kümelemenin, etiket görmeden bile veri içindeki doğal ayrımı belirli ölçüde yakalayabildiğini; ancak sınıflandırma kadar keskin bir sınır çizmediğini gösterir.

## Ölçütleri nasıl okuyacağız?

Bir modeli değerlendirirken şu sıra oldukça sağlıklıdır:

1. Önce genel tabloya bakın: doğruluk ve Cohen uyum katsayısı.
2. Sonra sınıf bazlı ölçütlere inin: özellikle `recall` düşük kalan sınıf var mı?
3. Ardından karışıklık matrisine bakın: hata hep aynı iki sınıf arasında mı toplanıyor?
4. Kümelemede önce iç yapıya bakın: silüet skoru kümeler sıkı mı?
5. Elinizde gerçek etiket varsa, son aşamada dış karşılaştırma yapın: düzeltilmiş Rand indeksi ne diyor?

Bu sıralama, sadece sınavdan alınan toplam nota bakmak yerine hangi konularda yanlış yapıldığını da görmeye benzer. İyi bir rapor, toplam başarıyı verir; ama asıl öğretici olan hata desenidir.

## Sık yapılan hatalar

- `classification_report` çıktısının Cohen uyum katsayısını verdiğini sanmak.
- Küçük veri setinde tek bir eğitim-test ayrımıyla kesin hüküm vermek.
- Sınıflar dengesiz olabileceği halde `stratify` kullanmamak.
- Uzaklık temelli algoritmaları ölçekleme yapmadan karşılaştırmak.
- Kümeleme sonucunu yalnızca gerçek etiketlere bakarak yorumlamak; iç yapı kalitesini görmezden gelmek.

Bu nottaki Python örnekleri özellikle bu beş noktayı düzeltmek için yazıldı.

## Sonuç

IRIS veri seti üzerinde çalışırken iki ayrı düşünme biçimi öğrenmiş olduk. Sınıflandırmada elimizde cevap anahtarı vardı; modelin ne kadar doğru öğrendiğini ölçtük. Kümelemede cevap anahtarı yoktu; benzer örneklerin doğal kümeler oluşturup oluşturmadığına baktık. Aynı veri setiyle iki farklı soru sormak, hangi yöntemin hangi problem için uygun olduğunu ayırt etmenin en temiz yollarından biridir.
