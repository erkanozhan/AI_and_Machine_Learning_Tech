# Yapay Zekâ ve Makine Öğrenmesi Teknolojileri

Bu depo, lisansüstü **Yapay Zekâ ve Makine Öğrenmesi Teknolojileri** dersinin notlarını ve örnek uygulamalarını içerir.

- **Web sürümü:** [erkanozhan.github.io/AI_and_Machine_Learning_Tech](https://erkanozhan.github.io/AI_and_Machine_Learning_Tech/). Aynı notlar; not listesi, tüm notlarda arama, formüller ve diyagramlarla birlikte.
- **PDF / yazdırma:** [Notların tamamı tek sayfada](https://erkanozhan.github.io/AI_and_Machine_Learning_Tech/yazdir.html) açılıp Ctrl+P ile "PDF olarak kaydet" seçilebilir. Aynı PDF `python araclar/pdf_olustur.py` komutuyla da üretilir (Chrome veya Edge gerekir).

**Ön koşul.** Makine öğrenmesine ilişkin lisans düzeyinde bir temeliniz yoksa önce lisans dersinin deposundaki tüm ders notlarını okumanız önerilir: [Makine Öğrenmesine Giriş (lisans)](https://github.com/erkanozhan/machinelearning).

## Önerilen Okuma Sırası

| Sıra | Dosya | Konu |
| :---: | :--- | :--- |
| 1 | [Expected_Utility.md](Expected_Utility.md) | Fayda teorisi ve beklenen fayda: belirsizlik altında karar verme, riskten kaçınma, Allais paradoksu |
| 2 | [Hidden_Markov_Models.md](Hidden_Markov_Models.md) | Markov zincirleri, Saklı Markov Modelleri; ileri, Viterbi ve Baum–Welch algoritmaları |
| 3 | [Kalman_Filters.md](Kalman_Filters.md) | Durum uzayı modeli, Kalman filtresi, EKF ve UKF; çözümlü kuş takibi örneği |
| 4 | [Feature_Importance_SHAP_OneHotEncoding.md](Feature_Importance_SHAP_OneHotEncoding.md) | Yorumlanabilir makine öğrenmesi: OneHot Encoding, özellik önemi, SHAP ve LIME |
| 5 | [Examples/iris_Weka-Python.md](Examples/iris_Weka-Python.md) | IRIS veri kümesinde sınıflandırma ve kümeleme: Weka ve Python |

## Dizin Yapısı

```text
*.md                     Ders notları
Examples/                IRIS örnekleri (Weka adımları ve Python betikleri)
  iris_classification.py   Sınıflandırma: 10 katlı çapraz doğrulama, Cohen kappa
  iris_clustering.py       Kümeleme: k-ortalamalar, silüet skoru, düzeltilmiş Rand indeksi
images/                  Şekiller (SVG)
requirements.txt         Python paketleri
index.html, _sidebar.md, docsify-ayar.js, docsify-stil.css
                         Web sürümü: docsify sayfası, not listesi, ayarlar ve stiller
yazdir.html              Notların tamamını tek sayfada gösteren yazdırma sayfası
araclar/pdf_olustur.py   PDF üreten betik
```

## Kurulum

Python 3.10 veya üzeri önerilir. Gerekli paketler:

```bash
python -m pip install -r requirements.txt
```

Örneklerin çalıştırılması:

```bash
cd Examples
python iris_classification.py
python iris_clustering.py
```

Weka adımları için Weka 3.8.x yeterlidir: <https://waikato.github.io/weka-wiki/downloading_weka/>.
