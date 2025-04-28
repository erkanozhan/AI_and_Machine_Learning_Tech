# Kalman Filtreleri: Basitten İleri Düzeye Kapsamlı Bir İnceleme

## 1. Giriş: Tahminin Gücü
* **Kalman Filtresi Nedir ve Neden Önemlidir?**
Kalman filtresi, dolaylı ve belirsiz ölçümlerden bir sistemin durumunu tahmin etmek için kullanılan optimal bir tahmin algoritmasıdır. Gürültülü ve eksik verilerle başa çıkma yeteneği, onu doğrudan ölçümün imkansız veya güvenilmez olduğu çeşitli alanlarda vazgeçilmez kılar. Kalman filtresinin gücü, bir dinamik sistemin matematiksel bir modelini ve bu sistem üzerindeki ölçümleri kullanarak zaman içindeki davranışına dair bilgi sağlamasında yatar.

Geleneksel sinyal filtrelerinden farklı olarak, Kalman filtresi sinyali gürültüyü azaltmak veya istenmeyen bileşenleri ortadan kaldırmak için doğrudan değiştirmez. Bunun yerine, bir sistemin zaman içindeki evrimini tanımlayan bir matematiksel modelden yararlanır. Bu model ve sistemden elde edilen ölçümler aracılığıyla, Kalman filtresi geçmişteki davranışları yumuşatabilir, mevcut davranışı tahmin edebilir veya gelecekteki davranışı öngörebilir. Bu çok yönlülük, onu hedef takibi, navigasyon ve kontrol gibi uygulamalarda temel bir bileşen haline getirmiştir.

Kalman filtresinin temelinde, bir sistemin durumuna ilişkin en iyi tahmini sağlamak için gürültülü ve eksik verileri işleme yeteneği yatar. "Optimal" terimi, belirli koşullar altında belirli bir hata metriğini en aza indirdiği anlamına gelir. Özellikle, Kalman filtresi, karesel hatanın ortalamasını en aza indirerek (MMSE) bir sürecin durumunu tahmin etmek için etkili ve hesaplama açısından verimli (özyinelemeli) bir yöntem sunar. Bu özyinelemeli doğası, onu gerçek zamanlı uygulamalar için uygun hale getirir, çünkü yalnızca mevcut girdi ölçümlerini ve daha önce hesaplanan durumu ve belirsizliğini kullanır.

* **Gürültülü Ölçümler ve Belirsiz Sistemler Sorunu.**
Gerçek dünya ölçümleri genellikle gürültülü ve doğası gereği hatalıdır. Sensörlerin sınırlamaları ve çevresel faktörler ölçüm gürültüsüne katkıda bulunur. Örneğin, GPS ölçümleri termal gürültü ve atmosferik etkilerden etkilenebilir. Ayrıca, dinamik sistemler, matematiksel modeller tarafından tam olarak yakalanamayan bozulmalara ve belirsizliklere (süreç gürültüsü) maruz kalır. Hedef hareketi rüzgar ve türbülans gibi dış faktörlerden etkilenebilir.

Hem ölçümlerdeki hem de sistem modellerindeki bu doğal kusurlar, anlamlı bilgileri çıkarmak için Kalman filtresi gibi bir yöntemi gerekli kılar. İdealize edilmiş modeller nadiren gerçek dünya karmaşıklıklarını mükemmel bir şekilde temsil eder. Dış faktörler ve modellerimizdeki basitleştirmeler, modelin tahmini ile sistemin gerçek davranışı arasında sapmalara yol açar. Bu nedenle, hem ölçüm sürecindeki hem de sistemin kendisindeki belirsizlikleri hesaba katan bir tahmin yaklaşımına ihtiyaç vardır.

* **Temel Fikir: Tahminleri Ölçümlerle Birleştirmek.**
Kalman filtresi, bir modele dayalı olarak sistemin durumunu tahmin ederek ve ardından bu tahmini, ağırlıklı bir ortalama kullanarak yeni ölçümlerle güncelleyerek çalışır. Ağırlıklar, tahminin ve ölçümün tahmini belirsizliği (kovaryans) tarafından belirlenir; daha kesin değerlere daha fazla "güvenilir". Bu özyinelemeli süreç, yalnızca mevcut girdi ölçümlerini ve daha önce hesaplanan durumu ve belirsizliğini kullanarak gerçek zamanlı tahmine olanak tanır.

Kalman filtresinin temel yeniliği, önceki bilgileri (tahmin) yeni kanıtlarla (ölçüm) güvenilirliklerine göre akıllıca birleştirme yeteneğidir. Bu ağırlıklı ortalama, modelimize ve ölçümlerimize duyduğumuz istatistiksel güvene dayanmaktadır. Özyinelemeli doğası, onu çevrimiçi uygulamalar için uygun hale getirir, çünkü yalnızca önceki duruma ihtiyaç duyar ve tüm ölçüm geçmişini depolamaktan kaçınır.

## 2. Temel: Durum-Uzay Gösterimi
* **Sistem Durumunu Tanımlama.**
Bir sistemin durumu, belirli bir zamanda koşulunu tanımlayan bir parametreler kümesidir. Örnekler arasında bir nesnenin konumu, hızı ve ivmesi bulunur. Durum genellikle bir durum vektörü olarak temsil edilir. Durum değişkenlerinin seçimi, sistemi etkili bir şekilde modellemek için çok önemlidir. Bu değişkenler, sistemin problemle ilgili davranışını tanımlamak için yeterli olmalıdır.

* **Durum Denklemi: Sistem Dinamiğini Modelleme.**
Durum denklemi, sistemin durumunun zaman içinde nasıl geliştiğini açıklar. Ayrık zamanlı bir sistem için tipik olarak şu şekilde temsil edilir:

$$
x(t + T) = \Phi(T)x(t) + w(t)
$$

Burada `x(t)`, t anındaki durum vektörüdür; `Φ(T)`, durumun t anından t + T anına nasıl dönüştüğünü açıklayan durum geçiş matrisidir; ve `w(t)`, modellenmemiş bozulmaları temsil eden süreç gürültüsü vektörüdür. Durum geçiş matrisi, sistemin deterministik evrimini somutlaştıran doğrusal bir operatördür. Süreç gürültüsü terimi, modelimizin mükemmel olmadığını ve dış etkilerin olabileceğini kabul eder.

* **Ölçüm Denklemi: Durumları Gözlemlere İlişkilendirme.**
Ölçüm denklemi, ölçümlerin sistemin durumuyla nasıl ilişkili olduğunu açıklar. Tipik olarak şu şekilde temsil edilir:

$$
y(t) = Hx(t) + v(t)
$$

Burada `y(t)`, t anındaki ölçüm vektörüdür; `H`, durum ile ölçüm arasındaki doğrusal ilişkiyi tanımlayan ölçüm matrisidir; ve `v(t)`, ölçüm sürecindeki hataları temsil eden ölçüm gürültüsü vektörüdür. Ölçüm matrisi, durum vektörünü durum uzayından ölçüm uzayına eşler ve tüm durum değişkenlerini doğrudan ölçmediğimiz gerçeğini hesaba katar. Ölçüm gürültüsü terimi, sensörlerimizin sınırlamalarını kabul ederek ölçüm sürecindeki yanlışlıkları ve belirsizlikleri hesaba katar.

* **Süreç Gürültüsü ve Ölçüm Gürültüsü: (Belirsizliği Hesaba Katma).**
Süreç gürültüsü `w(t)`'nin sıfır ortalamalı ve `Q(t)` kovaryans matrisine sahip beyaz Gauss gürültüsü olduğu varsayılır:

$$
Q(t) = E[w(t)w(t)ᵀ]
$$

Kovaryans matrisi `Q`, sistemin dinamiğindeki belirsizliği nicelendirir. Köşegen elemanlar, her durum değişkenindeki gürültünün varyansını temsil ederken, köşegen dışı elemanlar aralarındaki kovaryansları temsil eder. Benzer şekilde, ölçüm gürültüsü `v(t)`'nin de sıfır ortalamalı ve `R(t)` kovaryans matrisine sahip beyaz Gauss gürültüsü olduğu varsayılır:

$$
R(t) = E[v(t)v(t)ᵀ]
$$

Kovaryans matrisi `R`, ölçümlerdeki belirsizliği nicelendirir. `w(t)` ve `v(t)`'nin birbirinden bağımsız olduğu varsayılır. Bu varsayım, Kalman filtresinin matematiksel türetilmesini basitleştirir.

## 3. Kalman Filtresi Algoritması:
* **Başlatma: Başlangıç Durumunu ve Kovaryansını Ayarlama.**
Algoritma, başlangıç durum vektörü tahmini `x̂₀` ve kovaryans matrisi `P₀` ile başlar. Bu başlangıç değerleri, sistemin başlangıç koşulu hakkındaki en iyi tahminimizi ve buna ilişkin belirsizliği temsil eder. Ön bilgiye sahipsek, buraya dahil edebiliriz; aksi takdirde, bir tahmin ve yüksek bir belirsizlikle başlayabiliriz. Bazı uygulamalarda, başlangıç durumu ilk ölçümle başlatılabilir. Bu, başlangıç durumu hakkında ön bilgi olmadığında makul bir yaklaşım olabilir ve belirsizlik `P₀` tipik olarak ilk ölçümdeki beklenen gürültüye göre ayarlanır.

* **Tahmin Adımı: Durumu ve Kovaryansı Zamanda İleriye Yansıtma.**
    * Tahmini durum tahmini: Bu adım, önceki tahmine dayalı olarak bir sonraki durumu tahmin etmek için sistemin dinamiğini (`Φ_k` ile temsil edilir) kullanır. Modelimizi kullanarak zamanda ileriye doğru bir projeksiyondur.

    $$
    x̂_{k|k-1} = \Phi_k x̂_{k-1|k-1}
    $$

    * Tahmini durum kovaryans matrisi: Bu adım, belirsizliği zamanda ileriye doğru yansıtır ve süreç gürültüsü `Q_k`, modelimizin mükemmel olmaması ve dış bozulmaların sistemi etkileyebilmesi gerçeğini yansıtarak bu belirsizliğe katkıda bulunur. Zaman ilerledikçe, yalnızca modele dayalı tahminimize olan güvenimiz, modellenmemiş faktörler nedeniyle azalır.

    $$
    P_{k|k-1} = \Phi_k P_{k-1|k-1} \Phi_kᵀ + Q_k
    $$

* **Güncelleme Adımı: Tahminleri İyileştirmek İçin Ölçümleri Dahil Etme.**
    * Kalman Kazancı: Kalman kazancı, ölçüme mi yoksa tahmine mi ne kadar ağırlık verilmesi gerektiğini belirleyen bir matristir. Tahmini durum kovaryansı `P_{k|k-1}`, ölçüm matrisi `H_k` ve ölçüm gürültüsü kovaryansı `R_k`'ye bağlıdır. `R` küçükse (güvenilir ölçümler), kazanç daha yüksek olur ve ölçüme daha fazla ağırlık verilir. `P_{k|k-1}` küçükse (güvenilir tahmin), kazanç daha düşük olur ve tahmine daha fazla ağırlık verilir.

    $$
    K_k = P_{k|k-1} H_kᵀ (H_k P_{k|k-1} H_kᵀ + R_k)⁻¹
    $$

    * Güncellenmiş durum tahmini: Durum tahmini, tahmini duruma, gerçek ölçüm `z_k` ile tahmini duruma dayalı beklenen ölçüm (`H_k x̂_{k|k-1}`) arasındaki farkla orantılı bir düzeltme terimi eklenerek güncellenir. Bu fark, inovasyon veya artık olarak adlandırılır ve ölçümün tahmin tarafından yakalanmayan yeni bilgisini temsil eder.

    $$
    x̂_{k|k} = x̂_{k|k-1} + K_k (z_k – H_k x̂_{k|k-1})
    $$

    * Güncellenmiş durum kovaryans matrisi: Ölçüm dahil edildikten sonra durum tahminindeki belirsizlik azalır. Azalma miktarı Kalman kazancına bağlıdır. Daha güvenilir ölçümler (daha küçük R, daha büyük K), belirsizlikte daha büyük bir azalmaya yol açar.

    $$
    P_{k|k} = (I – K_k H_k) P_{k|k-1} \quad \text{veya} \quad P_{k|k} = P_{k|k-1} – K_k H_k P_{k|k-1}
    $$

* **Kalman Kazancının Rolü.**
Kalman kazancı `K`, tahmini durumu ölçümle optimal bir şekilde harmanlar. "Optimal" burada, güncellenmiş durum tahmininin ortalama karesel hatasını en aza indirdiği anlamına gelir. Bu, Kalman filtresi türetilmesinin temel bir teorik sonucudur. Kalman kazancı, a posteriori hata kovaryansını en aza indirmek için özel olarak seçilir. Yüksek bir Kalman kazancı, ölçümlere daha fazla ağırlık verir, bu da tahmini yeni verilere daha duyarlı hale getirir ancak potansiyel olarak daha gürültülü olabilir. Düşük bir Kalman kazancı, tahmini duruma daha fazla ağırlık verir, bu da daha yumuşak bir tahmine ancak sistemdeki değişikliklere daha yavaş tepkiye neden olabilir. Kalman kazancı, süreç gürültüsü kovaryansı `Q` ve ölçüm gürültüsü kovaryansı `R`'nin göreli büyüklüklerine bağlıdır. `Q`'nun `R`'ye oranı, filtrenin yanıt verme ve gürültü reddetme açısından davranışını belirler. Bu, `Q` ve `R`'nin doğru şekilde belirtilmesinin önemini vurgular.

## 4. Sezgisel İçgörüler  ve Matematiksel Temel
* **Kovaryans Matrisini Bir Belirsizlik Ölçüsü Olarak Anlamak.**
Kovaryans matrisi `P`, durum tahmininin tahmini doğruluğunu temsil eder. Durum hakkındaki inancımızın ne kadar yaygın olduğunun nicel bir ölçüsünü sağlar. Daha sıkı bir dağılım (daha küçük kovaryans), tahminde daha emin olduğumuz anlamına gelir. Köşegen elemanlar, her durum değişkeninin varyanslarını (her bileşendeki belirsizlik) temsil eder. Köşegen dışı elemanlar, farklı durum değişkenleri arasındaki kovaryansları (farklı bileşenlerdeki belirsizliklerin nasıl ilişkili olduğu) temsil eder. Kovaryans, bir durum değişkenindeki hataların diğerindeki hatalarla ilişkili olup olmadığını gösterir. Örneğin, takipta, konumdaki belirsizlik hızdaki belirsizlikle ilişkili olabilir. Daha büyük bir kovaryans, daha büyük bir belirsizliği gösterir.

* **Kalman Kazancı Tahmin ve Ölçüm Arasındaki Güveni Nasıl Dengeler.**
Ölçüm gürültüsü kovaryansı `R` küçükse (ölçümler güvenilirse), Kalman kazancı daha büyük olur ve ölçüme daha fazla ağırlık verir. Süreç gürültüsü kovaryansı `Q` küçükse (sistem modeli güvenilirse), Kalman kazancı daha küçük olur ve tahmine daha fazla ağırlık verir. Tersine, `R` büyükse, kazanç küçüktür ve `Q` büyükse, kazanç büyüktür. Kalman kazancı, modelin doğruluğuna göre ölçümlerin algılanan doğruluğuna dayalı olarak yeni ölçümlerin etkisini ayarlayan bir düzenleyici gibi davranır. Filtrenin gürültülü ölçümleri körü körüne takip etmesini veya yanlış bir modele inatla bağlı kalmasını önler.

* **Tahmin ve Güncelleme Sürecini Görselleştirme.**
Tahmin adımı, mevcut Gauss dağılımını (durum tahminini ve belirsizliğini temsil eder) zamanda ileriye doğru yansıtarak, daha geniş ve daha az kesin bir dağılımla sonuçlanarak görselleştirilebilir. Güncelleme adımı, bu tahmini dağılımın ölçümün Gauss dağılımı ile (belirsizliği ile birlikte) kesişmesini içerir ve iyileştirilmiş durum tahminini temsil eden yeni, daha dar bir dağılımla sonuçlanır. Kalman filtresi esasen bu Gauss dağılımlarını özyinelemeli olarak korur ve günceller. Kalman filtresi, Gauss dağılımları için özyinelemeli bir Bayes tahmincisi olarak anlaşılabilir. Önceki bir inançla (tahmin) başlar ve yeni kanıtlarla (ölçüm) güncelleyerek sonraki bir inanç (güncellenmiş tahmin) elde eder.

## 5. Gerçek Dünya Uygulamaları: Kalman Filtrelerinin Kullanıldığı Yerler
* **Nesne Takibi: Radardan Bilgisayarlı Görüye.**
Kalman filtreleri, radar ölçümlerindeki doğal gürültüyü etkili bir şekilde ele alarak yumuşak ve doğru bir takip sağlamak için havaalanları ve çevresindeki uçakları ve nesneleri takip etmek için kullanılır. Bilgisayarlı görü uygulamalarında füzeleri, yüzleri, kafaları ve elleri takip etmek için kullanılır. Kalman filtreleri, engellemeler ve gürültülü algılamalara rağmen video dizilerindeki nesnelerin tutarlı bir şekilde izlenmesine yardımcı olur. Araçlardaki çeşitli sensörlerden gelen verileri daha güvenilir bir durum tahmini sağlamak için birleştirmek üzere XY düzleminde bir aracın konumunu ve hızını tahmin etmek için kullanılır. Kalman filtreleri, görünüşte basit izleme görevleri bile Kalman filtrelerinin sağlamlığından faydalanabilir.

* **Navigasyon Sistemleri: Uçakları, Uzay Araçlarını ve Otonom Araçları Yönlendirme.**
Kalman filtreleri, Ay'a ve geri giden uzay araçlarının yörüngelerini tahmin etmek için Apollo programında kullanıldı. Bu tarihi uygulama, Kalman filtrelerinin kritik mühendislik görevlerindeki erken başarısını ve gücünü göstermektedir. Denizaltılar, seyir füzeleri ve yeniden kullanılabilir fırlatma araçları için navigasyon sistemlerinin uygulanmasında hayati öneme sahiptir. Kalman filtreleri, talepkar askeri ve havacılık uygulamaları için gerekli doğruluğu ve sağlamlığı sağlar. Otonom araçlarda daha doğru ve sağlam navigasyon için GPS ve atalet ölçüm birimi (IMU) verilerini birleştirmek üzere kullanılır. Sensör füzyonu, otonom sistemlerin güvenliği ve güvenilirliği için çok önemlidir.

* **Sinyal İşleme: Gürültüyü Giderme ve Gürültülü Sinyallerden Bilgi Çıkarma.**
Kalman filtreleri, ses sinyallerinin netliğini ve anlaşılırlığını artırmak için gürültüyü azaltarak konuşma sinyallerini iyileştirmek için kullanılır. Ekonomideki zaman serisi verilerini modellemek ve tahmin etmek için ekonometri alanında kullanılır. Bilgisayarlı görüde derinlik ölçümlerini ve özellik takibini stabilize etmek için kullanılır. Kalman filtreleri, gürültülü derinlik sensörlerini veya kararsız özellik dedektörlerini yumuşatabilir. Opsiyon fiyatlaması ve risk yönetimi için piyasa oynaklığının daha güvenilir tahminlerini sağlamak üzere finansal piyasalardaki oynaklığı tahmin etmek için kullanılır. Piyasa verileri doğası gereği gürültülü ve dinamiktir.

* **Diğer Uygulamalar: Finans, Hava Tahmini ve Daha Fazlası.**
Kalman filtreleri, finansal varlıklar arasındaki değişen ilişkilere uyum sağlayarak, varlıklar arasındaki hedge oranını dinamik olarak tahmin ederek eşleştirilmiş ticarette kullanılır. Beyin, motor sisteminin durumunu nasıl tahmin ettiğine dair gerçekçi bir model sağlayarak merkezi sinir sisteminin hareket kontrolünü modellemek için kullanılır. Hava durumu tahminleri ve ekonomik göstergeler karmaşık ve gürültülüdür ve Kalman filtreleri bu alanlarda sistemleri modellemek için kullanılabilir.

**Tablo 1: Kalman Filtrelerinin Yaygın Uygulamaları**

| Uygulama Alanı       | Özel Örnekler                                                                                               |
| :-------------------- | :---------------------------------------------------------------------------------------------------------- |
| Havacılık/Navigasyon  | Apollo programında uzay aracı yörünge tahmini, denizaltılar, seyir füzeleri ve yeniden kullanılabilir fırlatma araçları için navigasyon sistemleri, otonom araçlarda GPS ve IMU veri füzyonu |
| Robotik/Takip         | Havaalanları çevresinde radar ile uçak ve nesne takibi, bilgisayarlı görüde füze, yüz, kafa ve el takibi, XY düzleminde araç konum ve hız tahmini, video işlemmede aktörler üzerindeki işaretleyici nokta takibi |
| Sinyal İşleme       | Konuşma sinyallerinde gürültü azaltma, ekonometride zaman serisi veri analizi, bilgisayarlı görüde derinlik ölçümü stabilizasyonu ve özellik takibi, finansal piyasalarda oynaklık tahmini                       |
| Finans/Ekonomi        | Eşleştirilmiş ticarette hedge oranı dinamik tahmini                                                              |
| Diğer                 | Merkezi sinir sisteminin hareket kontrolü modellemesi                                                          |

## 6. Doğrusallığı Genişletilmiş Kalman Filtresi (EKF)
* **Doğrusal Olmayan Sistemlerin Zorluğu.**
Standart Kalman filtresi, durum geçiş ve ölçüm denklemlerinin durumun doğrusal fonksiyonları olduğu doğrusal sistemler için tasarlanmıştır. Ancak, birçok gerçek dünya sistemi doğrusal olmayan davranış sergiler. Örnekler arasında, dinamiğinde veya ölçümlerinde trigonometrik fonksiyonlar, karekökler veya diğer doğrusal olmayan ilişkiler bulunan sistemler yer alır. Bu, temel Kalman filtresinin geniş bir gerçek dünya problem yelpazesine uygulanabilirliğini önemli ölçüde sınırlar.

* **Doğrusallaştırma: Doğrusal Olmayan Fonksiyonları Yaklaştırma.**
Genişletilmiş Kalman Filtresi (EKF), doğrusal olmayan sistemleri ele almak için Kalman filtresini genişletir. Bunu, doğrusal olmayan durum geçiş (`f`) ve ölçüm (`h`) fonksiyonlarını, özellikle birinci dereceden yaklaşım olan Taylor serisi açılımını kullanarak mevcut tahminin etrafında doğrusallaştırarak yapar. Bu doğrusallaştırma, doğrusal olmayan fonksiyonların durum ve gürültüye göre Jakobiyen matrislerinin hesaplanmasını içerir. Durum geçiş fonksiyonunun kısmi türevlerinin matrisi olan durum Jakobiyeni (`F_k`) ve ölçüm fonksiyonunun kısmi türevlerinin matrisi olan ölçüm Jakobiyeni (`H_k`). `F_k`, tahmini durum `x̂_{k-1|k-1}` noktasında `f` fonksiyonunun Jakobiyeni iken, `H_k` tahmini durum `x̂_{k|k-1}` noktasında `h` fonksiyonunun Jakobiyenidir.

* **EKF Algoritması: Doğrusal Olmayan Modeller İçin Kalman Filtresini Uyarlama.**
Tahmin adımı, doğrusal olmayan durum geçiş fonksiyonunu doğrudan kullanır:

$$
x̂_{k|k-1} = f(x̂_{k-1|k-1})
$$

Tahmini kovaryans, durum Jakobiyeni kullanılarak güncellenir:

$$
P_{k|k-1} = F_k P_{k-1|k-1} F_kᵀ + Q_k
$$

Kalman kazancı, ölçüm Jakobiyeni kullanılarak hesaplanır:

$$
K_k = P_{k|k-1} H_kᵀ (H_k P_{k|k-1} H_kᵀ + R_k)⁻¹
$$

Güncelleme adımı, inovasyon teriminde doğrusal olmayan ölçüm fonksiyonunu kullanır:

$$
x̂_{k|k} = x̂_{k|k-1} + K_k (z_k – h(x̂_{k|k-1}))
$$

Kovaryans, doğrusal Kalman filtresinde olduğu gibi güncellenir:

$$
P_{k|k} = (I – K_k H_k) P_{k|k-1}
$$

* **EKF Ne Zaman Kullanılır ve Sınırlamaları Nelerdir.**
EKF, sistem modeli veya ölçüm modeli doğrusal olmayan ancak türevlenebilir olduğunda kullanılır. Doğrusal olmayan durum tahmini, navigasyon sistemleri ve GPS'de fiili bir standart olmuştur. Sınırlamaları arasında, doğrusal olmayan sistemler için genel olarak optimal bir tahminci olmaması yer alır. Doğrusallaştırma, özellikle yüksek derecede doğrusal olmayan sistemler için veya çalışma noktası doğrusallaştırma noktasından uzak olduğunda, yaklaşım hataları getirebilir. EKF, doğrusallaştırma yanlışsa sapma sorunları yaşayabilir.

## 7. Daha Sağlam Bir Yaklaşım: Unscented Kalman Filter (UKF)
* **Sigma Noktaları ve Unscented Transformation Kavramı.**
Unscented Kalman Filter (UKF), doğrusal olmayan durum tahmini için EKF'ye türevsiz bir alternatiftir. UKF, doğrusal olmayan fonksiyonları doğrusallaştırmak yerine, Unscented Transformation (UT) adı verilen deterministik bir örnekleme tekniği kullanır. UT, durumun olasılık dağılımını temsil etmek için sigma noktaları adı verilen dikkatlice seçilmiş minimum bir örnek nokta kümesi seçer. Bu sigma noktaları, durum dağılımının ortalamasını ve kovaryansını yakalar. UKF'nin temelinde, doğrusal olmayan fonksiyonların kendilerini yaklaştırmak yerine bir olasılık dağılımını yaklaştırmanın daha kolay olduğu fikri yatar.

* **UKF Algoritması: Doğrusal Olmayan Tahmin İçin Türevsiz Bir Alternatif.**
    * Sigma Noktası Üretimi: Mevcut durum tahmini ve kovaryansına dayalı olarak 2n+1 sigma noktası üretin (burada n, durumun boyutudur).
    * Tahmin: Her sigma noktasını doğrusal olmayan durum geçiş fonksiyonu aracılığıyla yayarak bir dizi dönüştürülmüş sigma noktası elde edin.
    * Bu dönüştürülmüş sigma noktalarının ağırlıklı ortalamasından tahmini durum ortalamasını ve kovaryansını hesaplayın.
    * Sigma noktalarını doğrusal olmayan ölçüm fonksiyonu aracılığıyla yayarak dönüştürülmüş ölçüm sigma noktaları elde edin.
    * Bu dönüştürülmüş ölçüm sigma noktalarının ağırlıklı ortalamasından tahmini ölçüm ortalamasını ve kovaryansını hesaplayın.
    * Tahmini durum ve tahmini ölçüm arasındaki çapraz kovaryansı hesaplayın.
    * Güncelleme: Kovaryans ve çapraz kovaryans matrislerini kullanarak Kalman kazancını hesaplayın.
    * Kalman kazancına ve gerçek ölçüm `z_k` ile tahmini ölçüm arasındaki farka dayalı olarak durum tahminini ve kovaryansını güncelleyin.

* **Yüksek Derecede Doğrusal Olmayan Senaryolarda UKF'nin EKF'ye Göre Avantajları.**
UKF, yüksek derecede doğrusal olmayan sistemler için genellikle EKF'den daha doğrudur, çünkü birinci dereceden doğrusallaştırmanın getirdiği hatalardan kaçınır. Bazı doğrusal olmayan fonksiyonlar için karmaşık ve hatta imkansız olabilen Jakobiyen matrislerinin hesaplanmasını gerektirmez. UKF, belirli doğrusal olmayan gürültü türlerini EKF'den daha iyi işleyebilir. Unscented Transformation, doğrusal olmayan bir dönüşüm geçirdikten sonra bir rastgele değişkenin istatistiklerini yakalamada genellikle daha doğrudur.

## 8. Filtrenizi Tasarlama: Gürültü Kovaryans Matrislerini Anlama
* **Süreç Gürültüsü Kovaryansının (Q) Önemi.**
Süreç gürültüsü kovaryans matrisi `Q`, sistem modelindeki belirsizliği veya sistemin evrimini etkileyen modellenmemiş bozulmaları temsil eder. Kalman filtresinin sistem modelinden gelen tahmine ne kadar güvendiğini belirler. Daha büyük bir `Q`, modelde daha fazla belirsizlik olduğunu gösterir ve filtrenin ölçümlere daha duyarlı olmasına ancak potansiyel olarak daha gürültülü olmasına izin verir. Daha küçük bir `Q`, modele daha fazla güven olduğunu gösterir, bu da daha yumuşak bir tahmine ancak modelden sapmalara daha yavaş tepkiye yol açar. `Q`, bir nesnenin gerçek hareketinin varsayılan hareket modelinden sapabileceği gerçeğini hesaba katar.

* **Ölçüm Gürültüsü Kovaryansının (R) Önemi.**
Ölçüm gürültüsü kovaryans matrisi `R`, sensörlerden elde edilen ölçümlerdeki belirsizliği veya hataları temsil eder. Kalman filtresinin ölçümlere ne kadar güvendiğini belirler. Daha büyük bir `R`, daha gürültülü ölçümleri gösterir ve filtrenin sistem modeline daha fazla güvenmesine ve ölçümlere daha yavaş tepki vermesine neden olur. Daha küçük bir `R`, daha güvenilir ölçümleri gösterir ve filtrenin ölçümlere daha fazla ağırlık vermesine ve daha duyarlı olmasına neden olur. `R`, ölçümler için kullanılan sensörlerin doğruluğunu ve güvenilirliğini yansıtır.

* **Q ve R İçin Başlangıç Değerleri Nasıl Seçilir.**
`R` genellikle sensör özelliklerinden veya sistem bilinen bir durumdayken ölçüm verileri toplanarak ve örnek kovaryansı hesaplanarak tahmin edilebilir. `Q`'nun seçilmesi genellikle daha zordur, çünkü sistem dinamiğinin modellenmemiş yönlerini temsil eder. `Q` için başlangıç değerleri, sistemin fiziksel anlayışına ve beklenen bozulmaların büyüklüğüne dayalı olabilir. Farklı durum değişkenlerindeki ve ölçümlerdeki gürültünün bağımsız olduğu varsayılarak, `Q` ve `R` için köşegen matrislerle başlamak yaygındır.

* **Sistem Davranışına ve Sensör Özelliklerine Dayalı Olarak Q ve R'yi Ayarlama Teknikleri.**
Ayarlama genellikle `Q` ve `R`'yi ayarlama ve filtrenin performansını gözlemleme sürecini içeren yinelemeli bir süreçtir. Filtre çıktısı çok gürültülüyse, `R` çok küçük veya `Q` çok büyük olabilir. Filtre değişikliklere yavaş tepki veriyorsa, `R` çok büyük veya `Q` çok küçük olabilir. Maksimum olabilirlik tahmini gibi teknikler, `Q` ve `R`'yi verilerden tahmin etmek için kullanılabilir. İnovasyon dizisinin (ölçümler ile tahmini ölçümler arasındaki fark) analizi, `Q` ve `R`'nin uygun şekilde ayarlanıp ayarlanmadığına dair içgörüler sağlayabilir. Yüksek manevra kabiliyetine sahip hedefler için daha büyük bir `Q` uygun olabilir.

## 9. Sonuç

**Tablo 2: Kalman Filtresi, EKF ve UKF Karşılaştırması**

| Özellik                       | Doğrusal Kalman Filtresi | Genişletilmiş Kalman Filtresi (EKF) | Unscented Kalman Filter (UKF) |
| :---------------------------- | :----------------------- | :---------------------------------- | :---------------------------- |
| Doğrusal Olmayanlığı Ele Alma | Hayır                    | Birinci Dereceden Doğrusallaştırma  | Unscented Transformation      |
| Doğrusallaştırma Yöntemi      | Yok                      | Jakobiyen Matrisleri                | Sigma Noktaları               |
| Jakobiyen Gereksinimi         | Yok                      | Gerekli                             | Gerekli Değil                 |
| Güçlü Doğrusal Olmayanlıkta Doğruluk | Sınırlı          | Genellikle Daha Düşük             | Genellikle Daha Yüksek        |
| Hesaplama Maliyeti            | Düşük                    | Orta                                | Orta                          |

Sonuç olarak, Kalman filtresi, gürültülü ve belirsiz verilerden dinamik sistemlerin durumunu tahmin etmek için güçlü ve çok yönlü bir araçtır. Temel ilkeleri, doğrusal olmayan sistemleri ele almak için uzantıları ve tasarım hususlarının anlaşılması, çeşitli mühendislik ve bilimsel disiplinlerdeki uygulamaları için hayati öneme sahiptir.