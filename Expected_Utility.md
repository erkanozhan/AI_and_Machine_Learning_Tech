# Fayda Teorisi (Utility Theory) ve Beklenen Fayda Teorisi (Expected Utility Theory)

## 1. Giriş ve Temel Kavramlar

**Fayda teorisi** (utility theory), insanların risk ve belirsizlik altında nasıl karar verdiklerini ve seçenekler arasında nasıl tercih yaptıklarını nicel olarak çözümlemeyi amaçlar. *Utility* sözcüğü Latince *utilitas* (yararlılık) kökünden gelir. Teori, bir kişinin seçenekler arasındaki tercihlerine ve bu tercihlerin sıralanmasına ilişkin varsayımlardan oluşur; bu varsayımlar sağlandığında tercihler sayısal bir **fayda fonksiyonuyla** temsil edilebilir.

**Beklenen fayda teorisi** (Expected Utility Theory, EUT), belirsizlik altında rasyonel karar vermeye ilişkin baskın kuramsal çerçevedir. Bu teoriye göre karar verici, seçeneklerini olası sonuçların faydalarının olasılıklarla ağırlıklandırılmış ortalamasına, yani **beklenen faydaya** (expected utility) göre sıralar ve en yüksek beklenen faydalı seçeneği seçer. Teori, yalnızca parasal değerlerin en iyi seçimi belirlemede yeterli olmadığını, sonuçların karar verici açısından taşıdığı öznel değerin (faydanın) dikkate alınması gerektiğini kabul eder. Belirsiz sonuçlar arasında nasıl seçim yapılacağı sorusuna yanıt aradığı için iktisat ve karar bilimlerinin tamamına yayılmış ve neoklasik iktisadi düşüncenin temel dayanaklarından biri olmuştur.

---

## 2. Kuramsal Temeller ve Kapsam

Beklenen fayda teorisi, von Neumann–Morgenstern fayda fonksiyonunu temel alır. Bu fonksiyon, belirsiz bir seçeneği (bir **piyango**, lottery: olası sonuçlar ve olasılıkları) karar vericinin onunla eşdeğer gördüğü kesin bir tutara, yani **kesinlik eşdeğerine** (certainty equivalent) dönüştürmeyi mümkün kılar. Teorinin matematiksel yapısı, rasgele değişken (random variable) ve beklenen değer kavramlarına dayanır.

Teori iki farklı amaçla kullanılır. **Normatif** (normative) olarak, rasyonel bir bireyin nasıl karar vermesi **gerektiğine** dair bir ölçüt sunar. **Betimleyici** (descriptive) olarak ise gerçek insanların nasıl karar **verdiğini** açıklamaya çalışır; ancak bu ikinci kullanımda, Bölüm 8'de görüleceği gibi, önemli sınırlılıkları vardır.

Geçmiş kesin, gelecek ise belirsiz olduğundan her karar aslında belirsiz gelecekler arasından birini seçmek demektir. Bu nedenle beklenen fayda teorisi, kişisel finanstan kurumsal stratejiye, proje yönetiminden yapay zekâda karar veren etmenlere (agents) kadar çok geniş bir alanda kullanılır. Özellikle yatırım seçeneklerini değerlendirmede ve risk ile getiri arasındaki ödünleşimi (trade-off) bireysel ya da kurumsal risk tercihlerine göre çözümlemede yapılandırılmış bir yaklaşım sunar.

---

## 3. Tarihsel Gelişim

Beklenen fayda teorisinin kökleri 1738 yılına uzanır. Daniel Bernoulli bu tarihte **St. Petersburg paradoksunu** (St. Petersburg paradox) ele almış ve karar vericilerin yalnızca parasal değeri değil, sonuçların kendilerine sağladığı faydayı dikkate aldığını ileri sürmüştür.

Paradoks şöyledir: Bir oyunda yazı gelene kadar madeni para atılır; yazı $n$'inci atışta gelirse oyuncu $2^n$ lira kazanır. Kazancın beklenen değeri

$$E[X] = \sum_{n=1}^{\infty} \frac{1}{2^n} \cdot 2^n = 1 + 1 + 1 + \cdots = \infty$$

olduğu hâlde, insanlar bu oyuna katılmak için birkaç liradan fazlasını ödemek istemez. Bernoulli, paranın faydasının azalan oranda arttığını (ek bir liranın zengine yoksula olduğundan daha az yarar sağladığını) öne sürerek logaritmik bir fayda fonksiyonu önerdi. $U(x) = \ln x$ ile oyunun beklenen faydası sonludur:

$$E[U(X)] = \sum_{n=1}^{\infty} \frac{1}{2^n} \ln(2^n) = \ln 2 \sum_{n=1}^{\infty} \frac{n}{2^n} = 2 \ln 2 \approx 1.386$$

Bu beklenen faydaya karşılık gelen kesin tutar $e^{2 \ln 2} = 4$ liradır; oyun bu karar verici için 4 lira değerindedir.

Bernoulli'nin yaklaşımı ölçülebilir, sayısal bir fayda, yani **kardinal fayda** (cardinal utility) gerektirdiğinden uzun süre sınırlı kabul gördü. 1926'da Frank Ramsey öznel olasılık ve fayda için ilk biçimsel aksiyomatik yaklaşımı geliştirdi (yayımı 1931). John von Neumann ve Oskar Morgenstern 1944 yılında yayımladıkları *Theory of Games and Economic Behavior* adlı eserde, tercihlere ilişkin birkaç aksiyomdan hareketle beklenen fayda kuralını türeterek alanda bir dönüm noktası oluşturdu. Bu formülasyonda fayda, yalnızca seçeneklerin sıralamasını veren **sıralı faydadan** (ordinal utility) fazlasını içerir; ancak keyfî bir artı ve pozitif bir çarpana kadar belirlidir. Leonard J. Savage (1954) ise öznel olasılıkları da içeren genel bir çerçeve kurdu.

Bu gelişmelerden sonra beklenen fayda teorisi hızla yayıldı ve 1950'lerden itibaren risk altında karar vermenin ekonomik çözümlemesinin temeli oldu. Erken dönem savunucuları arasında Chicago Üniversitesi'nden Milton Friedman ve Leonard J. Savage ile Cowles Komisyonu'ndan Jacob Marschak bulunur; MIT'den Paul Samuelson gibi bazı önemli iktisatçılar ise başlangıçta teoriye eleştirel yaklaşmıştır.

---

## 4. Karar Teorisi ve Belirsizlik

**Karar teorisi** (decision theory), bireylerin ya da sistemlerin seçenekler arasından nasıl seçim yaptığını inceler. Kararlar genellikle üç koşul altında ele alınır:

- **Kesinlik altında karar** (decision under certainty): Her seçeneğin sonucu bilinir.
- **Risk altında karar** (decision under risk): Sonuçlar belirli ve bilinen olasılıklarla gerçekleşir.
- **Belirsizlik altında karar** (decision under uncertainty): Sonuçların olasılıkları bile bilinmez; karar verici olasılıkları öznel olarak kestirmek zorundadır.

**Fayda**, bireyin tercihlerini ifade etmek için kullanılan soyut bir değerdir: Her olası sonuca, bireyin o sonuçtan ne kadar memnun olduğunu gösteren bir sayı atanır. Bu atama bir **fayda fonksiyonuyla** yapılır:

$$U: O \rightarrow \mathbb{R}$$

- $U$ (utility): fayda fonksiyonu.
- $O$ (outcomes): olası tüm sonuçların kümesi.
- $\rightarrow$: her sonucu bir sayıya eşleyen (mapping) gösterim.
- $\mathbb{R}$: gerçek sayılar kümesi.

Daha yüksek faydalı sonuç tercih edilir. Teori, karar vericinin tercihlerinin tamlık ve geçişlilik gibi rasyonellik koşullarını sağladığını varsayar (Bölüm 5.3).

---

## 5. Beklenen Fayda Teorisinin Matematiksel Modeli

### 5.1 Beklenen Fayda

Bir $A$ eyleminin farklı olasılıklarla çeşitli sonuçlara yol açabildiği durumda $A$'nın beklenen faydası şöyle tanımlanır:

$$EU(A) = \sum_{i} P(O_i \mid A)\, U(O_i)$$

- $EU(A)$: $A$ eyleminin beklenen faydası.
- $\sum_i$: tüm olası sonuçlar üzerinden toplam.
- $O_i$: gerçekleşebilecek $i$'inci sonuç.
- $P(O_i \mid A)$: $A$ eylemi seçildiğinde $O_i$ sonucunun gerçekleşme olasılığı (koşullu olasılık).
- $U(O_i)$: $O_i$ sonucunun karar vericiye sağladığı fayda.

Sonuç, eylem ile karar vericinin denetimi dışındaki **dünyanın durumu** tarafından birlikte belirleniyorsa (Bölüm 6), aynı tanım durumlar üzerinden yazılabilir. $s$ olası bir durum, $o(A, s)$ ise $A$ eylemi $s$ durumunda seçildiğinde ortaya çıkan sonuç olmak üzere:

$$EU(A) = \sum_{s} P(s)\, U\big(o(A, s)\big)$$

### 5.2 Karar Kuralı

En yüksek beklenen faydayı sağlayan eylem seçilir:

$$A^{*} = \operatorname*{argmax}_{A} EU(A)$$

$\operatorname{argmax}_A$, ardından gelen ifadeyi (burada $EU(A)$) en büyük yapan eylemi döndürür: "beklenen faydası en yüksek olan eylemi bul." Birden fazla eylem aynı en yüksek değeri veriyorsa bunlardan herhangi biri seçilebilir.

### 5.3 Aksiyomlar

von Neumann–Morgenstern teoremi, karar vericinin piyangolar üzerindeki tercihleri ($\succ$: tercih edilir, $\sim$: kayıtsız kalınır, $\succeq$: tercih edilir ya da kayıtsız kalınır) aşağıdaki dört aksiyomu sağlıyorsa, bu tercihleri beklenen faydayla temsil eden bir fayda fonksiyonunun var olduğunu gösterir:

1. **Tamlık (completeness):** Herhangi iki piyango $L$ ve $M$ için $L \succeq M$ ya da $M \succeq L$'dir (ya da ikisi birden). Karar verici her iki seçeneği karşılaştırabilir.
2. **Geçişlilik (transitivity):** $L \succeq M$ ve $M \succeq N$ ise $L \succeq N$'dir.
3. **Süreklilik (continuity):** $L \succeq M \succeq N$ ise $pL + (1-p)N \sim M$ olacak biçimde bir $p \in [0, 1]$ olasılığı vardır. Ortadaki seçenek, en iyi ve en kötü seçeneklerin uygun bir karışımına eşdeğerdir.
4. **Bağımsızlık (independence):** $L \succ M$ ise, herhangi bir $N$ piyangosu ve $p \in (0, 1]$ için $pL + (1-p)N \succ pM + (1-p)N$'dir. İki seçenek **aynı** üçüncü seçenekle **aynı oranda** karıştırıldığında tercih sırası değişmez.

---

## 6. Karar Sürecinin Bileşenleri: Şemsiye Örneği

Beklenen fayda teorisi, eylemleri ne kadar tercih edilir olduklarına göre sıralamanın bir yolunu sağlar: Beklenen faydası yüksek olan eylem daha çok tercih edilir.

Uzun bir yürüyüş planlandığını ve şemsiyenin alınıp alınmayacağına karar verilmesi gerektiğini düşünelim. Güneşli bir günde şemsiye taşımamak tercih edilir; ancak şemsiyesiz yağmura yakalanmak hiç istenmez. Bu karar problemi üç bileşene ayrılır (Steele ve Stefánsson, 2020):

1. **Eylemler (acts):** Karar vericinin seçebileceği seçenekler. Örnekte iki eylem vardır: şemsiyeyi almak ya da evde bırakmak.
2. **Durumlar (states):** Sonucu etkileyen ve karar vericinin denetimi dışındaki koşullar. Örnekte iki durum vardır: yağmur yağar ya da yağmaz.
3. **Sonuçlar (outcomes):** Bir eylemin bir durumda ortaya çıkardığı, karar vericinin kendisi için değer verdiği sonuçlar. Örnekte üç sonuç vardır: şemsiye taşımadan kuru kalmak, şemsiye taşıyarak kuru kalmak ve şemsiyesiz ıslanmak.

Bu çerçevede eylemler, durumlar ve sonuçlar, dünyanın nasıl olabileceğine ilişkin **olanakların** (possibilities) kümeleri olarak düşünülür. Tüm olanakları kapsayan en geniş küme $\Omega$ ile gösterilir; eylemler, durumlar ve sonuçlar $\Omega$'nın birer **bölüntüsünü** (partition) oluşturur. Yani her olanak tam olarak bir duruma, bir eyleme ve bir sonuca karşılık gelir: Karar verici tek bir eylem seçer, dünya tek bir durumda bulunur ve tek bir sonuç gerçekleşir (eve ya ıslak ya kuru dönülür). Ayrıca eylemler ile durumlar birbirinden bağımsız kabul edilir: Şemsiye almak yağmurun yağma olasılığını değiştirmez.

Problem, satırları eylemlere, sütunları durumlara karşılık gelen bir **karar matrisiyle** gösterilir; her hücre, ilgili eylem ilgili durumda seçildiğinde ortaya çıkan sonuçtur. Sonuçlara örnek olarak şu faydalar atanmış olsun:

| Eylem | Yağmur yağar | Yağmur yağmaz |
| :--- | :--- | :--- |
| Şemsiyeyi almak | Kuru, şemsiye taşındı: $U = 7$ | Kuru, şemsiye taşındı: $U = 7$ |
| Şemsiyeyi almamak | Islandı: $U = 0$ | Kuru, şemsiyesiz: $U = 10$ |

Yağmur olasılığı $p$ ile gösterilirse:

$$EU(\text{al}) = p \cdot 7 + (1-p) \cdot 7 = 7$$

$$EU(\text{alma}) = p \cdot 0 + (1-p) \cdot 10 = 10 - 10p$$

Hava durumu sunucusu yağmur olasılığını $p = 0.4$ olarak açıklamış olsun. Bu durumda $EU(\text{alma}) = 10 - 4 = 6 < 7 = EU(\text{al})$ olduğundan şemsiyeyi almak doğru karardır.

İki eylemin eşit beklenen fayda verdiği **eşik olasılık** $10 - 10p^{*} = 7$ eşitliğinden $p^{*} = 0.3$ bulunur. Yağmur olasılığı %30'un üzerindeyse şemsiye alınmalı, altındaysa evde bırakılmalıdır. Eşik, kişiden kişiye değişir: Islanmayı çok daha kötü bulan biri için ($U(\text{ıslandı})$ daha düşük) eşik daha küçük olur ve bu kişi şemsiyeyi daha sık alır. Beklenen fayda teorisi, aynı olasılık bilgisinin farklı tercihlere sahip kişileri neden farklı kararlara götürdüğünü bu biçimde açıklar.

---

## 7. Uygulamalı Örnek: Yatırım Kararı ve Riskten Kaçınma

Bir yatırımcı iki seçenek arasında karar vermektedir:

- **Yatırım A:** %50 olasılıkla 100 birim kâr, %50 olasılıkla 0 kâr.
- **Yatırım B:** Kesin olarak (%100 olasılıkla) 45 birim kâr.

**Beklenen değer (riskten bağımsız yatırımcı).** Fayda fonksiyonu olarak doğrudan kâr kullanılırsa, $U(x) = x$:

$$EU(A) = 0.5 \times 100 + 0.5 \times 0 = 50, \qquad EU(B) = 1 \times 45 = 45$$

Bu yatırımcı A'yı seçer. Doğrusal fayda fonksiyonunda beklenen fayda beklenen değere eşittir; böyle bir yatırımcıya **riskten bağımsız** (risk-neutral) denir.

**Azalan marjinal fayda (riskten kaçınan yatırımcı).** Bernoulli'nin gözlemine uygun olarak fayda, kârla birlikte azalan oranda artsın: $U(x) = \sqrt{x}$.

$$EU(A) = 0.5 \times \sqrt{100} + 0.5 \times \sqrt{0} = 0.5 \times 10 = 5$$

$$EU(B) = \sqrt{45} \approx 6.71$$

Bu kez B seçilir. A'nın beklenen değeri daha yüksek olduğu hâlde, kesin kazancın sağladığı fayda riskli seçeneğinkinden fazladır. İçbükey (concave) fayda fonksiyonuna sahip böyle bir karar vericiye **riskten kaçınan** (risk-averse) denir.

**Kesinlik eşdeğeri ve risk primi.** A seçeneğinin kesinlik eşdeğeri, faydası A'nın beklenen faydasına eşit olan kesin tutardır: $\sqrt{CE} = 5 \Rightarrow CE = 25$. Yani bu yatırımcı için A'yı elinde tutmak ile kesin 25 birim almak arasında fark yoktur. A'nın beklenen değeri (50) ile kesinlik eşdeğeri (25) arasındaki fark, **risk primidir** (risk premium): $50 - 25 = 25$ birim. Yatırımcı, belirsizlikten kurtulmak için beklenen kazancından 25 birim vazgeçmeye hazırdır. Sigorta satın almak da aynı mantıkla açıklanır.

---

## 8. Eleştiriler ve Alternatifler

Beklenen fayda teorisi rasyonel bir çerçeve sunsa da insan davranışı her zaman bu modele uymaz.

**Allais paradoksu.** Maurice Allais (1953) şu iki seçim problemini önermiştir:

| Seçim 1 | Sonuçlar |
| :--- | :--- |
| A | %100 olasılıkla 1 milyon TL |
| B | %10 olasılıkla 5 milyon TL, %89 olasılıkla 1 milyon TL, %1 olasılıkla 0 TL |

| Seçim 2 | Sonuçlar |
| :--- | :--- |
| C | %11 olasılıkla 1 milyon TL, %89 olasılıkla 0 TL |
| D | %10 olasılıkla 5 milyon TL, %90 olasılıkla 0 TL |

Deneylerde insanların çoğu birinci problemde A'yı, ikinci problemde D'yi seçer. Oysa bu iki seçim beklenen fayda teorisiyle bağdaşmaz. $U(0) = 0$ alınırsa:

- A ≻ B ise: $U(1\text{M}) > 0.10\, U(5\text{M}) + 0.89\, U(1\text{M})$, yani $0.11\, U(1\text{M}) > 0.10\, U(5\text{M})$.
- D ≻ C ise: $0.10\, U(5\text{M}) > 0.11\, U(1\text{M})$.

İki eşitsizlik birbirinin tersidir; hiçbir fayda fonksiyonu ikisini birlikte sağlayamaz. İki problem arasındaki tek fark, her iki seçeneğe ortak olan "%89 olasılıkla 1 milyon TL" sonucunun "%89 olasılıkla 0 TL" ile değiştirilmesidir; bağımsızlık aksiyomuna göre bu değişiklik tercihi etkilememelidir. İnsanlar ise kesinliğe (birinci problemdeki A seçeneğine) olasılık oranının ötesinde bir değer vermektedir.

**Beklenti teorisi (prospect theory).** Kahneman ve Tversky (1979), bu tür sapmaları açıklamak için beklenti teorisini önermiştir. Bu teoriye göre bireyler sonuçları bir referans noktasına göre kazanç ya da kayıp olarak değerlendirir; kayıplar eşit büyüklükteki kazançlardan daha güçlü hissedilir (kayıptan kaçınma, loss aversion) ve küçük olasılıklara olduklarından daha fazla ağırlık verilir. Beklenti teorisi, insanların gerçekte nasıl karar verdiğini betimlemede beklenen fayda teorisinden daha başarılıdır; beklenen fayda teorisi ise rasyonel kararın nasıl olması gerektiğine ilişkin normatif ölçüt olarak önemini korumaktadır.

---

## Kaynaklar

- Bernoulli, D. (1738/1954). Exposition of a new theory on the measurement of risk. *Econometrica*, 22(1), 23–36.
- von Neumann, J. ve Morgenstern, O. (1944). *Theory of Games and Economic Behavior*. Princeton University Press.
- Savage, L. J. (1954). *The Foundations of Statistics*. Wiley.
- Allais, M. (1953). Le comportement de l'homme rationnel devant le risque. *Econometrica*, 21(4), 503–546.
- Kahneman, D. ve Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2), 263–291.
- Steele, K. ve Stefánsson, H. O. (2020). Decision theory. *Stanford Encyclopedia of Philosophy*. https://plato.stanford.edu/entries/decision-theory/
- Russell, S. ve Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4. baskı), Bölüm 16. Pearson.
