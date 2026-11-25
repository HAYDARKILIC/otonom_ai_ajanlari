# Matematiksel Temeller

Bu belge, not defterlerine ve modül docstring'lerine dağılmış matematiksel
içeriği tek bir başvuru kaynağında toplar.

---

## 1. Vektör benzerliği geometrisi

Birim normlu $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ vektörleri için,

$$
\cos(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top \mathbf{v},
\qquad
\|\mathbf{u} - \mathbf{v}\|_2^2 = 2 - 2\mathbf{u}^\top \mathbf{v}.
$$

Dolayısıyla kosinüs benzerliği, iç çarpım ve karesel Öklid uzaklığı birim
küre üzerinde *monoton olarak eşdeğerdir* ve üçünden herhangi birinin
ürettiği sıralama aynıdır. Kodlayıcımız, arka uç sınırlarındaki metrik
uyumsuzluğu hatalarını ortadan kaldırmak için çıktıları tam da bu nedenle
birim küreye normalize eder.

---

## 2. HNSW seviye ataması

Bir $x$ noktası şu en üst katmana eklenir:

$$
\ell(x) = \left\lfloor -\ln(u) \cdot m_L \right\rfloor,
\qquad u \sim U(0,1),
\qquad m_L = \frac{1}{\ln M}.
$$

Ortaya çıkan katman doluluk dağılımı $1 - 1/M$ parametreli geometrik bir
dağılımdır ve beklenen katman boyutu şudur:

$$
\mathbb{E}\bigl[\,|\text{katman } \ell|\,\bigr] = \frac{N}{M^\ell}.
$$

Bu nedenle en üst katmandan başlayan açgözlü iniş, en alt katmana ulaşmadan
önce beklenen olarak $O(\log_M N)$ noktadan geçer; bu da düzgün dağılımlı
veri için $O(\log N)$ arama maliyeti sağlar.

---

## 3. Bi-Encoder ve Cross-Encoder

**Bi-Encoder.** Sorgu ve belge bağımsız olarak gömülür:

$$
s_{\text{bi}}(q, d) = \phi(q)^\top \phi(d), \qquad \phi:\Sigma^\* \to \mathbb{R}^d.
$$

Belge gömmeleri $\phi(d)$ bir kez önceden hesaplanıp önbelleğe alınabilir;
bu da $N$ belge üzerinde erişimi sorgu başına $O(Nd)$ yapar.

**Cross-Encoder.** Çift birleştirilir ve birlikte işlenir:

$$
s_{\text{cross}}(q, d) = f_\theta\bigl([\text{CLS}]\,q\,[\text{SEP}]\,d\bigr) \in \mathbb{R}.
$$

Ortak dikkat (attention), ince taneli token düzeyindeki etkileşimleri
yakalar ancak önbelleğe almayı engeller: $N$ adayı yeniden sıralamak $N$
Transformer ileri geçişine mal olur.

`src/reranking/cross_encoder.py` içinde uygulanan **eğitim hedefleri**:

- *Noktasal BCE:*

  $$
  \mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_i \bigl(y_i\log\sigma(s_i) + (1-y_i)\log(1-\sigma(s_i))\bigr).
  $$

- *İkili marj* (menteşe / hinge):

  $$
  \mathcal{L}_{\text{margin}} = \frac{1}{N}\sum_i \max\bigl(0, m - (s_i^+ - s_i^-)\bigr).
  $$

---

## 4. HyDE

Bir $q$ sorgusu için $\mathrm{LLM}(q)$, halüsinasyonla üretilmiş bir yanıt
paragrafını göstersin. HyDE erişim skorunu

$$
s(q, d) = \phi(q)^\top \phi(d)
\qquad\text{yerine}\qquad
s_{\text{HyDE}}(q, d) = \phi\bigl(\mathrm{LLM}(q)\bigr)^\top \phi(d)
$$

ile değiştirir.

$T > 0$ sıcaklığında örneklenen $K$ halüsinasyonla şu Monte-Carlo
tahminini kullanırız:

$$
\hat{\phi}_q = \frac{1}{K} \sum_{k=1}^K \phi\bigl(\mathrm{LLM}_k(q)\bigr).
$$

Ortalama almak, tek tek halüsinasyonların katkıda bulunduğu varyansı
azaltır. Ayrıca konu dışı bir halüsinasyonun geri çağırmayı (recall)
tamamen yok edememesi için $\phi(q)$'nun kendisini de ortalamaya dâhil
ederiz.

---

## 5. nDCG@k

İlgililik dereceleri $r_i \in \{0,1,2,\dots\}$ olan bir sıralama için,

$$
\text{DCG}@k = \sum_{i=1}^{k} \frac{2^{r_i} - 1}{\log_2(i + 1)},
\qquad
\text{nDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k},
$$

burada IDCG ideal olarak sıralanmış sıralamadan hesaplanır. Bu,
`scripts/benchmark_retrieval.py` tarafından raporlanan metriktir.

---

## 6. Çoklu ajan döngüsünün sonlanması

$t$ yinelemesindeki durum $s_t = (c_t, e_t)$ olsun; burada $c_t$ kod,
$e_t$ ise en son hatadır. Orkestratör, görülmüş durumların SHA-256
özetlerinden oluşan bir $\mathcal{H}$ kümesi tutar. Eğer
$\text{hash}(s_t) \in \mathcal{H}$ ise döngü durur. $\mathcal{H}$ sonlu
olduğundan ve monoton olarak büyüdüğünden — ve durum uzayı büyük olsa da
beklenti olarak LLM'in çıktı dağılımıyla sınırlı olduğundan — döngü
neredeyse kesin olarak $|\mathcal{H}|$ yineleme içinde sonlanır.
`max_iterations` bütçesi deterministik bir en kötü durum sınırı sağlar.
