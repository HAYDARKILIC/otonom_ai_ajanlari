# Kıyaslamalar

Bu belge, depodaki bileşenler için temel erişim ve yeniden sıralama
metriklerini kaydeder. Ölçüm aracı `scripts/benchmark_retrieval.py`'dir.

> **Önemli not:** Aşağıdaki tablo, deponun orijinal yazarının kendi
> ortamında ölçtüğü değerlerdir. O ölçümde kullanılan veri dilimi depoyla
> birlikte yayımlanmamıştı. `tests/fixtures/beir_toy/` altındaki 50 sorguluk
> veri kümesi bu sürümde **sonradan elle hazırlanmıştır** (bilgi erişimi ve
> NLP konularında 50 kısa belge ve her biri için başka sözcüklerle ifade
> edilmiş bir sorgu). Bu yüzden tablodaki sayılar bu veriyle birebir
> yeniden üretilemeyebilir; kendi donanımınızda şöyle ölçün:
>
> ```bash
> python scripts/benchmark_retrieval.py \
>     --corpus tests/fixtures/beir_toy/corpus.jsonl \
>     --queries tests/fixtures/beir_toy/queries.jsonl --k 10
> ```
>
> Betik yalnızca ilk satırı (Bi-Encoder) ölçer; HyDE ve Cross-Encoder
> satırları için `src/reranking/` bileşenlerini hatta ekleyin.

| Aşama                                        | Recall@10 | nDCG@10 | Gecikme (ms / sorgu) |
|----------------------------------------------|-----------|---------|----------------------|
| Yalnızca Bi-Encoder (bge-small-en-v1.5)      |    0.71   |   0.58  |          8           |
| Bi-Encoder + HyDE                            |    0.78   |   0.63  |        ~480 (LLM)    |
| Bi-Encoder + Cross-Encoder rerank (ilk 50)   |    0.71   |   0.74  |         95           |
| Bi-Encoder + HyDE + Cross-Encoder rerank     |    0.78   |   0.79  |       ~575           |

**Donanım.** Tüm ölçümler, bge-small MPS üzerinde ve cross-encoder CPU
üzerinde çalışırken tek bir Apple M3 Pro ile alınmıştır. Sayılar yalnızca
niteliksel eğilimleri göstermek için raporlanmıştır; HyDE
yapılandırmalarındaki baskın maliyet gömme adımı değil, LLM çağrısıdır.

**Uyarılar.**
- Oyuncak dilimde 50 sorgu vardır; yayımlanmış BEIR sayıları binlerce sorgu
  üzerindendir. Mutlak değerleri gösterge niteliğinde kabul edin.
- HyDE, LLM'e ve sıcaklığa bağlıdır; yukarıdaki sayılar $K = 1$ hipotezle
  $T = 0.7$ sıcaklığında Claude Opus kullanır.
- Cross-Encoder kazançları, Recall@k yüksek ancak sıralama zayıf olduğunda
  en büyüktür — bu da yeniden sıralamanın en faydalı olduğu rejimdir.
