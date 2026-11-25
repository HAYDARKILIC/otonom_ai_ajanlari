# BEIR oyuncak dilimi

`scripts/benchmark_retrieval.py` için 50 sorguluk bir test verisi. Tam ölçekli
değerlendirme için resmî BEIR sürümüyle değiştirin. Biçim:

- `corpus.jsonl`  — {"id": "...", "text": "..."}
- `queries.jsonl` — {"qid": "...", "query": "...", "relevant": ["id1", "id2"]}

Bu sürümdeki veri elle hazırlanmıştır: bilgi erişimi, NLP ve ajan sistemleri
konularında 50 kısa İngilizce belge ve her biri için tam olarak bir ilgili
belgeye sahip, başka sözcüklerle ifade edilmiş bir sorgu. Gömme modeli İngilizce
olduğu için veri de İngilizcedir. `tests/test_end_to_end.py` kıyaslama betiğini
bu veriyle uçtan uca çalıştırır.
