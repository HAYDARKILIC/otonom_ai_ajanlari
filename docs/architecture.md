# Sistem Mimarisi

Bu belge, **Ajan Tabanlı Bağlam Mühendisliği** deposunu oluşturan altı
modülün mimarisini ve aralarındaki veri akışını açıklar.

## Üst düzey katmanlama

```
        ┌──────────────────────────────────────────────────────────┐
        │  Bitirme: Akademik Araştırma Asistanı  (Hafta 6)         │
        └────────┬────────────────────────────────────┬────────────┘
                 │                                    │
        ┌────────▼───────────┐                ┌───────▼───────────┐
        │ Çoklu Ajan Sistemi │                │  ReAct Ajanı      │
        │ (Hafta 5)          │                │  (Hafta 4)        │
        └────────┬───────────┘                └───────┬───────────┘
                 │                                    │
                 └──────────────┬─────────────────────┘
                                │
        ┌───────────────────────▼─────────────────────────────────┐
        │  Yeniden sıralama hattı (Hafta 3): sorgu yeniden yazma + │
        │  HyDE + Bi-Encoder → Cross-Encoder                       │
        └───────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────▼─────────────────────────────────┐
        │  Hiyerarşik RAG (Hafta 2): parçalayıcılar + kodlayıcı +  │
        │  vektör deposu (Chroma / Milvus)                         │
        └───────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────▼─────────────────────────────────┐
        │  Model Context Protocol katmanı (Hafta 1):               │
        │  JSON-RPC 2.0 aktarımı + araç kayıt defteri + kaynaklar  │
        └──────────────────────────────────────────────────────────┘
```

## Modül sorumlulukları

### `mcp_core/` — Hafta 1
Spesifikasyona uygun bir MCP sunucusu sağlar. Kararlar:

- **Stdio aktarımı** birincil arayüzdür. WebSocket ve HTTP aktarımları
  spesifikasyonda yer alır ancak pedagojik değer katmadan ciddi karmaşıklık
  eklediklerinden burada uygulanmamıştır.
- **Kablo üzerinde her yerde Pydantic.** Her JSON-RPC mesajı `BaseModel`
  üzerinden ayrıştırılıp yeniden serileştirilir; böylece bozuk trafik
  uygulama koduna ulaşamaz.
- **Yetenekler asgari düzeydedir.** Yalnızca `tools` ve `resources`
  duyurulur; `prompts` ve `sampling` protokolde yer alır ama uygulanmamıştır.

### `rag_engine/` — Hafta 2
Hiyerarşik ve cümle penceresi erişimini uygular. Kararlar:

- **Gömmeler kodlayıcıda L2-normalize edilir.** Bu, arka uç sınırlarındaki
  metrik uyumsuzluğu hatalarını (kosinüs - nokta çarpımı - Öklid) ortadan
  kaldırır.
- **Arka uçlar takılıp çıkarılabilir.** Varsayılan `ChromaStore`'dur;
  ölçek için yerine `MilvusStore` takılabilir.
- **Ebeveyn metinleri kalıcıdır.** Yalnızca çocuk parçalar gömülür; ebeveyn
  metinleri `SQLiteParentStore` ile vektör deposunun yanında saklanır. Böylece
  indeksi bir süreçte kurup başka bir süreçte sorgulamak mümkündür.
- **Parça kimlikleri deterministiktir** (`{belge}::p{i}::c{j}`). Aynı belgeyi
  yeniden indekslemek kayıtları çoğaltmaz, üzerine yazar.
- **HNSW uygulaması üretim için değil, öğretim içindir.** Üretim yolu
  Chroma'nın HNSW'sidir; `hnsw_native.py`, not defterinin seviye atama
  matematiğini ve açgözlü arama algoritmasını adım adım gösterebilmesi için
  vardır.

### `reranking/` — Hafta 3
Sorgu yeniden yazma, HyDE ve Cross-Encoder yeniden sıralamasını uygular.

- **Yeniden sıralayıcı bir Protocol'dür.** `CrossEncoderReranker` ve
  `CohereReranker` aynı arayüzü karşılar; hat hangisinin kullanıldığını
  bilmez.
- **Eğitim yardımcıları dışa açıktır ancak üretimde kullanılmaz.** Noktasal
  BCE ve ikili marj kayıpları, matematiğin adım adım anlatımı için Hafta 3
  not defterinde içe aktarılır.

### `react_agent/` — Hafta 4
Saf Python ile yazılmış bir ReAct ajanı. Temel tercihler:

- **Yerel fonksiyon çağırma değil, metin protokolü.** Anthropic ve OpenAI
  yapılandırılmış araç çağrılarını desteklese de burada metin protokolü
  kullanılır; çünkü bu, ayrıştırıcıyı görünür kılar ve uygulamayı bozuk LLM
  çıktısını zarifçe ele almaya zorlar — ki pratikte bunu yapmak zorundadır.
- **Ayrıştırıcı hoşgörülüdür.** Gevşek JSON'u kabul eder, kod çitlerini
  (code fence) temizler ve ajanın kendi kendini düzeltmesi için LLM'e geri
  enjekte ettiği yapılandırılmış hatalar üretir.

### `multi_agent/` — Hafta 5
Kodlayıcı–Yürütücü–Eleştirmen topolojisi. Temel tercihler:

- **Döngü tespiti tur sayımı değil, içerik özeti (hash) kullanır.**
  Kodlayıcı art arda iki kez aynı hatayla aynı kodu üretirse sistem
  tıkanmıştır; orkestratör deterministik olarak durur.
- **Yürütücü, zaman aşımı olan bir alt süreçte çalışır.** Gerçek yalıtım
  (konteynerler) bir dağıtım meselesidir; arayüz buna uyum sağlayacak
  şekilde tasarlanmıştır.

### `academic_assistant/` — Hafta 6
Bitirme projesi. Diğer tüm modülleri birleştirir: veri alma → MCP destekli
SQLite deposu → hiyerarşik RAG → cross-encoder ile yeniden sıralama → ana
hat güdümlü LLM sentezi → Markdown literatür taraması.

## Veri akışı: uçtan uca sorgu

```
kullanıcı sorgusu
  └──▶ QueryRewriter.rewrite()                # isteğe bağlı; LLM çağrısı
        └──▶ HierarchicalRAG.retrieve(k=50)
              ├── sorguyu kodla
              ├── child_store.query(k=50)
              └── ebeveyn genişletme
        └──▶ CrossEncoderReranker.rerank(k=5)
              └── (q,d) çiftleri üzerinde Transformer ileri geçişi
        └──▶ LLM sentezi
              └── nihai yanıt / rapor bölümü
```

## Hata türleri ve önlemler

| Hata türü | Önlem |
|--------------|-----------|
| ReAct döngüsünde bozuk LLM yanıtı | `ParseError`; düzeltici mesaj enjeksiyonu; sınırlı yeniden deneme bütçesi |
| ReAct döngüsünde araç istisnası | İstisna `Observation: ERROR ...` biçimine dönüştürülür; LLM toparlanabilir |
| Sonsuz ReAct döngüsü | Katı `max_steps` bütçesi |
| İki ajan arasında salınım (Kodlayıcı/Eleştirmen) | Durum özeti tabanlı döngü tespiti |
| Vektör arka ucu metrik uyumsuzluğu | Kodlayıcıda L2-normalize; her yerde kosinüs |
| Sentezde alıntı ile yeniden ifade arasında kayma | İstem uydurma atıfları yasaklar; ana hat → bölüm → kaynakça yapısı LLM'i sınırlar |
| MCP dosya sisteminde sembolik bağ ile korumalı alandan kaçış | `realpath()` üzerinde `os.path.commonpath` denetimi |
| Hesap makinesi aracında kod çalıştırma (`eval` kaçışı) veya devasa üslerle DoS | `eval` yok; AST beyaz listesi, üs/sonuç/ifade uzunluğu sınırları |
| `read_file` ile korumalı alan dışındaki dosyaların okunması | Araç bir kök dizine bağlanır; `../`, mutlak yol ve sembolik bağ reddedilir |
| Log satırlarının MCP stdout kanalını bozması | Tüm loglar stderr'e yazılır |
| Araç hatasının protokol hatası gibi görünmesi | Araç istisnaları `isError: true` sonucu olarak döner; LLM görüp toparlanabilir |
| Opus 4.7+ modellerinde `temperature` kaynaklı 400 hatası | İstemci parametreyi ilk 400 hatasında düşürüp yeniden dener ve bunu hatırlar |
| İndeksin yeniden oluşturulmasında kayıtların çoğalması | Deterministik parça kimlikleri + upsert |
| Ayrı süreçte sorgulanan indekste ebeveyn genişletmesinin kaybolması | Kalıcı `SQLiteParentStore`; cümle penceresi komşuları depodan kimlikle getirilir |
| Hata izindeki rastgele geçici yollar yüzünden döngü tespitinin hiç tetiklenmemesi | Yürütücü çıktısındaki geçici dizin yolları normalize edilir |
