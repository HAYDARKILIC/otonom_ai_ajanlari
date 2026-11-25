<div align="center">

# Otonom Yapay Zekâ Ajanları ve Bağlam Mühendisliği

**Model Context Protocol sunucuları, gelişmiş RAG hatları ve otonom çoklu ajan sistemleri — üst düzey ajan çatıları kullanılmadan, temel ilkelerden yola çıkılarak uygulanmıştır.**

</div>

---

## Genel Bakış

Bu depo, otonom yapay zekâ ajanlarının yapı taşlarını — Model Context Protocol, hiyerarşik erişim, yeniden sıralama (re-ranking), ReAct döngüsü ve çoklu ajan orkestrasyonu — yaklaşık 3.000 satırlık açıklamalı Python koduyla **sıfırdan** uygular. LangChain yok, LlamaIndex yok, CrewAI yok, AutoGen yok.

Amaç kolaylık değil derinliktir: her protokol el sıkışması, gömme (embedding) geometrisi, yeniden sıralama kaybı ve akıl yürütme döngüsü, bunları gizleyen bir çatıdan içe aktarılmak yerine açıkça yazılmış ve açıklanmıştır. Kod altı haftalık bir müfredat olarak düzenlenmiştir ve dağıtıma hazır bir sistem olarak sunulur — tip tanımlı, test edilmiş ve üretim ortamı için yapılandırılmış.

---

## Müfredat

| Hafta | Konu | Teori | Uygulama |
|:----:|-------|--------|----------|
| **1** | **Model Context Protocol — Kurulum ve Mimari** | Modellerin dış dünyayla konuşma standartları. MCP mimarisi: İstemci, Sunucu ve Araçlar arasındaki protokol. | Yerel bir dosya sistemini ve özel bir veritabanını bir LLM istemcisine (ör. Claude Desktop) güvenli biçimde açan özel bir MCP sunucusu. |
| **2** | **Gelişmiş RAG — İndeksleme ve Vektör Optimizasyonu** | Temel RAG'in sınırları. Hiyerarşik indeksleme (ebeveyn-çocuk belgeler), cümle penceresi erişimi ve vektör uzayları (Chroma, Pinecone, Milvus). | Büyük bir akademik makale derlemini hiyerarşik olarak indeksleyip bir vektör veritabanında kalıcı hâle getiren bir RAG hattı. |
| **3** | **Gelişmiş RAG — Yeniden Sıralama ve Sorgu Dönüşümü** | Sorgu yeniden yazma, Varsayımsal Belge Gömmeleri (HyDE). Bi-Encoder ve Cross-Encoder karşılaştırması: matematiksel maliyet/performans dengesi. | Arama sonuçlarını bir Cohere Re-ranker veya açık kaynaklı bir Cross-Encoder ile optimize eden bir filtreleme katmanı. |
| **4** | **ReAct Döngüsü ve Araç Kullanımı** | Ajanın akıl yürüt-ve-eyle döngüsü. Düşünce (Thought), Eylem (Action) ve Gözlem (Observation) adımlarının istem (prompt) ve işlev düzeyinde kurgulanması. LLM JSON / fonksiyon çağırma mekanizmaları. | Hiçbir harici ajan kütüphanesi (LangChain, CrewAI vb.) kullanmadan saf Python'daki matematiksel araçları kullanan bir ReAct ajanı. |
| **5** | **Çoklu Ajan Sistemleri ve Durum Yönetimi** | Ajanlar arasında iş bölümü: yönetici, uzman ve eleştirmen modeller. Durum takibi, sonsuz döngünün önlenmesi ve bellek yönetimi. | "Kendini düzelten bir kod ajanı": bir ajan kod yazar, bir diğeri onu test edip hataları bildirir, ilki de bunları düzeltir. |
| **6** | **Bitirme Projesi — Akademik Asistan Ajanı** | Önceki tüm bileşenlerin tek bir otonom sistemde birleştirilmesi. | arXiv API'sinden makaleleri çeken, MCP aracılığıyla yerel olarak saklayan, RAG ile analiz eden ve bir Markdown raporu sunan otonom bir ajan. |

Her haftanın, matematiksel modelleri ve çalıştırılabilir bir uygulamayı kapsayan ayrı bir not defteri [`notebooks/`](notebooks/) klasöründe bulunur.

---

## Temel Özellikler

- **Hiçbir üst düzey ajan bağımlılığı yok.** ReAct döngüsü, araç kayıt defteri, mesaj yönlendirme ve çoklu ajan durum makinesinin tamamı elle yazılmıştır. Kontrol akışını gizleyen bir çatı yoktur.
- **Kablo düzeyinden itibaren spesifikasyona uygun bir MCP sunucusu.** JSON-RPC 2.0 çerçeveleme, eksiksiz `initialize` / `tools/list` / `tools/call` / `resources/read` yaşam döngüsü ve yetenek müzakeresi — sarmalanmamış, uygulanmış ve test edilmiştir.
- **Matematik kodun içinde yaşar.** Seviye atama olasılığı `1/ln(M)` ve açgözlü (greedy) aramayı açıkça gösteren sıfırdan bir HNSW indeksi; değiştirilebilir kayıplar olarak cross-encoder eğitim hedefleri (noktasal BCE ve ikili marj); Monte-Carlo gömme tahmini olarak HyDE.
- **Birleştirilebilir erişim temel öğeleri.** Hiyerarşik ebeveyn-çocuk parçalama ve cümle penceresi erişimi, ortak bir arayüzün arkasında bağımsız stratejilerdir.
- **Takılıp çıkarılabilir arka uçlar.** Vektör depoları (Chroma / Milvus) ve yeniden sıralayıcılar (açık kaynak Cross-Encoder / Cohere) koddan değil, yapılandırmadan değiştirilir.
- **Sonlanması kanıtlanabilir çoklu ajan döngüsü.** Kodlayıcı–Yürütücü–Eleştirmen sistemi sınırlı bir yeniden deneme bütçesi ve durum özeti (hash) tabanlı bir döngü dedektörü uygulayarak durmayı garanti eder.
- **Güvenli araçlar.** LLM'e açık araçlar güvenilmez girdi varsayımıyla yazılmıştır: hesap makinesi `eval` yerine AST tabanlı bir değerlendirici kullanır, `read_file` ve MCP dosya kaynakları bir korumalı alanın dışına çıkamaz, SQLite kaynakları yalnızca var olan tabloları okur.
- **Üretim hijyeni.** Pydantic ile doğrulanan JSON-RPC mesajları, stderr'e yazan yapılandırılmış loglama (stdout MCP protokolüne ayrılmıştır), %85 kapsam eşiğine sahip pytest, ruff + mypy, pre-commit kancaları ve bir GitHub Actions CI hattı.

---

## Depo Yapısı

```
otonom_ai_ajanlari/
├── src/
│   ├── mcp_core/            # Hafta 1 — MCP sunucusu, JSON-RPC aktarımı, araç kayıt defteri
│   ├── rag_engine/          # Hafta 2 — parçalayıcılar, hiyerarşik indeksleyici, HNSW, vektör depoları
│   ├── reranking/           # Hafta 3 — sorgu yeniden yazma, HyDE, bi/cross-encoder
│   ├── react_agent/         # Hafta 4 — ReAct döngüsü, araç kayıt defteri, ayrıştırıcı
│   ├── multi_agent/         # Hafta 5 — orkestratör, ajanlar, durum, bellek
│   ├── academic_assistant/  # Hafta 6 — bitirme: arXiv → MCP → RAG → ajan
│   └── utils/               # loglama, yapılandırma, LLM istemci soyutlaması
├── notebooks/               # Altı haftalık not defteri (teori + çalıştırılabilir demolar)
├── tests/                   # pytest test paketi (birim, uçtan uca, güvenlik, regresyon)
│   └── fixtures/beir_toy/   # kıyaslama için 50 sorguluk oyuncak veri kümesi
├── docs/                    # mimari, matematiksel temeller, kıyaslamalar
├── scripts/                 # veri alma, indeksleme ve değerlendirme betikleri
├── configs/                 # ortam değişkeni enterpolasyonlu YAML yapılandırmaları
├── data/                    # ham / işlenmiş / vektör deposu (veriler gitignore'da)
├── .github/workflows/ci.yml # lint + tip denetimi + testler (Python 3.10 ve 3.12)
└── pyproject.toml           # paket tanımı, isteğe bağlı bağımlılık grupları, araç ayarları
```

---

## Kurulum

```bash
git clone https://github.com/HAYDARKILIC/otonom_ai_ajanlari.git
cd otonom_ai_ajanlari

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt     # her şey (torch ve gömme modelleri dâhil)
pip install -e .

# Geliştirme için (testler, lint, not defterleri)
pip install -r requirements-dev.txt
pre-commit install
```

Yalnızca ihtiyacınız olanı kurmak isterseniz `pyproject.toml` içindeki gruplar kullanılabilir:

```bash
pip install -e .                      # çekirdek: MCP sunucusu, ReAct ajanı, çoklu ajan sistemi
pip install -e ".[ml,vector]"         # + gömme modeli, cross-encoder ve Chroma (Hafta 2, 3, 6)
pip install -e ".[dev,vector]"        # testler için (torch gerektirmez)
pip install -e ".[all]"               # hepsi
```

`.env.example` dosyasını `.env` olarak kopyalayın ve kimlik bilgilerinizi girin (`src` paketi içe aktarıldığında `.env` otomatik olarak yüklenir):

```env
LLM_PROVIDER=anthropic             # anthropic | openai
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...                 # isteğe bağlı — yalnızca Cohere Rerank için
CHROMA_PERSIST_DIR=./data/vector_store/chroma
```

---

## Kullanım

**Hafta 1 — özel MCP sunucusunu başlatın**

```bash
python -m src.mcp_core --config configs/mcp_server.yaml
# veya kurulumdan sonra: otonom-mcp --config configs/mcp_server.yaml
```

Yapılandırmadaki `sandbox_root` ve `db_path` yolları yoksa oluşturulur. Etkinleştirilebilir araçlar: `calculator`, `search_arxiv` ve (yalnızca `sandbox_root` içini okuyabilen) `read_file`.

Sunucuyu Claude Desktop'a bağlamak için `claude_desktop_config.json` dosyasına şunu ekleyin (yolları kendi sisteminize göre düzenleyin):

```json
{
  "mcpServers": {
    "otonom-ai-ajanlari": {
      "command": "/tam/yol/otonom_ai_ajanlari/.venv/bin/python",
      "args": ["-m", "src.mcp_core", "--config", "configs/mcp_server.yaml"],
      "cwd": "/tam/yol/otonom_ai_ajanlari"
    }
  }
}
```

**Hafta 2 — hiyerarşik bir indeks oluşturun**

```bash
python scripts/ingest_arxiv.py --query "retrieval augmented generation" --max-results 200 \
    --output data/processed/arxiv.jsonl
python scripts/build_index.py --input data/processed/arxiv.jsonl --strategy hierarchical --backend chroma
```

İndeks `data/vector_store/chroma` altına, ebeveyn metinleri de yanındaki `arxiv_rag_parents.sqlite` dosyasına yazılır. Betik aynı girdiyle yeniden çalıştırıldığında kayıtlar çoğalmaz, üzerine yazılır. Bu varsayılanlar `configs/rag_arxiv.yaml` ile aynıdır; böylece Hafta 3'teki `from_config` aynı indeksi bulur.

**Hafta 3 — yeniden sıralama ile erişim yapın**

```python
from src.rag_engine.pipeline import HierarchicalRAG
from src.reranking.pipeline import RerankingPipeline

rag = HierarchicalRAG.from_config("configs/rag_arxiv.yaml")
reranker = RerankingPipeline.from_config("configs/reranker_cross_encoder.yaml")

# Gömme modeli ve derlem İngilizce olduğundan sorgular da İngilizce yazılmalıdır.
query = "How does HyDE compare to query rewriting?"
candidates = rag.retrieve(query, k=50)
top = reranker.rerank(query, candidates, k=5)
```

**Hafta 4 — ReAct ajanını çalıştırın**

```python
from src.react_agent.agent import ReActAgent
from src.react_agent.tools import calculator, search_arxiv
from src.utils.llm_client import build_client

llm = build_client("anthropic", "claude-opus-5-5")
agent = ReActAgent(llm=llm, tools=[calculator, search_arxiv], max_steps=10)
print(agent.run("HyDE üzerine 2024'ten üç makale bul ve özetle.").answer)
```

**Hafta 5 — kendini düzelten çoklu ajan sistemini çalıştırın**

```python
from src.multi_agent.orchestrator import SelfCorrectingCoder

# LLM_PROVIDER ortam değişkenine göre istemci kurulur; isterseniz llm=... ile kendiniz verin.
result = SelfCorrectingCoder(max_iterations=5).solve(
    "Merge sort'u uygula ve 1000 elemanlı karıştırılmış bir liste üzerinde doğrula."
)
print(result.final_code, result.iterations)
```

**Hafta 6 — bitirme projesi Akademik Asistan'ı çalıştırın**

```bash
python -m src.academic_assistant.cli \
    --topic "self-rewarding language models" --max-papers 25 \
    --output reports/review.md
```

---

## Test

```bash
pytest --cov=src --cov-report=term-missing   # kapsam %85'in altına düşerse başarısız olur
ruff check src/ scripts/ tests/
mypy src/
```

Testler hermetiktir: gerçek LLM yerine senaryolu bir `FakeLLM`, gömme modeli yerine deterministik bir karma (hashing) kodlayıcı kullanır; API anahtarı, model indirmesi veya ağ erişimi gerektirmez. Uçtan uca testler gerçek bir Chroma deposuyla çalışır (`chromadb` kurulu değilse atlanır). LLM istemcileri gerçek `anthropic` / `openai` SDK'larıyla, HTTP katmanı taklit edilerek test edilir.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) her push ve pull request'te Python 3.10 ve 3.12 üzerinde lint, tip denetimi ve testleri çalıştırır.

### LLM modelleri hakkında not

Varsayılan Anthropic modeli `claude-opus-5-5`'tir. Claude Opus 4.7 ve sonraki modeller `temperature` parametresini desteklemez; istemci bunu ilk 400 hatasında algılar ve parametre olmadan yeniden dener. Bu nedenle bu modellerde HyDE'nin birden çok hipotezi daha az çeşitli olabilir.

---

## Kaynak Materyal

- **Üretken Yapay Zekâ Entegrasyonu** — güncel teknik makaleler ve dokümantasyon setleri. MCP ve ajanlar hızla geliştiği için, tek bir sabit metin yerine her modülde güncel kaynaklara atıf yapılmıştır.
- **Natural Language Processing with Transformers** — Lewis Tunstall, Leandro von Werra ve Thomas Wolf (O'Reilly, 2022).

Modüllerde atıf yapılan temel makaleler: Yao vd. (ReAct, 2023); Gao vd. (HyDE, 2022); Malkov ve Yashunin (HNSW, 2018); Reimers ve Gurevych (Sentence-BERT, 2019); Anthropic (Model Context Protocol Spesifikasyonu).

---

## Lisans

MIT — bkz. [LICENSE](LICENSE).
