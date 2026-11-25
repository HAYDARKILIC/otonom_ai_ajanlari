"""Ajan Tabanlı Bağlam Mühendisliği — Sıfırdan.

Aşağıdakilerin üretim kalitesinde, matematiği öne alan bir uygulaması:
    - Model Context Protocol (MCP) sunucuları ve istemcileri
    - Hiyerarşik erişimle zenginleştirilmiş üretim (RAG)
    - Bi-encoder / cross-encoder yeniden sıralama hatları
    - ReAct akıl yürütme ve eyleme döngüsü
    - Kendini düzelten çoklu ajan sistemleri
"""

__version__ = "0.1.0"

# Proje kökündeki (veya çalışma dizinindeki) .env dosyasını yükle; böylece
# ANTHROPIC_API_KEY gibi değişkenler elle export edilmeden kullanılabilir.
# Zaten tanımlı ortam değişkenlerinin üzerine yazılmaz.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except ImportError:  # python-dotenv kurulu değilse sessizce geç
    pass
