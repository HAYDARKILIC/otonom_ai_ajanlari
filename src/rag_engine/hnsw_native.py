"""Sıfırdan bir HNSW (Hierarchical Navigable Small World) indeksi.

Bu uygulama, üretim vektör depolarının gizlediği matematiği göstermek için
Hafta 2 not defterinde kullanılır. :mod:`pipeline` içindeki üretim kodu
Chroma kullanır, ancak buradaki algoritmalar Malkov ve Yashunin'e (2018,
IEEE TPAMI) sadıktır.

Algoritmik özet
---------------
HNSW, vektör kümesi üzerinde *L* katmanlı bir graf hiyerarşisi tutar.
Yeni bir :math:`x` noktasına şu seviye atanır

.. math::

    \\ell = \\lfloor -\\ln(u) \\cdot m_L \\rfloor, \\quad u \\sim U(0, 1)

burada :math:`m_L = 1 / \\ln(M)`. Bu, üstel olarak azalan bir beklenen
katman doluluğu üretir: noktaların kabaca :math:`M^{-\\ell}` kadarı
:math:`\\ell` katmanında yer alır. En üst katman bir giriş noktası
küçük-dünya grafı gibi davranır; en alt katman yoğun bir kNN grafıdır.

Arama açgözlü bir iniştir: en üst katmandan girilir, açgözlü genişletmeyle
yerel olarak en yakın komşu bulunur, ardından bir alt katmana inilip
iyileştirilir. En alt katmanda arama, ``ef`` boyutlu bir öncelik kuyruğuyla
*k* en yakın komşuyu döndürür.

Beklenen sorgu maliyeti düzgün dağılımlı veri için :math:`O(\\log N)`'dir —
kaba kuvvet aramanın :math:`O(N)` maliyetine göre çarpıcı bir iyileşme.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Node:
    vector: np.ndarray
    payload: dict
    neighbors: list[list[int]] = field(default_factory=list)  # katman başına komşu listeleri


class HNSWIndex:
    """Kosinüs benzerliği vektörleri üzerinde öğretici bir HNSW uygulaması.

    Parametreler
    ------------
    dim : int
        Vektör boyutu.
    M : int
        Katman başına düğüm başına en fazla komşu sayısı (katman 0'ın üstünde).
        Katman 0, orijinal makaleye göre ``M * 2`` kullanır.
    ef_construction : int
        Ekleme sırasında dinamik aday listesi boyutu.
    ef_search : int
        Sorgu sırasında dinamik aday listesi boyutu.
    seed : int
        Seviye atamasının tekrarlanabilirliği için rastgele sayı üreteci tohumu.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 1234,
    ) -> None:
        self.dim = dim
        self.M = M
        self.M0 = M * 2
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m_L = 1.0 / math.log(M)  # seviye atama normalizasyon sabiti
        self._rng = random.Random(seed)
        self._nodes: list[_Node] = []
        self._entry_point: int | None = None
        self._max_layer: int = -1

    # ------------------------------------------------------------------ #
    # Genel API                                                          #
    # ------------------------------------------------------------------ #

    def add(self, vector: np.ndarray, payload: dict) -> int:
        if vector.shape != (self.dim,):
            raise ValueError(f"Beklenen boyut ({self.dim},), gelen {vector.shape}.")
        level = self._sample_level()
        node_id = len(self._nodes)
        node = _Node(vector=vector, payload=payload,
                     neighbors=[[] for _ in range(level + 1)])
        self._nodes.append(node)

        if self._entry_point is None:
            self._entry_point = node_id
            self._max_layer = level
            return node_id

        # En üst katmandan level+1'e kadar açgözlü iniş.
        curr = self._entry_point
        for layer in range(self._max_layer, level, -1):
            curr = self._greedy_search(vector, curr, layer)

        # `level`'a kadar her katmanda en yakın ef_construction komşuya bağlan.
        for layer in range(min(level, self._max_layer), -1, -1):
            candidates = self._search_layer(vector, curr, self.ef_construction, layer)
            m = self.M0 if layer == 0 else self.M
            selected = self._select_neighbors_heuristic(vector, candidates, m)
            node.neighbors[layer] = [c for _, c in selected]
            # Karşılıklı kenarları ekle; aşırı dolu komşu listelerini buda.
            for _, neighbor_id in selected:
                self._nodes[neighbor_id].neighbors[layer].append(node_id)
                if len(self._nodes[neighbor_id].neighbors[layer]) > m:
                    self._prune(neighbor_id, layer, m)
            curr = selected[0][1]

        if level > self._max_layer:
            self._max_layer = level
            self._entry_point = node_id
        return node_id

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[float, dict]]:
        if self._entry_point is None:
            return []
        curr = self._entry_point
        for layer in range(self._max_layer, 0, -1):
            curr = self._greedy_search(query, curr, layer)
        results = self._search_layer(query, curr, max(self.ef_search, k), layer=0)
        results.sort(key=lambda x: x[0])  # artan uzaklık
        return [(d, self._nodes[nid].payload) for d, nid in results[:k]]

    # ------------------------------------------------------------------ #
    # İç işleyiş                                                         #
    # ------------------------------------------------------------------ #

    def _sample_level(self) -> int:
        u = self._rng.random()
        # ℓ = ⌊-ln(u) · m_L⌋
        return int(math.floor(-math.log(u + 1e-12) * self.m_L))

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        # Girdilerin L2-normalize olduğunu varsayan kosinüs uzaklığı.
        return float(1.0 - np.dot(a, b))

    def _greedy_search(self, query: np.ndarray, entry: int, layer: int) -> int:
        curr = entry
        curr_d = self._distance(query, self._nodes[curr].vector)
        improved = True
        while improved:
            improved = False
            for neighbor in self._nodes[curr].neighbors[layer]:
                d = self._distance(query, self._nodes[neighbor].vector)
                if d < curr_d:
                    curr_d, curr, improved = d, neighbor, True
        return curr

    def _search_layer(
        self, query: np.ndarray, entry: int, ef: int, layer: int
    ) -> list[tuple[float, int]]:
        # İki yığın: adaylar (uzaklığa göre min-yığın), sonuçlar (uzaklığa göre max-yığın).
        entry_d = self._distance(query, self._nodes[entry].vector)
        candidates: list[tuple[float, int]] = [(entry_d, entry)]
        results: list[tuple[float, int]] = [(-entry_d, entry)]
        visited: set[int] = {entry}

        while candidates:
            d, curr = heapq.heappop(candidates)
            worst_in_results = -results[0][0]
            if d > worst_in_results:
                break
            for neighbor in self._nodes[curr].neighbors[layer]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                nd = self._distance(query, self._nodes[neighbor].vector)
                worst_in_results = -results[0][0]
                if nd < worst_in_results or len(results) < ef:
                    heapq.heappush(candidates, (nd, neighbor))
                    heapq.heappush(results, (-nd, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)
        return [(-neg_d, nid) for neg_d, nid in results]

    def _select_neighbors_heuristic(
        self, query: np.ndarray, candidates: list[tuple[float, int]], m: int
    ) -> list[tuple[float, int]]:
        """Çeşitliliği koruyan sezgisel seçim (Malkov ve Yashunin, Alg. 4)."""
        sorted_cands = sorted(candidates, key=lambda x: x[0])
        selected: list[tuple[float, int]] = []
        for d, cand_id in sorted_cands:
            if len(selected) >= m:
                break
            # Adayı yalnızca sorguya, önceden seçilmiş herhangi bir komşuya olduğundan
            # daha yakınsa kabul et — bu, açısal çeşitliliği korur.
            cand_vec = self._nodes[cand_id].vector
            ok = True
            for _, s_id in selected:
                if self._distance(cand_vec, self._nodes[s_id].vector) < d:
                    ok = False
                    break
            if ok:
                selected.append((d, cand_id))
        return selected

    def _prune(self, node_id: int, layer: int, m: int) -> None:
        node = self._nodes[node_id]
        scored = [
            (self._distance(node.vector, self._nodes[nid].vector), nid)
            for nid in node.neighbors[layer]
        ]
        kept = self._select_neighbors_heuristic(node.vector, scored, m)
        node.neighbors[layer] = [nid for _, nid in kept]
