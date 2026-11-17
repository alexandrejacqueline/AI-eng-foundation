import sys

# Ajout du dossier src au chemin d'importation
sys.path.insert(0, "src")

from foundation.rag.bm25 import BM25Index
from foundation.rag.chunking import chunk_text
from foundation.rag.hybrid import reciprocal_rank_fusion
from foundation.rag.reranker import Reranker
from foundation.rag.store import index_chunks, search as chroma_search

# 1. TEXTE SOURCE (Simulé)
DOCUMENT = """
Fiche Produit MDD 2026.
Le Shampoing Bio MDD (EAN 3660001) coûte 4.50 EUR en temps normal. Cependant, durant le mois de novembre, son prix chute exceptionnellement à 2.99 EUR pour les membres.
Le Savon Doux Amande (EAN 3660002) coûte 2.00 EUR.
Conditions de livraison : La livraison est offerte à partir de 50 EUR d'achat en magasin.
Le Savon Doux Amande (EAN 3660003) coûte 5.00 EUR.
"""

print("==================================================")
print("1. DECOUPAGE EN CHUNKS (chunking.py)")
print("==================================================")
chunks = chunk_text(DOCUMENT, chunk_size=110, chunk_overlap=30)
for c in chunks:
    print(f"➡️ Chunk {c.chunk_id} [{c.start_char}:{c.end_char}] : {c.text!r}\n")

QUESTION = "Quel est le prix en promotion du produit 3660001 ?"
print(f"❓ QUESTION UTILISATEUR : {QUESTION!r}\n")

print("==================================================")
print("2. RECHERCHE DENSE (ChromaDB / store.py)")
print("==================================================")
collection = index_chunks(chunks)
chroma_res = chroma_search(collection, QUESTION, n_results=3)
dense_texts = chroma_res.get("documents", [[]])[0]
print(f"ChromaDB a trouvé {len(dense_texts)} extraits basés sur la similarité vectorielle.")

# Retrouver les objets Chunk correspondants aux résultats ChromaDB
dense_chunks = [c for c in chunks if c.text in dense_texts]
for rank, c in enumerate(dense_chunks, start=1):
    print(f"  Rang #{rank} (Dense) -> Chunk {c.chunk_id}: {c.text[:60]}...")

print("\n==================================================")
print("3. RECHERCHE SPARSE (BM25 / bm25.py)")
print("==================================================")
bm25_index = BM25Index(chunks)
bm25_res = bm25_index.search(QUESTION, top_k=3)
sparse_chunks = [r.chunk for r in bm25_res]
for rank, r in enumerate(bm25_res, start=1):
    print(f"  Rang #{rank} (BM25 Score: {r.score:.2f}) -> Chunk {r.chunk.chunk_id}: {r.chunk.text[:60]}...")

print("\n==================================================")
print("4. FUSION HYBRIDE RRF (hybrid.py)")
print("==================================================")
hybrid_results = reciprocal_rank_fusion(dense_chunks, sparse_chunks, k=60)
for rank, res in enumerate(hybrid_results[:3], start=1):
    print(
        f"  Rang #{rank} (RRF Score: {res.rrf_score:.4f}) -> Chunk {res.chunk.chunk_id} "
        f"[Dense Rank: {res.dense_rank}, Sparse Rank: {res.sparse_rank}]"
    )

print("\n==================================================")
print("5. RERANKING CROSS-ENCODER (reranker.py)")
print("==================================================")
reranker = Reranker()
candidates = [h.chunk for h in hybrid_results[:3]]
final_ranked = reranker.rerank(query=QUESTION, chunks=candidates, top_k=2)

for rank, r in enumerate(final_ranked, start=1):
    print(f"🏆 Gagnant #{rank} (Score Rerank: {r.score:.2f}) -> Chunk {r.chunk.chunk_id}:")
    print(f"   {r.chunk.text}\n")