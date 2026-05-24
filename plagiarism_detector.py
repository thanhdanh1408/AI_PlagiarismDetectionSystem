"""
Plagiarism Detection Engine
- TF-IDF + Cosine Similarity
- Sentence Embedding + Cosine Similarity
- Hybrid (ket hop ca hai)
- Tim kiem trong dataset VISP
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessor import preprocess_text, split_into_sentences
import time


class TFIDFEngine:
    """TF-IDF + Cosine Similarity"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)

    def compute_similarity(self, text1, text2):
        p1 = preprocess_text(text1)
        p2 = preprocess_text(text2)
        if not p1 or not p2:
            return 0.0
        try:
            matrix = self.vectorizer.fit_transform([p1, p2])
            return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
        except:
            return 0.0

    def compute_sentence_similarities(self, sents1, sents2):
        if not sents1 or not sents2:
            return np.array([])
        processed = [preprocess_text(s) or "empty" for s in sents1] + [preprocess_text(s) or "empty" for s in sents2]
        try:
            matrix = self.vectorizer.fit_transform(processed)
            n1 = len(sents1)
            return cosine_similarity(matrix[:n1], matrix[n1:])
        except:
            return np.zeros((len(sents1), len(sents2)))


class EmbeddingEngine:
    """Sentence Embedding + Cosine Similarity (Đã tối ưu cho Tiếng Việt)"""
    def __init__(self, model_name='keepitreal/vietnamese-sbert'):
        self.model = None
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Tự động chọn GPU nếu có, ngược lại dùng CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.model = SentenceTransformer(self.model_name, device=device)
            print(f"[OK] Model loaded: {self.model_name} on {device.upper()}")
        except Exception as e:
            print(f"[WARN] Cannot load embedding model: {e}")

    def compute_similarity(self, text1, text2):
        if self.model is None: return 0.0
        try:
            from sentence_transformers import util
            e1 = self.model.encode(text1, convert_to_tensor=True)
            e2 = self.model.encode(text2, convert_to_tensor=True)
            return float(max(0.0, min(1.0, util.cos_sim(e1, e2).item())))
        except Exception as e:
            print(f"[ERROR] compute_similarity error: {e}")
            return 0.0

    def compute_sentence_similarities(self, sents1, sents2):
        if self.model is None or not sents1 or not sents2:
            return np.zeros((len(sents1), len(sents2)))
        try:
            from sentence_transformers import util
            e1 = self.model.encode(sents1, convert_to_tensor=True)
            e2 = self.model.encode(sents2, convert_to_tensor=True)
            return util.cos_sim(e1, e2).cpu().numpy()
        except Exception as e:
            print(f"[ERROR] compute_sentence_similarities error: {e}")
            return np.zeros((len(sents1), len(sents2)))


class DatasetSearcher:
    """Tim kiem van ban tuong dong trong dataset VISP"""
    def __init__(self, dataset_path='visp_train.xlsx', max_docs=3000):
        self.corpus = []
        self.corpus_ids = []
        self.corpus_sources = []
        self.corpus_topics = []
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)
        self.tfidf_matrix = None
        self._load_dataset(dataset_path, max_docs)

    def _load_dataset(self, path, max_docs):
        print(f"[*] Loading dataset for search: {path} (max {max_docs} docs)...")
        try:
            df = pd.read_excel(path)
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
            # Lay mau ngau nhien
            if len(df) > max_docs:
                df = df.sample(n=max_docs, random_state=42).reset_index(drop=True)
            for _, row in df.iterrows():
                orig = str(row['original_text'])
                if orig and orig != 'nan' and len(orig) > 10:
                    self.corpus.append(orig)
                    self.corpus_ids.append(str(row.get('original_id', '')))
                    self.corpus_sources.append(str(row.get('source', '')))
                    self.corpus_topics.append(str(row.get('topic', '')))
            # Pre-compute TF-IDF matrix
            processed = [preprocess_text(t) or "empty" for t in self.corpus]
            self.tfidf_matrix = self.vectorizer.fit_transform(processed)
            print(f"[OK] Loaded {len(self.corpus)} documents into search index")
        except Exception as e:
            print(f"[ERROR] Cannot load dataset: {e}")

    def search(self, query_text, top_k=10, threshold=0.1):
        """Tim cac van ban tuong dong voi query trong dataset"""
        if not self.corpus or self.tfidf_matrix is None:
            return []
        processed_query = preprocess_text(query_text)
        if not processed_query:
            return []
        try:
            query_vec = self.vectorizer.transform([processed_query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            # Lay top_k ket qua co similarity >= threshold
            top_indices = similarities.argsort()[::-1][:top_k]
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score >= threshold:
                    results.append({
                        'text': self.corpus[idx],
                        'score': round(score, 4),
                        'source': self.corpus_sources[idx],
                        'topic': self.corpus_topics[idx],
                        'doc_id': self.corpus_ids[idx],
                    })
            return results
        except:
            return []


class PlagiarismDetector:
    """Bo phat hien dao van chinh"""
    TFIDF_WEIGHT = 0.4
    EMBEDDING_WEIGHT = 0.6

    def __init__(self, use_embedding=True, dataset_path='visp_train.xlsx', max_docs=3000):
        print("[*] Initializing Plagiarism Detector...")
        self.tfidf_engine = TFIDFEngine()
        print("[OK] TF-IDF Engine ready")
        self.embedding_engine = None
        if use_embedding:
            self.embedding_engine = EmbeddingEngine()
            if self.embedding_engine.model is None:
                self.embedding_engine = None
        # Load dataset for search
        self.searcher = DatasetSearcher(dataset_path, max_docs)
        print("[OK] System ready!\n")

    def search_dataset(self, text, top_k=10, threshold=0.1):
        """Tim van ban tuong dong trong dataset"""
        return self.searcher.search(text, top_k=top_k, threshold=threshold)

    def detect(self, text1, text2, method='hybrid'):
        start = time.time()
        if not text1 or not text2:
            return {'similarity_score':0,'tfidf_score':0,'embedding_score':0,
                    'level':'low','message':'Vui long nhap van ban',
                    'matched_sentences':[],'processing_time':0}
        tfidf_score = self.tfidf_engine.compute_similarity(text1, text2)
        emb_score = 0.0
        if method in ('embedding','hybrid') and self.embedding_engine:
            emb_score = self.embedding_engine.compute_similarity(text1, text2)
        if method == 'tfidf':
            final = tfidf_score
        elif method == 'embedding' and self.embedding_engine:
            final = emb_score
        elif method == 'hybrid' and self.embedding_engine:
            final = self.TFIDF_WEIGHT * tfidf_score + self.EMBEDDING_WEIGHT * emb_score
        else:
            final = tfidf_score
        level, msg = self._classify(final)
        matched = self._find_matches(text1, text2, method)
        return {
            'similarity_score': round(final, 4),
            'tfidf_score': round(tfidf_score, 4),
            'embedding_score': round(emb_score, 4),
            'level': level, 'message': msg,
            'matched_sentences': matched,
            'processing_time': round(time.time() - start, 3)
        }

    def _classify(self, score):
        pct = score * 100
        if score < 0.30:
            return ('low', f'Noi dung khac nhau ({pct:.1f}%)')
        elif score < 0.60:
            return ('medium', f'Co kha nang trung lap ({pct:.1f}%)')
        else:
            return ('high', f'Dao van cao ({pct:.1f}%)')

    def _find_matches(self, text1, text2, method, threshold=0.5):
        s1 = split_into_sentences(text1)
        s2 = split_into_sentences(text2)
        if not s1 or not s2: return []
        if method == 'embedding' and self.embedding_engine:
            sim = self.embedding_engine.compute_sentence_similarities(s1, s2)
        else:
            sim = self.tfidf_engine.compute_sentence_similarities(s1, s2)
        if sim.size == 0: return []
        matched = []
        for i in range(len(s1)):
            for j in range(len(s2)):
                if sim[i][j] >= threshold:
                    matched.append({'sent1':s1[i].strip(),'sent2':s2[j].strip(),'score':round(float(sim[i][j]),4)})
        matched.sort(key=lambda x: x['score'], reverse=True)
        return matched[:20]
