"""
NLP Pipeline - Intelligent Student Resume Analysis
PS-3 Assignment: Resume Analysis & Internship Recommendation
"""
#!pip install nltk gensim hmmlearn scikit-learn pandas numpy tensorflow

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 0 — Install & Import
# ─────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import RegexpParser
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("maxent_ne_chunker", quiet=True)
    nltk.download("words", quiet=True)
except ImportError:
    raise ImportError("Run: pip install nltk")

try:
    from gensim.models import Word2Vec
except ImportError:
    raise ImportError("Run: pip install gensim")

try:
    from hmmlearn import hmm
except ImportError:
    raise ImportError("Run: pip install hmmlearn")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report,
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix
    )
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    raise ImportError("Run: pip install scikit-learn")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[INFO] TensorFlow not found. Neural LM will use n-gram fallback.")


# ─────────────────────────────────────────────
# DOMAIN CONFIG
# ─────────────────────────────────────────────
INTERNSHIP_DOMAINS = {
    "Data Science": [
        "python", "pandas", "numpy", "matplotlib", "seaborn", "scikit", "sklearn",
        "statistics", "regression", "classification", "clustering", "sql", "tableau",
        "data analysis", "data visualization", "machine learning", "jupyter", "r programming"
    ],
    "Artificial Intelligence": [
        "tensorflow", "keras", "pytorch", "deep learning", "neural network", "nlp",
        "computer vision", "reinforcement learning", "transformer", "bert", "gpt",
        "opencv", "cnn", "rnn", "lstm", "ai", "artificial intelligence"
    ],
    "Web Development": [
        "html", "css", "javascript", "react", "angular", "vue", "nodejs", "django",
        "flask", "bootstrap", "php", "mongodb", "mysql", "restapi", "typescript",
        "frontend", "backend", "fullstack", "web", "jquery"
    ],
    "Mobile App Development": [
        "android", "ios", "flutter", "kotlin", "swift", "react native", "java",
        "mobile", "app development", "xml", "firebase", "xcode", "android studio",
        "dart", "objective-c"
    ],
    "Cyber Security": [
        "cybersecurity", "network security", "ethical hacking", "penetration testing",
        "firewall", "encryption", "cryptography", "kali linux", "metasploit",
        "vulnerability", "security audit", "wireshark", "nmap", "ceh", "cissp"
    ],
    "Cloud Computing": [
        "aws", "azure", "google cloud", "gcp", "docker", "kubernetes", "devops",
        "terraform", "ansible", "jenkins", "ci/cd", "microservices", "cloud",
        "lambda", "s3", "ec2", "serverless", "infrastructure"
    ],
    "Business Analytics": [
        "excel", "powerbi", "tableau", "business intelligence", "bi", "analytics",
        "kpi", "dashboard", "market research", "forecasting", "erp", "sap",
        "business analysis", "reporting", "data driven", "stakeholder"
    ]
}

SKILL_KEYWORDS = list(set(kw for keywords in INTERNSHIP_DOMAINS.values() for kw in keywords))

STANDARD_INTERNSHIP_DOMAINS = [
    "Data Science",
    "Web Development",
    "Mobile App Development",
    "Artificial Intelligence",
    "Cyber Security",
    "Cloud Computing",
    "Business Analytics"
]

CATEGORY_TO_DOMAIN = {
    "Frontend Developer": "Web Development",
    "Backend Developer": "Web Development",
    "Full Stack Developer": "Web Development",
    "Mobile App Developer (iOS/Android)": "Mobile App Development",
    "Machine Learning Engineer": "Artificial Intelligence",
    "Data Scientist": "Data Science",
    "Python Developer": "Data Science",
    "Cloud Engineer": "Cloud Computing",
    "Cyber Security Analyst": "Cyber Security"
}

# ─────────────────────────────────────────────
# COMPONENT 1 — Preprocessing
# ─────────────────────────────────────────────
class ResumePreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def clean(self, text: str) -> str:
        """Remove noise from resume text."""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)          # URLs
        text = re.sub(r"\S+@\S+", "", text)                  # emails
        text = re.sub(r"[^a-z0-9\s\+\#]", " ", text)        # special chars (keep + and # for C++, C#)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list:
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list) -> list:
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def process(self, text: str) -> dict:
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        sentences = sent_tokenize(cleaned)
        return {
            "cleaned": cleaned,
            "tokens": tokens,
            "filtered_tokens": filtered,
            "sentences": sentences
        }


# ─────────────────────────────────────────────
# COMPONENT 2 — POS Tagger + Skill Extractor
# ─────────────────────────────────────────────
class POSTagger:
    def tag(self, tokens: list) -> list:
        """Return (word, POS_tag) pairs."""
        return pos_tag(tokens)

    def extract_skills(self, text: str) -> list:
        """Match resume text against known skill keywords."""
        text_lower = text.lower()
        found = []
        for skill in SKILL_KEYWORDS:
            if skill in text_lower:
                found.append(skill)
        return sorted(set(found))

    def extract_nouns_and_proper(self, tagged: list) -> list:
        """Pull out NN* (nouns) and NNP* (proper nouns) — likely tech/project terms."""
        return [word for word, tag in tagged if tag.startswith("NN")]

    def analyze(self, tokens: list, raw_text: str) -> dict:
        tagged = self.tag(tokens)
        tag_counts = Counter(tag for _, tag in tagged)
        skills = self.extract_skills(raw_text)
        nouns = self.extract_nouns_and_proper(tagged)
        return {
            "tagged_tokens": tagged[:30],   # sample for display
            "tag_distribution": dict(tag_counts.most_common(10)),
            "skills_found": skills,
            "key_nouns": list(set(nouns))[:20]
        }


# ─────────────────────────────────────────────
# COMPONENT 2A — Parsing
# ─────────────────────────────────────────────
class ResumeParser:
    def __init__(self):
        self.grammar = """NP: {<JJ>*<NN.*>+}
VP: {<VB.*><NP|PP>+}"""
        self.parser = RegexpParser(self.grammar)

    def parse(self, tokens: list) -> dict:
        tagged = pos_tag(tokens)
        tree = self.parser.parse(tagged)
        phrases = {"NP": [], "VP": []}
        for subtree in tree.subtrees(filter=lambda t: t.label() in phrases):
            phrase = " ".join(word for word, _ in subtree.leaves())
            phrases[subtree.label()].append(phrase)
        return phrases

    def analyze(self, text: str) -> dict:
        tokens = word_tokenize(str(text).lower())
        return self.parse(tokens)


# ─────────────────────────────────────────────
# COMPONENT 3 — Word2Vec Embeddings
# ─────────────────────────────────────────────
class EmbeddingModel:
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def train(self, corpus: list):
        """corpus = list of token lists (one per resume)."""
        self.model = Word2Vec(
            sentences=corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=10
        )
        print(f"  [Embeddings] Trained Word2Vec on {len(corpus)} documents. Vocab size: {len(self.model.wv)}")

    def get_resume_vector(self, tokens: list) -> np.ndarray:
        """Average word vectors to get a document-level vector."""
        vectors = []
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.vector_size)

    def most_similar_to_resume(self, tokens: list, topn=5) -> list:
        """Find tech terms most similar to the resume's content."""
        resume_vec = self.get_resume_vector(tokens)
        if np.all(resume_vec == 0):
            return []
        try:
            sims = self.model.wv.similar_by_vector(resume_vec, topn=topn + 20)
            # Filter to only skill keywords
            skill_sims = [(w, round(float(s), 3)) for w, s in sims if w in SKILL_KEYWORDS]
            return skill_sims[:topn]
        except Exception:
            return []


# ─────────────────────────────────────────────
# COMPONENT 4 — Neural Language Model (LSTM)
# ─────────────────────────────────────────────
class NeuralLM:
    """
    Trains a simple character/word-level LSTM to model resume language patterns.
    Used to compute perplexity-like scores per domain.
    Falls back to n-gram model if TensorFlow is unavailable.
    """

    def __init__(self, vocab_size=3000, max_len=50):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>") if TF_AVAILABLE else None
        self.model = None
        self.ngram_counts = Counter()

    def _build_model(self):
        model = Sequential([
            Embedding(self.vocab_size, 64, input_length=self.max_len - 1),
            LSTM(128, return_sequences=False),
            Dropout(0.2),
            Dense(self.vocab_size, activation="softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def _prepare_sequences(self, texts: list):
        """Create (context → next_word) training pairs."""
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        X, y = [], []
        for seq in sequences:
            for i in range(1, len(seq)):
                n = seq[max(0, i - self.max_len + 1): i + 1]
                padded = [0] * (self.max_len - len(n)) + n
                X.append(padded[:-1])
                y.append(padded[-1])
        return np.array(X), np.array(y)

    def train_ngram_fallback(self, texts: list):
        """Simple bigram LM as fallback."""
        for text in texts:
            words = text.split()
            for i in range(len(words) - 1):
                self.ngram_counts[(words[i], words[i+1])] += 1
        print("  [Neural LM] Trained bigram language model (TF fallback).")

    def train(self, texts: list, epochs=3):
        if not TF_AVAILABLE:
            self.train_ngram_fallback(texts)
            return
        print("  [Neural LM] Preparing sequences...")
        X, y = self._prepare_sequences(texts)
        if len(X) == 0:
            print("  [Neural LM] Not enough data. Using fallback.")
            self.train_ngram_fallback(texts)
            return
        self.model = self._build_model()
        self.model.fit(X, y, epochs=epochs, batch_size=64, verbose=0)
        print(f"  [Neural LM] LSTM trained on {len(X)} sequences.")

    def predict_next_words(self, seed_text: str, n=5) -> list:
        """Predict next likely words given a seed phrase."""
        if not TF_AVAILABLE or self.model is None:
            # Fallback: most common next words
            words = seed_text.lower().split()
            if words:
                last = words[-1]
                candidates = [(w2, c) for (w1, w2), c in self.ngram_counts.items() if w1 == last]
                candidates.sort(key=lambda x: -x[1])
                return [w for w, _ in candidates[:n]]
            return []
        seq = self.tokenizer.texts_to_sequences([seed_text])
        padded = pad_sequences(seq, maxlen=self.max_len - 1, padding="pre")
        preds = self.model.predict(padded, verbose=0)[0]
        top_indices = np.argsort(preds)[-n:][::-1]
        idx_to_word = {v: k for k, v in self.tokenizer.word_index.items()}
        return [idx_to_word.get(i, "") for i in top_indices if i in idx_to_word]


# ─────────────────────────────────────────────
# COMPONENT 5 — HMM for Sequence Modeling
# ─────────────────────────────────────────────
class HMMDomainModel:
    """
    One Gaussian HMM per domain.
    Each HMM learns the distribution of Word2Vec vectors for resumes in that domain.
    At inference, we score the resume against each HMM and pick the best.
    """

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.models = {}
        self.label_encoder = LabelEncoder()

    def train(self, X_dict: dict):
        """
        X_dict: { domain_label: [resume_vector, ...] }
        """
        for domain, vectors in X_dict.items():
            if len(vectors) < self.n_components:
                continue
            arr = np.array(vectors)
            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                n_iter=50,
                random_state=42
            )
            try:
                model.fit(arr)
                self.models[domain] = model
            except Exception as e:
                print(f"  [HMM] Could not train for {domain}: {e}")
        print(f"  [HMM] Trained {len(self.models)} domain HMMs.")

    def predict(self, vector: np.ndarray) -> tuple:
        """Return (best_domain, score_dict)."""
        scores = {}
        v = vector.reshape(1, -1)
        for domain, model in self.models.items():
            try:
                score = model.score(v)
                scores[domain] = round(float(score), 3)
            except Exception:
                scores[domain] = float("-inf")
        if not scores:
            return "Unknown", {}
        best = max(scores, key=scores.get)
        return best, scores


# ─────────────────────────────────────────────
# COMPONENT 6 — TF-IDF + Random Forest Classifier
# ─────────────────────────────────────────────
class DomainClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def train(self, texts: list, labels: list):
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)
        self.is_trained = True
        print(f"  [Classifier] Accuracy: {acc:.2%}")
        print(f"  [Classifier] Macro F1: {f1_macro:.2%}, Weighted F1: {f1_weighted:.2%}")
        return {
            "accuracy": round(acc, 4),
            "report": report,
            "classes": list(self.label_encoder.classes_),
            "confusion_matrix": cm.tolist(),
            "macro_precision": round(precision_macro, 4),
            "macro_recall": round(recall_macro, 4),
            "macro_f1": round(f1_macro, 4),
            "weighted_precision": round(precision_weighted, 4),
            "weighted_recall": round(recall_weighted, 4),
            "weighted_f1": round(f1_weighted, 4)
        }

    def predict(self, text: str) -> tuple:
        X = self.vectorizer.transform([text])
        pred_idx = self.clf.predict(X)[0]
        proba = self.clf.predict_proba(X)[0]
        domain = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = round(float(proba.max()), 4)
        # Build top-3 probabilities
        top3_idx = np.argsort(proba)[-3:][::-1]
        top3 = [(self.label_encoder.inverse_transform([i])[0], round(float(proba[i]), 3)) for i in top3_idx]
        return domain, confidence, top3

    def keyword_fallback(self, text: str) -> tuple:
        """Pure keyword scoring — used when classifier isn't trained."""
        text_lower = text.lower()
        scores = {}
        for domain, keywords in INTERNSHIP_DOMAINS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[domain] = score
        best = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        conf = round(scores[best] / total, 3)
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top3 = [(d, round(s/total, 3)) for d, s in top3]
        return best, conf, top3


# ─────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────
class ResumePipeline:
    def __init__(self):
        self.preprocessor = ResumePreprocessor()
        self.pos_tagger = POSTagger()
        self.embedding_model = EmbeddingModel()
        self.neural_lm = NeuralLM()
        self.hmm_model = HMMDomainModel()
        self.parser = ResumeParser()
        self.classifier = DomainClassifier()
        self.trained = False
        self.metrics = {}

    def normalize_category(self, category: str, text: str = "") -> str:
        category = str(category).strip()
        if category in CATEGORY_TO_DOMAIN:
            return CATEGORY_TO_DOMAIN[category]

        text_lower = str(text).lower()
        scores = {
            domain: sum(1 for kw in INTERNSHIP_DOMAINS.get(domain, []) if kw in text_lower)
            for domain in STANDARD_INTERNSHIP_DOMAINS
        }
        if scores:
            best = max(scores, key=scores.get)
            return best if scores[best] > 0 else "Unknown"
        return "Unknown"

    def fit(self, df: pd.DataFrame):
        """
        Train all components on a dataset.
        df must have columns: 'Resume' and 'Category'
        """
        print("\n" + "="*55)
        print("  TRAINING NLP PIPELINE")
        print("="*55)

        texts = df["Resume"].tolist()
        original_labels = df["Category"].tolist()
        labels = [self.normalize_category(label, text)
                  for label, text in zip(original_labels, texts)]

        # 1. Preprocess all resumes
        print("\n[1/5] Preprocessing resumes...")
        processed = [self.preprocessor.process(t) for t in texts]
        cleaned_texts = [p["cleaned"] for p in processed]
        all_tokens = [p["filtered_tokens"] for p in processed]

        # 2. Train Word2Vec embeddings
        print("[2/5] Training Word2Vec embeddings...")
        self.embedding_model.train(all_tokens)

        # 3. Build vectors for each resume (for HMM)
        print("[3/5] Building resume vectors + training HMM...")
        domain_vectors = {d: [] for d in set(labels)}
        for tokens, label in zip(all_tokens, labels):
            vec = self.embedding_model.get_resume_vector(tokens)
            domain_vectors[label].append(vec)
        self.hmm_model.train(domain_vectors)

        # 4. Train Neural LM
        print("[4/5] Training Neural Language Model...")
        self.neural_lm.train(cleaned_texts[:500])  # cap for speed

        # 5. Train RF classifier
        print("[5/5] Training internship domain classifier...")
        self.metrics = self.classifier.train(cleaned_texts, labels)

        self.trained = True
        print("\n✓ Pipeline training complete.\n")

    def analyze(self, resume_text: str, original_category: str = "Unknown") -> dict:
        """Run full analysis on a single resume."""
        proc = self.preprocessor.process(resume_text)
        pos_results = self.pos_tagger.analyze(proc["tokens"], resume_text)
        parse_results = self.parser.analyze(resume_text)

        resume_vec = self.embedding_model.get_resume_vector(proc["filtered_tokens"])
        similar_skills = self.embedding_model.most_similar_to_resume(proc["filtered_tokens"])

        seed = " ".join(pos_results["skills_found"][:3]) if pos_results["skills_found"] else "data"
        predicted_words = self.neural_lm.predict_next_words(seed, n=5)

        hmm_domain, hmm_scores = self.hmm_model.predict(resume_vec)

        if self.trained and self.classifier.is_trained:
            clf_domain, confidence, top3_domains = self.classifier.predict(proc["cleaned"])
        else:
            clf_domain, confidence, top3_domains = self.classifier.keyword_fallback(resume_text)

        return {
            "resume_category": original_category,
            "recommended_internship_domain": clf_domain,
            "domain_confidence": confidence,
            "top_domain_candidates": top3_domains,
            "skills_extracted": pos_results["skills_found"],
            "key_nouns": pos_results["key_nouns"],
            "phrase_chunks": parse_results,
            "pos_tag_distribution": pos_results["tag_distribution"],
            "embedding_similar_skills": similar_skills,
            "neural_lm_predictions": predicted_words,
            "hmm_domain": hmm_domain,
            "hmm_scores": dict(sorted(hmm_scores.items(), key=lambda x: -x[1])[:5]),
            "cleaned_text": proc["cleaned"],
            "token_count": len(proc["tokens"]),
            "filtered_token_count": len(proc["filtered_tokens"]),
            "sentence_count": len(proc["sentences"]),
            "performance_metrics": self.metrics
        }


# ─────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────
def load_dataset(path: str = None) -> pd.DataFrame:
    """
    Load resume dataset.
    Expected columns: Category, Resume
    If no path is provided, this function attempts to load kaggle_resume_dataset.csv from the script folder.
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print(f"[Dataset] Loaded {len(df)} resumes from {path}")
        return df

    default_path = os.path.join(os.path.dirname(__file__), "kaggle_resume_dataset.csv")
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        print(f"[Dataset] Loaded {len(df)} resumes from {default_path}")
        return df

    raise FileNotFoundError(
        "Dataset not found. Please provide path to kaggle_resume_dataset.csv or place it next to pipeline.py."
    )


def display_result(result: dict):
    print("\n" + "="*55)
    print(" OUTPUT FROM TEST DATASET")
    print("="*55)
    print(f"Extracted student skills: {', '.join(result.get('skills_extracted', [])) or 'None'}")
    print(f"Recommended internship domain: {result.get('recommended_internship_domain', 'Unknown')}")
    print(f"Resume category: {result.get('resume_category', 'Unknown')}")
    print(f"Domain confidence: {result.get('domain_confidence', 0.0):.2f}")

    print("\nNLP analysis results:")
    print(f"  - Cleaned text: {result.get('cleaned_text', '')[:200]}{'...' if len(result.get('cleaned_text', '')) > 200 else ''}")
    print(f"  - Token count: {result.get('token_count')}")
    print(f"  - Filtered token count: {result.get('filtered_token_count')}")
    print(f"  - Sentence count: {result.get('sentence_count')}")
    print(f"  - POS tag distribution: {result.get('pos_tag_distribution')}")
    print(f"  - Key nouns: {result.get('key_nouns')}")
    print(f"  - Phrase chunks (NP): {result.get('phrase_chunks', {}).get('NP', [])[:10]}")
    print(f"  - Phrase chunks (VP): {result.get('phrase_chunks', {}).get('VP', [])[:10]}")
    print(f"  - Similar embedding skills: {result.get('embedding_similar_skills')}")
    print(f"  - Neural LM predictions: {result.get('neural_lm_predictions')}")
    print(f"  - HMM predicted domain: {result.get('hmm_domain')}")
    print(f"  - HMM scores: {result.get('hmm_scores')}")
    print(f"  - Top domain candidates: {result.get('top_domain_candidates')}")

    print("\nPerformance metrics:")
    metrics = result.get('performance_metrics', {})
    print(f"  - Classifier accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"  - Macro F1: {metrics.get('macro_f1', 'N/A')}")
    print(f"  - Weighted F1: {metrics.get('weighted_f1', 'N/A')}")
    if isinstance(metrics.get('classes'), list) and isinstance(metrics.get('confusion_matrix'), list):
        print("  - Confusion matrix classes:")
        for cls, row in zip(metrics.get('classes'), metrics.get('confusion_matrix')):
            print(f"      {cls}: {row}")
    if isinstance(metrics.get('report'), dict):
        print(f"  - Classification report classes: {list(metrics.get('report', {}).keys())[:5]}")
    print("" + "="*55 + "\n")


# ─────────────────────────────────────────────
# MAIN — Run & Export Results
# ─────────────────────────────────────────────
def run_pipeline(resume_text: str, dataset_path: str = None, original_category: str = "Unknown") -> dict:
    df = load_dataset(dataset_path)
    pipeline = ResumePipeline()
    pipeline.fit(df)
    result = pipeline.analyze(resume_text, original_category=original_category)
    return result


if __name__ == "__main__":
    #Train the pipeline on the dataset
    df = load_dataset()
    pipeline = ResumePipeline()
    pipeline.fit(df)
    

    # Test the pipeline on the entire test dataset and display results for each resume
    df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), "kaggle_resume_dataset_test.csv"))

    for idx, test_row in df_test.iterrows():
        print(f"\n--- Test record {idx + 1} / {len(df_test)} ---")
        result = pipeline.analyze(test_row["Resume"], original_category=test_row["Category"])
        display_result(result)
