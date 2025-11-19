# SpotLite

[![zh](https://img.shields.io/badge/lang-中文-blue)](https://github.com/wayneclub/SpotLite/blob/main/README.zh-Hant.md)

SpotLite is a system for personalized, aspect‑based restaurant recommendation powered by Google Maps data and advanced NLP.

## Documentation
- [Architecture](docs/ARCHITECTURE.md)
- [Developer Guide](docs/DEVELOPER_GUIDE.md)
- [Scraping Guide](docs/SCRAPING.md)
- [NLP Pipeline](docs/NLP_PIPELINE.md)
- [API / CLI](docs/API.md)
- [CLI Guide](docs/CLI.md)
- [Roadmap](docs/ROADMAP.md)

---

## 1. Motivation & Problem Statement

Existing restaurant search platforms (such as Google Maps and Yelp) mainly rely on a **rating + scattered keyword** model. However, this approach presents three major pain points for travelers with specific needs:

1. **Information Overload**: Users must open and read numerous reviews to understand specific aspects like “Is it quiet?” or “Is it reasonably priced?”, which makes decision-making inefficient.
2. **Fragmented Keywords**: Semantically similar expressions (e.g., cheap, affordable, budget-friendly, good value) are not automatically grouped, resulting in incomplete information retrieval and poor holistic understanding.
3. **Lack of Deep Personalization**: Existing filtering options are too basic and cannot handle complex multi-criteria queries such as “close to the attraction, romantic ambiance, but not crowded.”

### Our Solution: SpotLite

To address these issues, we propose **SpotLite**, a system that automatically recommends nearby restaurants when the user provides a **Google Maps location link**. It generates a comprehensive, AI-powered report that includes **aspect-based analysis, keyword clustering, summarization, and personalized ranking**.

| Feature | Google Maps | SpotLite (Proposed System) |
|:---|:---:|:---:|
| Rating-based ranking | ✔ | ✔ |
| Scattered keyword search | ✔ | ✘ (Automatic clustering) |
| **Aspect-based analysis** | ✘ | ✔ |
| **AI-generated summary** | ✘ | ✔ |
| **Multi-dimensional personalization** | ✘ | ✔ |

**Scope**: To ensure focus and model reliability, the initial phase will only process **English-language reviews from the past year**.

---

## 2. Related Work

Our system builds on foundational NLP models and algorithms, especially Transformer-based architectures such as BERT \cite{devlin2018bert}, to achieve deep linguistic understanding.

- **Aspect-Based Sentiment Analysis (ABSA)**: We follow classical approaches \cite{hu2004mining} for aspect extraction and sentiment classification — the core of our review analysis module.
- **Keyword Extraction**: We adopt the graph-based **TextRank** algorithm \cite{mihalcea2004textrank} to identify key terms, which is more robust than simple frequency-based models.
- **Semantic Clustering**: To group synonyms, we utilize **Sentence-BERT** embeddings \cite{reimers2019sentence} for semantic similarity computation.
- **Summarization**: We employ a fine-tuned **T5 (Text-to-Text Transfer Transformer)** model \cite{raffel2020exploring} to perform aspect-aware abstractive summarization.

---

## 3. System Design & Architecture

### 3.1 Processing Pipeline

```markdown
Google Maps Link → Geocoding → Nearby Search → Reviews
↓ Cleaning & Deduplication
↓ Aspect Classification
↓ Keyword Extraction & Semantic Clustering
↓ Sentiment Analysis & Quantification
↓ AI Summarization
↓ Personalized Filtering & Ranking
→ Output JSON / PWA Card
```

### 3.2 Core Components

1. **Data Preprocessing**
   - **Tools**: `langdetect`, SimHash, MiniLM embeddings.
   - **Functions**: Filter non-English reviews; use SimHash for syntactic deduplication and MiniLM embeddings for semantic deduplication to ensure high data quality.

2. **Aspect Classification**
   - **Taxonomy**: Taste, Price, Service, Ambiance, Wait Time, Location, Cuisine, Special Needs.
   - **Method**: Semi-supervised approach using a seed dictionary for initial tagging, then expanding with embedding similarity to capture unseen synonyms.

3. **Sentiment Analysis & Quantification**
   - **Unit of Analysis**: Sentence-level sentiment per aspect (positive/negative).
   - **Quantitative Metrics**: Aggregate counts of `positive_mentions` and `negative_mentions` for the past year.
   - **Keyword Attribution**: Collect aspect-related positive and negative keywords.
   - **Scoring Formula**:
     $$Score_{aspect} = \sigma(w_1 \cdot \log(N) + w_2 \cdot \overline{|S|})$$
     where $N$ is the mention count, $\overline{|S|}$ is the average sentiment intensity, and $w_1, w_2$ are weights.

4. **AI Summarization & Personalized Ranking**
   - **Summarization**: A fine-tuned **t5-small** model generates concise 2–3 sentence summaries based on top-scoring aspects and their sentiment keywords.
   - **Personalized Ranking**: The final ranking score integrates rating, number of reviews, distance, aspect scores, and user preference matching:
     $$final\_score = \alpha \cdot rating + \beta \cdot \log(1+reviews) - \gamma \cdot distance + \delta \cdot pref\_boost$$

---

### 3.3 Example API Input / Output

#### Input (JSON)
```json
{
  "location_url": "https://share.google/nWiVMXDYMuUHvmxte",
  "search_radius_meters": 800,
  "preferences": {
    "must_have": ["Japanese Cuisine"],
    "nice_to_have": ["quiet", "priced"],
    "exclude_keywords": ["queue", "noisy"]
  }
}
```

#### Output (JSON)
```json
[
  {
    "rank": 1,
    "name": "The Tuna Sushi",
    "ai_summary": "A highly-rated spot known for its fresh fish. The ambiance is consistently described as quiet, making it ideal for dates. While praised for quality, some note the prices are on the higher end.",
    "aspect_analysis": [
      {
        "aspect": "Taste",
        "score": 0.94,
        "positive_mentions": 85,
        "negative_mentions": 5,
        "positive_keywords": ["fresh fish", "quality tuna"],
        "negative_keywords": ["rice was mushy"]
      },
      {
        "aspect": "Ambiance",
        "score": 0.87,
        "positive_mentions": 65,
        "negative_mentions": 10,
        "positive_keywords": ["quiet", "intimate"],
        "negative_keywords": ["a bit dark"]
      }
    ],
    "pros": ["High-quality ingredients", "Quiet atmosphere"],
    "cons": ["Higher price point"]
  }
]
```

---

## 4. Hypothesis & Evaluation

**Research Hypothesis**:
A system integrating aspect-based features and multi-criteria preference ranking will achieve significantly higher **relevance (measured by NDCG)** and **user efficiency (measured by task completion time)** compared to baseline rating-based systems.

### 4.1 Module-Level Evaluation
- **Aspect Classification**: F1-score ≥ 0.75 on a manually labeled dataset (N=200).
- **Keyword Clustering**: ≥ 80% accuracy compared to a standard synonym reference set.

### 4.2 End-to-End Evaluation
- **Ranking Relevance**: Achieve ≥ 15% improvement in NDCG@10 over Google Maps baseline.
- **User Efficiency Study**: In a controlled user study, reduce task completion time by ≥ 50%, with ≤ 10% difference in accuracy compared to the control group.

---

## 5. Timeline & Deliverables

The project will be completed within **four weeks**, with milestones and suggested responsibilities as follows:

| Week | Milestone | Deliverables | Suggested Owner |
|:---:|:---|:---|:---|
| **W1** | Data Acquisition & Preprocessing | Clean JSON output for Top-20 reviews | A (API), B (Cleaning) |
| **W2** | Core NLP Module I | Aspect, sentiment, and keyword annotation per review | C (Classification), D (Extraction) |
| **W3** | Core NLP Module II | Full JSON with summarization and personalized ranking | E (Clustering/Ranking), F (Summarization) |
| **W4** | Integration & Evaluation | PWA demo + final evaluation report | All members |

### Final Deliverables
1. **Backend API** – Input: Google Maps link → Output: full analytical JSON
2. **Frontend Demo (PWA)** – Mobile-friendly web interface for visualization
3. **Final Report** – Includes system diagrams, evaluation results, and case studies
4. **Demo Video** – 1–2 min screen recording showing the workflow

---

## 6. Conclusion & Future Work

This project integrates NLP, search experience optimization, and quantitative evaluation to build a recommendation system that truly addresses travelers’ pain points.

**Future Directions**:
- Extend to **Hotel Recommendation** (e.g., convenient transport, quiet, good breakfast).
- Extend to **Activity Recommendation** (e.g., family-friendly, unique experiences, high value).
- Ultimately, evolve into a complete **Smart Travel Recommendation Solution**.

---

## 7. References

- Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT.
- Mihalcea, R., & Tarau, P. (2004). *TextRank: Bringing Order into Texts*. EMNLP.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
- Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR.
- Hu, M., & Liu, B. (2004). *Mining and Summarizing Customer Reviews*. KDD.


# File: docs/ARCHITECTURE.md
# SpotLite Architecture

This document describes the system architecture, module structure, and data flow for SpotLite.

## 1. High-Level Pipeline
Google Maps Link → URL Expansion → Details + Reviews → Cleaning → NLP Pipeline → Summary → Output

## 2. Project Structure
```
(keep empty sections; user will expand)
```

# File: docs/DEVELOPER_GUIDE.md
# SpotLite Developer Guide

This guide covers config loading, logging, module imports, and execution flow.

# File: docs/SCRAPING.md
# Google Maps Scraping Guide

Details for Selenium-based scraping, short-link resolution, scrolling logic, and review extraction.

# File: docs/NLP_PIPELINE.md
# NLP Pipeline Documentation
Covers: Aspect extraction, keyword clustering, sentiment analysis, AI summarization.

# File: docs/API.md
# CLI & API Reference
Usage for: details, reviews, analyze, debug mode.

# File: docs/ROADMAP.md
# SpotLite Roadmap
Future features and tasks.
