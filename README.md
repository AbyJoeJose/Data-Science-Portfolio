# Aby Joe Jose — Data Science Portfolio

📍 Potsdam, NY &nbsp;|&nbsp; [LinkedIn](https://www.linkedin.com/in/aby-joe-jose-88959021b/) &nbsp;|&nbsp; 📧 abyjoejose00@gmail.com

> End-to-end data science projects spanning machine learning, NLP, SQL analytics, deep learning, LLMs, and interactive dashboards — built on real-world datasets.

---

## 📂 Projects

### 🤖 Machine Learning

---

#### [🏠 AirBnB Price Prediction](./AirBnB-Price-Prediction/)
Predicts nightly listing prices for New York City Airbnb properties using regression and classification models. Includes geospatial analysis, feature engineering, SMOTE for class imbalance, PCA, and comparison of Linear Regression, Random Forest, and XGBoost with GridSearchCV.

**Tools:** Python · Pandas · Scikit-learn · XGBoost · imbalanced-learn · Seaborn  
**Techniques:** Regression · Classification · SMOTE · PCA · GridSearchCV · Feature Engineering

---

#### [🌦️ Australia Weather Prediction](./Australia-Weather-Prediction/)
Binary classifier predicting daily rainfall in the **Melbourne area** using 7,557 weather observations from 2008–2017. Applies scikit-learn Pipeline with ColumnTransformer, GridSearchCV with StratifiedKFold, and compares Random Forest (**84% accuracy**) and Logistic Regression (83%).

**Tools:** Python · Pandas · Scikit-learn · Matplotlib · Seaborn  
**Techniques:** Binary Classification · Pipeline · GridSearchCV · StratifiedKFold · Feature Engineering

---

#### [🚲 Bike Sharing Demand Prediction](./Bike-Sharing-Profit-Prediction/)
Forecasts daily bike rental demand using iterative OLS Linear Regression with RFE feature selection and VIF-based multicollinearity elimination across **10 successive models (LM1–LM10)**. Applied to 730 daily records from 2018–2019. Final model achieves R²=0.777 (train) / **0.780 (test)** with 11 interpretable predictors.

**Tools:** Python · Pandas · Scikit-learn · Statsmodels · Matplotlib · Seaborn  
**Techniques:** OLS Regression · RFE · VIF · Multicollinearity Elimination · Feature Engineering

---

#### [🏦 Credit Default Prediction](./Credit-Default-Prediction/)
End-to-end credit risk project predicting whether a loan applicant will default using **~307K home credit applications**. Covers deep EDA across two datasets (application + previous application history), feature engineering, SMOTE for class imbalance (~8% default rate), and comparison of 4 classifiers. **XGBoost achieves ~91% accuracy and ~0.80 ROC-AUC**. Includes threshold tuning to maximize recall — minimizing missed defaulters — and feature importance analysis with actionable business recommendations.

**Tools:** Python · Pandas · Scikit-learn · XGBoost · imbalanced-learn · Seaborn  
**Techniques:** Binary Classification · EDA · SMOTE · Threshold Tuning · Feature Importance · ROC-AUC

---

#### [🗽 NYC Borough Prediction / Property Price Prediction](./NYC-Borough-Property-Price-Prediction/)

A two-part ML project applied to **84,548 NYC Property Sales transactions** — first framed as a multiclass classification task, then extended as a regression task on the same dataset and preprocessing pipeline.

**Part 1 — Borough Prediction (Classification):** Predicts which of NYC's 5 boroughs a property belongs to using 6 models (Logistic Regression, Decision Tree, SVC, Random Forest, Gradient Boosting, Neural Network). **Random Forest is the best classifier: F1=0.72, Accuracy=76%.** SMOTE handles the severe class imbalance (Brooklyn 48%, Manhattan 2%).

**Part 2 — Property Price Prediction (Regression):** Predicts sale price after filtering to arm's-length transactions ($10K–$10M, retaining 34,636 of 45,333 cleaned records). Borough becomes an input feature and SALE PRICE becomes the target. 6 regression models + Neural Network trained end-to-end with the same preprocessing pipeline. **Random Forest is the best regressor: R²=0.58, RMSE=$664K, MAE=$339K.**

| Model | Task | Key Metric |
|-------|------|-----------|
| Logistic Regression | Classification | Accuracy 63% |
| Decision Tree | Classification | F1 0.68 · Accuracy 72% |
| SVC | Classification | Accuracy 63% |
| **Random Forest** | **Classification** | **F1 0.72 · Accuracy 76%** |
| Gradient Boosting | Classification | F1 0.67 · Accuracy 73% |
| Neural Network | Classification | F1 0.68 · Accuracy 74% |
| Linear Regression (Ridge) | Regression | R²=0.36 · RMSE=$819K |
| Decision Tree Regressor | Regression | R²=0.51 · RMSE=$718K |
| SVR | Regression | R²=0.15 · RMSE=$1.05M |
| **Random Forest Regressor** | **Regression** | **R²=0.58 · RMSE=$664K** |
| Gradient Boosting Regressor | Regression | R²=0.57 · RMSE=$673K |
| Neural Network (256→128→64→32) | Regression | R²=0.55 · RMSE=$687K |

**Tools:** Python · Pandas · Scikit-learn · TensorFlow/Keras · imbalanced-learn · Seaborn  
**Techniques:** Multiclass Classification · Regression · SMOTE · GridSearchCV · Neural Networks · Pipeline · Feature Engineering

---

#### [📱 Telecom Customer Churn Prediction](./Telecom-Churn-Prediction/)
Identifies telecom customers at risk of cancelling their subscription — enabling proactive retention campaigns. Prioritizes recall to minimize missed churners. Applied to 99,999 customers with RFE feature selection down to 19 features and SMOTE for class balancing. Best model achieves **83.4% test accuracy**.

**Tools:** Python · Pandas · Scikit-learn · XGBoost · imbalanced-learn  
**Techniques:** Binary Classification · SMOTE · RFE · ROC-AUC · Precision-Recall

---

#### [🚀 SpaceX Falcon 9 Launch Success Prediction](./SpaceX-Launch-Success-Prediction/)
End-to-end capstone predicting whether a Falcon 9 first stage will land successfully — enabling launch cost estimation. Spans 5 labs: SpaceX REST API data collection, SQL analysis (10 queries on SQLite), interactive Folium maps, Plotly Dash dashboard, and 4 ML models with GridSearchCV. **Decision Tree wins at 88.9% test accuracy.**

**Tools:** Python · Pandas · Scikit-learn · Folium · Plotly Dash · Requests · BeautifulSoup · SQLite  
**Techniques:** Classification · REST API · Web Scraping · SQL · GridSearchCV · Interactive Maps

---

### 📋 Policy & Workforce Analytics

---

#### [📊 Employment, Wage & Occupation Policy Analysis](./Employment-Wage-and-Occupation-Policy/)
Quantitative policy research project applying population-level employment analysis, regression modeling, non-parametric hypothesis testing, wage inequality measurement, and anomaly detection to New York State's Occupational Employment and Wage Statistics (OEWS) — across **7,059 records spanning 10 regions and 23 major occupation groups**. OLS regression (R² = 0.974) identifies occupation group as the dominant wage driver over region. Mann-Whitney U (p = 3.6×10⁻²⁴) and Kruskal-Wallis (p = 2.1×10⁻⁵¹) confirm statistically significant wage stratification. Gini coefficients, Isolation Forest anomaly detection, and compliance flagging produce 7 enforcement-relevant policy findings.

**Tools:** Python · Pandas · Statsmodels · Scikit-learn · SciPy · Seaborn · Matplotlib  
**Techniques:** OLS Regression · Non-Parametric Hypothesis Testing · Gini Coefficient · Isolation Forest · Anomaly Detection · Wage Compliance Analysis · Coefficient of Variation · Policy Research

---

### 🔤 Natural Language Processing

---

#### [🔤 NLP Pipeline: Job Market Analysis](./NLP-Job-Market-Analysis/)
End-to-end NLP pipeline applied to real-world job postings from ZipRecruiter — progressing from classical NLP through deep learning and production LLM systems across 9 modules.

| Module | Technique |
|--------|-----------|
| Data Collection & EDA | Regex, JSON parsing, acronym & date extraction |
| Text Classification | CountVectorizer + Logistic Regression (16 preprocessing scenarios, ~83% F1) |
| Tokenization | NLTK, BPE tokenizer from scratch, Herdan's Law |
| Word Vectors | TF-IDF, GloVe (50d), Word2Vec (CBOW + Skip-gram), cosine similarity |
| Neural Networks | PyTorch feedforward network on TF-IDF features (10k dims, 20 epochs) |
| Language Generation | Fine-tuned DistilGPT2 — beam search, top-p, top-k sampling |
| NER / Skill Tagging | Fine-tuned DistilBERT (BIO scheme) — O · B-SKILL · I-SKILL |
| Prompt Engineering | Mistral-7B on AWS Bedrock — structured JSON skill extraction |
| RAG System | FAISS + sentence-transformers + Mistral-7B + LangChain |

**Tools:** Python · NLTK · HuggingFace Transformers · PyTorch · Gensim · LangChain · AWS Bedrock · FAISS  
**Techniques:** Text Classification · NER · Language Modeling · Semantic Search · RAG · Prompt Engineering

---

### 🗄️ SQL & Data Engineering

---

#### [🎬 IMDB Movie Analysis (SQL)](./IMDB-Movie-Analysis-SQL/)
SQL-based exploratory analysis answering **29 structured business questions** for RSVP Movies across 6 relational tables (movie, genre, ratings, names, role_mapping, director_mapping). Delivers data-driven recommendations on genre, director, cast, and production partners for their 2022 global film release.

**Key findings:** Drama is the dominant genre · James Mangold is top director · Vijay Sethupathi leads Indian actors · Marvel Studios tops by total votes

**Tools:** SQL · MySQL  
**Techniques:** Window Functions · CTEs · DENSE_RANK · LEAD · Weighted Averages · Subqueries

---

#### [🏦 Sales Data Warehouse & ETL Pipeline (SQL)](./Sales-Data-Warehouse-SQL/)
Designs and implements a production-style **star-schema data warehouse** for a retail business handling both sales and rental transactions — including 4 dimension tables, a Revenue fact table, an IntermediateFact staging layer, a one-way aggregate, and a DailyStoreSnapshot with 7 KPI columns. Built and loaded using MySQL.

**Tools:** SQL · MySQL  
**Techniques:** Star Schema · ETL · Surrogate Keys · Snapshot Fact Tables · Aggregate Tables

---

### 📊 Data Visualization & Dashboards

---

#### [📚 Best Selling Books by 5 Leading Authors (R / Plotly)](./Best-Selling-Books/)
Analyzes Amazon bestselling books to identify the top 5 authors by a composite Score metric (`User.Rating × 1,000 + Reviews`) and ranks their top 5 books each — rendered as a **faceted interactive Plotly bar chart** with hover tooltips showing full title, price, genre, and score.

**Tools:** R · ggplot2 · Plotly · dplyr · RColorBrewer  
**Techniques:** Composite Scoring · Deduplication · Interactive Visualization · Faceting

---

#### [🛒 Walmart 2024–25 Sales Dashboard (R Shiny)](./Walmart-Sales-RShiny/)
Interactive R Shiny dashboard visualizing Walmart customer purchase amounts by product category — with **5 toggleable demographic filters** (Age Group, City, Gender, Repeat Customer, Discount Applied) that dynamically update a bar chart using reactive programming. Each filter has a checkbox to disable it entirely, preserving selections while showing all data.

**Tools:** R · Shiny · shinyjs · ggplot2 · dplyr · scales  
**Techniques:** Reactive Programming · Dynamic Filtering · Interactive Dashboard · Age Binning

---

### 🌫️ Data Integration & APIs

---

#### [🌫️ Weather & Air Quality Data Integration (NYC)](./Weather-Air-Quality-NY/)
Integrates PM2.5 air quality data (EPA API), weather observations (NOAA API), and wind direction (Open-Meteo API) for Albany and Syracuse, NY into a normalized MySQL database using SQLAlchemy with `ON DUPLICATE KEY UPDATE`. Reveals the relationship between meteorological conditions and pollution levels through 5 visualizations including polar plots and seasonal analysis.

**Tools:** Python · Pandas · Requests · SQLAlchemy · MySQL · Matplotlib  
**Techniques:** REST API Integration · ETL · Relational Database Design · Environmental Data Analysis

---

## 🛠️ Skills

| Category | Skills |
|----------|--------|
| **Languages** | Python · R · SQL |
| **ML & Modeling** | Scikit-learn · XGBoost · Statsmodels · PyTorch · TensorFlow/Keras · Regression · Classification · Clustering |
| **NLP & LLMs** | NLTK · HuggingFace Transformers · Gensim · AWS Bedrock · LangChain · FAISS |
| **Data & Databases** | Pandas · NumPy · MySQL · SQLite · ETL · Star Schema · Data Warehousing |
| **Visualization** | Matplotlib · Seaborn · Plotly · Folium · ggplot2 · R Shiny |
| **Tools** | Jupyter Notebook · Git · GitHub · Google Colab · AWS |

---

## 📬 Contact

- 💼 [LinkedIn](https://www.linkedin.com/in/aby-joe-jose-88959021b/)
- 📧 abyjoejose00@gmail.com

---

*Each project folder contains a dedicated README with problem statement, methodology, results, and instructions to run.*
