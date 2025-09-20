import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import nltk
from nltk.corpus import stopwords
from google_play_scraper import app, reviews, Sort
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from collections import Counter
from tqdm import tqdm
from packaging import version
import time

# --- App Configuration ---
st.set_page_config(
    page_title="Advanced Google Play NLP Analyzer",
    page_icon="🔬",
    layout="wide",
)

# --- NLTK and spaCy Model Loading ---
# These will run only once.
@st.cache_resource
def load_nlp_models():
    """Load spaCy and download NLTK stopwords and tokenizers."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        # Load spaCy model with only the components we need (tagger, lemmatizer) for speed
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        st.info("Downloading spaCy model... this may take a minute.")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return nlp, set(stopwords.words("english"))

NLP, NLTK_STOPWORDS = load_nlp_models()
SPACY_STOPWORDS = NLP.Defaults.stop_words
COMBINED_STOPWORDS = NLTK_STOPWORDS.union(SPACY_STOPWORDS)


# --- Helper & Core Functions from your Notebook ---

# 1. Scraping and Basic Info
@st.cache_data(show_spinner="Fetching app details...")
def get_app_details(app_id):
    try:
        return app(app_id)
    except Exception as e:
        st.error(f"Error fetching app details for '{app_id}': {e}")
        return None

@st.cache_data(show_spinner=False) # Spinner is disabled for custom progress bar
def scrape_reviews(app_id, target_count):
    """Scrapes reviews in batches to show progress."""
    all_reviews = []
    token = None
    batch_size = 1000  # Increased batch size for faster scraping of large amounts

    progress_bar = st.progress(0, text=f"Starting review scrape for {app_id}...")
    
    try:
        while len(all_reviews) < target_count:
            # Fetches a batch of reviews
            rvs, token = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=batch_size,
                continuation_token=token
            )
            
            if not rvs:
                break  # No more reviews to fetch
                
            all_reviews.extend(rvs)
            
            # Update progress bar
            progress_value = min(len(all_reviews) / target_count, 1.0)
            progress_bar.progress(progress_value, text=f"Scraping reviews... Fetched {len(all_reviews)} of {target_count}")

            if not token:
                break # Reached the end of reviews
        
        progress_bar.empty() # Remove the progress bar
        st.success(f"Successfully scraped {len(all_reviews)} reviews.")
        df = pd.DataFrame(all_reviews)
        return df.head(target_count) # Ensure we don't exceed the target
    except Exception as e:
        progress_bar.empty()
        st.error(f"An error occurred during scraping: {e}")
        return pd.DataFrame()

# 2. Preprocessing
def normalize_version(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    m = re.search(r'(\d+(?:\.\d+)+)', s)
    if not m: return np.nan
    parts = m.group(1).split('.')
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else parts[0]

def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Removed @st.cache_data to ensure this re-runs for each new analysis
def run_preprocessing(_df):
    df = _df.copy()
    with st.status("Preprocessing data...", expanded=True) as status:
        status.update(label="Converting dates and sorting...")
        df['at'] = pd.to_datetime(df['at'])
        df_sorted = df.sort_values('at', ascending=False).reset_index(drop=True)
        
        status.update(label="Normalizing and filling app versions...")
        df_sorted['version_raw'] = df_sorted['appVersion'].fillna(df_sorted.get('reviewCreatedVersion'))
        df_sorted['version_norm'] = df_sorted['version_raw'].apply(normalize_version)
        df_sorted['version_filled'] = df_sorted['version_norm'].ffill().bfill()
        
        df_clean = df_sorted[['content','score','version_filled','at','thumbsUpCount']].copy()
        df_clean.rename(columns={'version_filled':'appVersion'}, inplace=True)
        df_clean.dropna(subset=['content'], inplace=True)
        
        status.update(label="Applying spaCy lemmatization (optimized)...")
        
        texts = df_clean['content'].tolist()
        processed_texts = []
        total_rows = len(texts)
        progress_bar = st.progress(0, text=f"Lemmatizing {total_rows} reviews...")

        # Use nlp.pipe for efficient batch processing
        docs = NLP.pipe(texts, batch_size=50)

        for i, doc in enumerate(docs):
            # Process and append the lemmatized text
            processed_texts.append(" ".join([
                token.lemma_.lower() for token in doc if
                token.is_alpha
                and len(token.lemma_) >= 3
                and token.lemma_.lower() not in COMBINED_STOPWORDS
            ]))
            # Update progress bar periodically to avoid slowing down the UI
            if (i + 1) % 100 == 0:
                progress_bar.progress((i + 1) / total_rows, text=f"Lemmatizing reviews... ({i+1}/{total_rows})")
        
        df_clean['preproc_text'] = processed_texts
        progress_bar.empty()
        status.update(label="Preprocessing complete!", state="complete", expanded=False)
    return df_clean

# 3. Term Frequency Analysis
# Removed @st.cache_data to ensure this re-runs for each new analysis
def analyze_term_frequency(_df_clean):
    generic_opinion_words = {
        'good','bad','nice','worst','poor','best','excellent','awesome','super',
        'great','ok','okay','useless','useful','better','amazing','love','hate',
        'liked','disliked','perfect','friendly','cool','able','hai','app'
    }
    stop_words = list(COMBINED_STOPWORDS.union(generic_opinion_words))
    texts = _df_clean['preproc_text'].dropna().tolist()
    
    if not texts:
        return pd.DataFrame(), pd.DataFrame()

    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1, 2), min_df=10)
    X = vectorizer.fit_transform(texts)
    
    word_counts = pd.DataFrame({
        'term': vectorizer.get_feature_names_out(),
        'count': X.toarray().sum(axis=0)
    }).sort_values('count', ascending=False).reset_index(drop=True)

    top_terms = word_counts.head(100)['term'].tolist()
    
    freq_data = []
    for term in top_terms:
        pattern = rf'\b{re.escape(term)}\b'
        for score in range(1, 6):
            mask = _df_clean['score'] == score
            count = _df_clean.loc[mask, 'preproc_text'].str.contains(pattern, regex=True).sum()
            freq_data.append({'term': term, 'score': score, 'count': count})
            
    freq_df = pd.DataFrame(freq_data)
    pivot_df = freq_df.pivot(index='term', columns='score', values='count').fillna(0)
    
    # Bad terms: 1-star count > 5-star count
    bad_terms = pivot_df[pivot_df[1] > pivot_df[5]].copy()
    bad_terms['delta'] = bad_terms[1] - bad_terms[5]
    
    # Good terms: (4+5) > (1+2)
    pivot_df["high"] = pivot_df[4] + pivot_df[5]
    pivot_df["low"] = pivot_df[1] + pivot_df[2]
    good_terms = pivot_df[pivot_df["high"] > pivot_df["low"]].copy()
    good_terms['delta'] = good_terms['high'] - good_terms['low']
    
    return bad_terms.sort_values('delta', ascending=False), good_terms.sort_values('delta', ascending=False)

# 4. Version Analysis
# Removed @st.cache_data to ensure this re-runs for each new analysis
def analyze_versions(_df_clean):
    results = {}
    current_year = pd.Timestamp.now().year
    
    reviews_this_year = _df_clean[_df_clean['at'].dt.year == current_year]
    if not reviews_this_year.empty:
        results['most_common_this_year'] = reviews_this_year['appVersion'].value_counts().idxmax()
    
    last_date = _df_clean['at'].max()
    four_months_ago = last_date - pd.DateOffset(months=4)
    recent_reviews = _df_clean[_df_clean['at'] >= four_months_ago]
    if not recent_reviews.empty:
        results['most_common_recent'] = recent_reviews['appVersion'].value_counts().idxmax()
        
    df_valid_versions = _df_clean.dropna(subset=['appVersion']).copy()
    if not df_valid_versions.empty:
        df_valid_versions['version_obj'] = df_valid_versions['appVersion'].apply(lambda v: version.parse(str(v)))
        results['latest_version'] = df_valid_versions.loc[df_valid_versions['version_obj'].idxmax(), 'appVersion']
    
    return results

# 5. Clustering
def summarize_texts(texts, sentences_count=2):
    """Summarizes text, ensuring NLTK 'punkt' is available."""
    # Robust check for NLTK 'punkt' tokenizer
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    if not texts: return "No content available for summary."
    text_data = " ".join(texts)[:20000] 
    try:
        parser = PlaintextParser.from_string(text_data, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return " ".join(str(s) for s in summary)
    except Exception:
        # Silently fail and return a snippet instead of showing a warning to the user
        return (text_data[:300] + '...') if len(text_data) > 300 else text_data


def top_keywords_per_cluster(X, labels, vectorizer, top_n=8):
    terms = vectorizer.get_feature_names_out()
    keywords = {}
    for i in np.unique(labels):
        mask = labels == i
        cluster_tfidf = X[mask].sum(axis=0)
        top_indices = np.asarray(cluster_tfidf).flatten().argsort()[-top_n:]
        keywords[i] = [terms[j] for j in top_indices][::-1]
    return keywords

# Removed @st.cache_data to ensure this re-runs for each new analysis
def run_clustering(_df, _bad_terms_list):
    if _df.empty or not _bad_terms_list:
        return {}

    # This status will be displayed within the calling function's context
    pattern = '|'.join([rf'\b{re.escape(term)}\b' for term in _bad_terms_list])
    mask = _df['preproc_text'].str.contains(pattern, regex=True, na=False)
    df_filtered = _df[mask].copy()

    if len(df_filtered) < 20:
        st.warning(f"Not enough reviews ({len(df_filtered)}) with problematic terms to perform clustering for this version group.")
        return {}
    
    st.info(f"Vectorizing {len(df_filtered)} reviews with TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000, min_df=5, ngram_range=(1,2))
    X = vectorizer.fit_transform(df_filtered['preproc_text'])
    
    st.info("Finding optimal number of clusters (k)...")
    k_range = range(4, min(13, len(df_filtered)//10))
    scores = {}
    if k_range:
        for k in k_range:
            km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = km.fit_predict(X)
            if len(np.unique(labels)) > 1:
                # Use a smaller sample_size for silhouette to speed it up
                sample_size = min(1000, X.shape[0] - 1)
                score = silhouette_score(X, labels, sample_size=sample_size)
                scores[k] = score
    
    optimal_k = max(scores, key=scores.get) if scores else 5
    
    st.info(f"Performing final clustering with k={optimal_k}...")
    km = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_filtered['cluster'] = km.fit_predict(X)
    
    st.info("Extracting keywords and generating summaries...")
    keywords = top_keywords_per_cluster(X, df_filtered['cluster'], vectorizer)
    
    results = {}
    for i in range(optimal_k):
        cluster_df = df_filtered[df_filtered['cluster'] == i]
        results[i] = {
            'count': len(cluster_df),
            'keywords': keywords.get(i, []),
            'summary': summarize_texts(cluster_df['content'].tolist()),
            'percentage': (len(cluster_df) / len(df_filtered)) * 100
        }
        
    return results

def display_clustering_results(results):
    if results:
        labels = [f"Cluster {i}" for i in results.keys()]
        sizes = [d['count'] for d in results.values()]
        
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        
        for i, data in sorted(results.items(), key=lambda item: item[1]['count'], reverse=True):
            with st.expander(f"**Cluster {i}: {data['keywords'][0]} & {data['keywords'][1]}** ({data['count']} reviews, {data['percentage']:.1f}%)"):
                st.markdown(f"**Keywords:** `{', '.join(data['keywords'])}`")
                st.markdown("**Summary:**")
                st.write(data['summary'])

def reset_analysis_state():
    """Clears all session state variables related to a previous analysis."""
    keys_to_clear = [
        'analysis_done', 'app_id', 'app_info', 'raw_df', 'df_clean', 
        'bad_terms', 'good_terms', 'version_info', 'clustering_results_latest', 
        'clustering_results_common', 'clustering_results_historical'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# --- Streamlit UI ---
st.title("🔬 Advanced Google Play NLP Analyzer")
st.markdown("From scraping to clustering, turn Google Play reviews into actionable insights. Based on the workflow from `nlp.py`.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    app_id_input = st.text_input(
        "Enter Google Play App ID",
        placeholder="com.google.android.apps.maps",
        on_change=reset_analysis_state
    )
    review_count = st.slider("Number of Reviews to Scrape", 100, 100000, 2000, 100)
    analyze_button = st.button("Analyze App ✨", use_container_width=True)

# --- Main App Logic ---
if analyze_button and app_id_input:
    app_id = app_id_input.strip()
    app_info = get_app_details(app_id)
    
    if app_info:
        raw_df = scrape_reviews(app_id, review_count)
        if not raw_df.empty:
            st.session_state.app_id = app_id
            st.session_state.app_info = app_info
            st.session_state.raw_df = raw_df
            st.session_state.analysis_done = True


if 'analysis_done' in st.session_state:
    # --- Display Header ---
    info = st.session_state.app_info
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.image(info['icon'], width=100)
    with col2:
        st.header(info['title'])
        st.markdown(f"**Developer:** {info['developer']} | **Genre:** {info['genre']} | **Installs:** {info['installs']}")

    # --- Analysis Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview & Raw Data", 
        "🧼 Preprocessing",
        "📈 Topic Analysis",
        "🆚 Version Analysis",
        "🧩 Issue Clustering",
        "📉 Trend Analysis"
    ])

    with tab1:
        st.subheader("Scraped Reviews Dataset")
        st.dataframe(st.session_state.raw_df, use_container_width=True)
        csv = st.session_state.raw_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Raw Data as CSV 💾", csv, f"{st.session_state.app_id}_reviews.csv", "text/csv")

    with tab2:
        st.subheader("NLP Preprocessing Pipeline")
        st.info("This step involves cleaning text, normalizing app versions, and applying spaCy for lemmatization and stopword removal.")
        if 'df_clean' not in st.session_state:
            st.session_state.df_clean = run_preprocessing(st.session_state.raw_df)
        st.dataframe(st.session_state.df_clean, use_container_width=True)

    with tab3:
        st.subheader("Good vs. Bad Term Analysis")
        st.markdown("Identifies terms that are disproportionately present in low-rated reviews compared to high-rated ones, and vice-versa.")
        
        with st.expander("💡 How is this calculated?"):
            st.markdown("""
            This analysis helps find the most impactful words by calculating a **"delta"** score for each term. This score measures how strongly a word is associated with a positive or negative experience.

            - **For 😠 Problematic Terms:** We focus on extreme dissatisfaction.
                - `delta = (Count in 1★ reviews) - (Count in 5★ reviews)`
                - A large positive delta means the term is a strong indicator of a critical issue that leads to 1-star reviews.

            - **For 😊 Positive Terms:** We look for general satisfaction.
                - `high = Count in 4★ + 5★ reviews`
                - `low = Count in 1★ + 2★ reviews`
                - `delta = high - low`
                - A large positive delta means the term is a strong indicator of what users love about the app.
            """)

        if 'df_clean' in st.session_state:
            if 'bad_terms' not in st.session_state:
                st.session_state.bad_terms, st.session_state.good_terms = analyze_term_frequency(st.session_state.df_clean)
            
            st.markdown("#### 😠 Problematic Terms (More Frequent in 1★ than 5★ Reviews)")
            fig, ax = plt.subplots(figsize=(12, 6))
            st.session_state.bad_terms.head(20).plot(kind='bar', y='delta', ax=ax, legend=False, color='salmon')
            ax.set_title("Top 20 Problematic Terms")
            ax.set_ylabel("Difference in Count (1★ minus 5★)")
            st.pyplot(fig)
            st.dataframe(st.session_state.bad_terms.head(20), use_container_width=True)

            st.markdown("#### 😊 Positive Terms (More Frequent in 4★+5★ than 1★+2★ Reviews)")
            fig, ax = plt.subplots(figsize=(12, 6))
            st.session_state.good_terms.head(20).plot(kind='bar', y='delta', ax=ax, legend=False, color='skyblue')
            ax.set_title("Top 20 Positive Terms")
            ax.set_ylabel("Difference in Count (High Ratings minus Low Ratings)")
            st.pyplot(fig)
            st.dataframe(st.session_state.good_terms.head(20), use_container_width=True)

    with tab4:
        st.subheader("Analysis by App Version")
        if 'df_clean' in st.session_state:
            if 'version_info' not in st.session_state:
                st.session_state.version_info = analyze_versions(st.session_state.df_clean)
            
            st.markdown("#### Key Version Identifiers")
            v_info = st.session_state.version_info
            m1, m2, m3 = st.columns(3)
            m1.metric("Latest Released Version", v_info.get('latest_version', 'N/A'))
            m2.metric("Most Reviewed (This Year)", v_info.get('most_common_this_year', 'N/A'))
            m3.metric("Most Reviewed (Last 4 Mos)", v_info.get('most_common_recent', 'N/A'))
            
            st.markdown("#### Review Count per Version")
            version_counts = st.session_state.df_clean['appVersion'].value_counts().sort_index(ascending=False)
            fig, ax = plt.subplots(figsize=(12, 5))
            version_counts.head(25).plot(kind='bar', ax=ax)
            ax.set_title("Number of Reviews per App Version (Top 25)")
            st.pyplot(fig)
        
    with tab5:
        st.subheader("Automated Clustering of Problematic Reviews")
        st.markdown("This uses K-Means clustering to group reviews containing 'Problematic Terms' into distinct issue categories for different version groups.")
        
        if 'bad_terms' in st.session_state and not st.session_state.bad_terms.empty:
            version_info = st.session_state.version_info
            df_clean = st.session_state.df_clean
            bad_terms_list = st.session_state.bad_terms.index.tolist()

            # --- LATEST VERSION CLUSTERING ---
            st.markdown(f"---")
            st.subheader(f"Latest Version Analysis: `{version_info.get('latest_version', 'N/A')}`")
            if 'clustering_results_latest' not in st.session_state:
                df_latest = df_clean[df_clean['appVersion'] == version_info.get('latest_version')]
                st.session_state.clustering_results_latest = run_clustering(df_latest, bad_terms_list)
            display_clustering_results(st.session_state.clustering_results_latest)

            # --- MOST COMMON THIS YEAR CLUSTERING ---
            st.markdown(f"---")
            st.subheader(f"Most Common Version This Year Analysis: `{version_info.get('most_common_this_year', 'N/A')}`")
            if 'clustering_results_common' not in st.session_state:
                df_common = df_clean[df_clean['appVersion'] == version_info.get('most_common_this_year')]
                st.session_state.clustering_results_common = run_clustering(df_common, bad_terms_list)
            display_clustering_results(st.session_state.clustering_results_common)

            # --- HISTORICAL VERSIONS CLUSTERING ---
            st.markdown(f"---")
            st.subheader("Historical Versions Analysis")
            if 'clustering_results_historical' not in st.session_state:
                key_versions = {version_info.get('latest_version'), version_info.get('most_common_this_year')}
                df_historical = df_clean[~df_clean['appVersion'].isin(key_versions)]
                st.session_state.clustering_results_historical = run_clustering(df_historical, bad_terms_list)
            display_clustering_results(st.session_state.clustering_results_historical)


    with tab6:
        st.subheader("Track Specific Terms Across Versions")
        if 'df_clean' in st.session_state:
            df_clean = st.session_state.df_clean
            
            st.markdown("#### Track Individual Terms (OR Logic)")
            st.markdown("Enter comma-separated terms to see how mentions of **any** of them have changed.")
            user_terms_or = st.text_input("Enter terms (e.g., login, crash, slow)", "login, slow", key="or_terms")
            
            if st.button("Plot OR Trends"):
                terms_list = [t.strip().lower() for t in user_terms_or.split(',') if t.strip()]
                if terms_list:
                    for term in terms_list:
                        mask = df_clean['preproc_text'].str.contains(rf'\b{re.escape(term)}\b', regex=True, na=False)
                        df_term = df_clean[mask]
                        
                        counts_1star = df_term[df_term['score'] == 1].groupby('appVersion').size()
                        counts_5star = df_term[df_term['score'] == 5].groupby('appVersion').size()
                        
                        df_plot = pd.DataFrame({'1-star': counts_1star, '5-star': counts_5star}).fillna(0)
                        
                        if not df_plot.empty:
                            st.markdown(f"##### Trend for `{term}`")
                            st.line_chart(df_plot, color=["#FF4B4B", "#2ECC71"])
                        else:
                            st.warning(f"No reviews found containing the term '{term}'..")
                else:
                    st.warning("Please enter at least one term for OR logic.")

            st.markdown("---")
            st.markdown("#### Track Co-occurring Terms (AND Logic)")
            st.markdown("Enter terms separated by 'and' to see how mentions of reviews containing **all** of them have changed.")
            user_terms_and = st.text_input("Enter terms (e.g., login and slow)", "payment and failed", key="and_terms")

            if st.button("Plot AND Trends"):
                terms_list = [t.strip().lower() for t in user_terms_and.split('and') if t.strip()]
                if len(terms_list) > 1:
                    mask = pd.Series(True, index=df_clean.index)
                    for term in terms_list:
                        mask &= df_clean['preproc_text'].str.contains(rf'\b{re.escape(term)}\b', regex=True, na=False)
                    
                    df_filtered = df_clean[mask]
                    combined_term = ' & '.join(terms_list)
                    
                    counts_1star = df_filtered[df_filtered['score'] == 1].groupby('appVersion').size()
                    counts_5star = df_filtered[df_filtered['score'] == 5].groupby('appVersion').size()

                    df_plot = pd.DataFrame({'1-star': counts_1star, '5-star': counts_5star}).fillna(0)

                    if not df_plot.empty:
                        st.markdown(f"##### Trend for `{combined_term}`")
                        st.line_chart(df_plot, color=["#FF4B4B", "#2ECC71"])
                    else:
                        st.warning(f"No reviews found containing all terms: '{combined_term}'.")
                else:
                    st.warning("Please enter at least two terms separated by 'and'.")


else:
    st.info("⬅️ Enter a Google Play App ID in the sidebar to begin analysis.")


