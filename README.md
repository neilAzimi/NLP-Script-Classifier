# NLP Movie Script Genre Classifier

## Task 1 – Text Preprocessing & Vectorization
### Dataset Creation
The Internet Movie Script Database was scraped using requests and BeautifulSoup. Script text for 1,259 movies was saved to a CSV, along with the titles, and associated genres. Once the initial data collection was complete, we used spaCy to remove non-English scripts, for a final dataset size of 1,211 entries.
### Text Preprocessing 
The script column of the database contains a lengthy string of unstructured text pertaining to the movie script and contains stage instructions, signs, and script elements that are uniquely formatted in each script and considered noise; these elements are first removed and then passed into an instance of a SpaCy NLP model for data cleaning and lemmatization. 

An extensive list of custom stop-words was built using the English language stop-word lists from NLTK and SpaCy; custom stop-words were also added after inspecting results following each iteration of running the preprocessing pipeline. The selected custom stop-words generally fall under the categories of script formatting terms, camera/directing instructions, character or action cues, dialogue exclamations, scene locations/objects, written numbers, extremely common violence descriptors, profanity, and symbols. 

The text is tokenized and filtered with this stop-word list, as well as removing tokens that are punctuation, numerical digits or whitespace. Tokens with a part-of-speech other than nouns, verbs, adjectives are also filtered out, as well as one-character and two-character words that are usually non-descriptive in the English language. Finally, the SpaCy model’s NER is used to recognize and remove tokens containing names of people; the small version of the model proved to be insufficient for filtering entity names and required an upgrade to the medium English model for adequate results. 

The main filtering function of with NER was parallelized to process in batches across multiple cores due to initially extensive runtimes on a single processing core. 

### TF-IDF Vectorization

TF-IDF vectorization was implemented dynamically as part of a pipeline when training our classification models. The parameters min_df and max_df were chosen based on the final model performance using GridSearchCV.

### Doc2Vec Vectorization



### LDA Vectorization & Topic Modeling

## Task 2 - Classification Model

### Classification  Methodologies

#### TF-IDF Method

#### Doc2Vec Method

#### LDA Method

### Comparison of Classification Results by Vectorization Method

## Task 3 - Description of Dashboard
The Movie Genre Classification Dashboard is a comprehensive web application built with Dash and Bootstrap that provides interactive visualization and analysis of movie genre classification results. Here's a structured description:

Overview:
The dashboard serves as an analytical tool for exploring and understanding movie genre classification using various machine learning approaches. It integrates multiple datasets and models to provide insights into genre prediction and classification performance.

Key Features:

1. Dataset Overview
   - Displays basic dataset statistics and a preview of the movie data
   - Shows total number of movies and available features

2. Model Comparison
   - Visualizes performance metrics across different classification methods (Doc2Vec, TF-IDF, LDA)
   - Presents F1-Macro scores for model comparison
   - Shows genre wise F1 scores to understand classification performance per genre

3. Genre Classification Visualization
   - Interactive SVD (Singular Value Decomposition) visualization
   - Allows filtering by specific genres
   - Displays movie distribution in reduced dimensionality space

4. Movie Classification
   - Interactive interface for testing movie descriptions
   - Supports multiple classification methods
   - Provides probability scores for different genres
   - Shows top predicted genres for input text

5. LDA Topic Analysis
   - Visualizes topic distribution across the dataset
   - Shows topic importance by genre
   - Provides insights into latent themes in movie descriptions

6. Movie Explorer
   - Interactive search and filter functionality
   - Allows filtering by multiple genres
   - Displays detailed movie information in a sortable table

7. TF-IDF Analysis
   - Shows classification metrics (precision, recall, F1-score) by genre
   - Allows filtering by specific genres
   - Visualizes model performance across different metrics

Technical Implementation:
- Built using Dash framework with Bootstrap styling
- Integrates multiple machine learning models (Doc2Vec, TF-IDF, LDA)
- Uses Plotly for interactive visualizations
- Implements responsive design for better user experience
- Includes data preprocessing and model pipeline management

This dashboard provides a comprehensive tool for analyzing and understanding movie genre classification, making it valuable for both technical and non-technical users interested in movie genre analysis and machine learning applications in text classification.


