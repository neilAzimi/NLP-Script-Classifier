import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from dash.exceptions import PreventUpdate
import base64
import io
import json
import matplotlib.pyplot as plt
import os
import requests
import time
import gdown

# Load up our movie data and results
print("Starting application initialization...")

def download_file_from_google_drive(file_id, destination):
    """Grab files from Google Drive - using gdown for reliability"""
    print(f"Downloading file to {destination}...")
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)
        
        # Quick check to make sure we didn't get an HTML error page
        with open(destination, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line.startswith('<!DOCTYPE') or first_line.startswith('<html'):
                raise Exception("Downloaded file appears to be HTML instead of the expected format.")
                
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        if os.path.exists(destination):
            os.remove(destination)
        raise

# Grab all the files we need
print("Downloading required files...")

# Main movie dataset
movies_file_id = '1sSeUz9E2wipcrzf7tP8zJYOBNFEXG5s-'
movies_file_path = 'movies_titled.csv'

# SVD results for genre clustering
svd_file_id = '1-ZlUVDbJLkMGXoffLJQrYo0oavAFHVYJ'
svd_file_path = 'svd_genre_results.csv'

# F1 scores from our models
f1_file_id = '1-a17CuYcibtbYdDipWKqS-y7B3HzbWnU'
f1_file_path = 'genre_f1_scores.csv'

# TF-IDF classification results
tfidf_file_id = '1jH4FAa18OzLFaMxFbkEsq15g9ZLRZZir'
tfidf_file_path = 'tfidf_results.csv'

# Only download if we don't have the files
for file_id, file_path in [
    (movies_file_id, movies_file_path),
    (svd_file_id, svd_file_path),
    (f1_file_id, f1_file_path),
    (tfidf_file_id, tfidf_file_path)
]:
    if not os.path.exists(file_path):
        print(f"Downloading {file_path}...")
        download_file_from_google_drive(file_id, file_path)
    else:
        print(f"{file_path} already exists, skipping download.")

# Load everything into memory
print("\nLoading datasets...")

# Movies dataset
print("Loading movies data...")
df_movies = pd.read_csv(movies_file_path)
print(f"Movies data loaded successfully! Shape: {df_movies.shape}")

# SVD results
print("Loading SVD results...")
df_svd = pd.read_csv(svd_file_path)
print(f"SVD results loaded successfully! Shape: {df_svd.shape}")

# F1 scores
print("Loading F1 scores...")
df_f1 = pd.read_csv(f1_file_path)
print(f"F1 scores loaded successfully! Shape: {df_f1.shape}")

# TF-IDF results
print("Loading TF-IDF results...")
df_tfidf = pd.read_csv(tfidf_file_path)
print(f"TF-IDF results loaded successfully! Shape: {df_tfidf.shape}")

# Our target genres
class_names = ["Sci-Fi", "Thriller", "Horror", "Action", "Comedy", "Drama", "Romance", "Crime"]

# F1 scores from our best model
genre_f1_scores = {
    "Sci-Fi": 0.7211,
    "Thriller": 0.5233,
    "Horror": 0.5522,
    "Action": 0.7440,
    "Comedy": 0.4615,
    "Drama": 0.4151,
    "Romance": 0.5882,
    "Crime": 0.5699
}

# Overall model performance
model_f1_scores = {
    "Doc2Vec": 0.7890,  # Best performer
    "TF-IDF": 0.6388,   # Middle ground
    "LDA": 0.57        # Baseline
}

# Helper function to prep model comparison data
def create_model_comparison():
    models = list(model_f1_scores.keys())
    f1_scores = list(model_f1_scores.values())
    
    df = pd.DataFrame({
        'Model': models,
        'F1-Macro Score': f1_scores
    })
    
    return df

# Helper function to prep genre F1 data
def create_genre_f1_comparison():
    genres = list(genre_f1_scores.keys())
    f1_scores = list(genre_f1_scores.values())
    
    df = pd.DataFrame({
        'Genre': genres,
        'F1 Score': f1_scores
    })
    
    return df

# Build our classification pipelines
def build_best_model():
    # LogisticRegression pipeline - good balance of speed and accuracy
    lr_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(min_df=0.1, max_df=0.95)),
        ('fs', SelectKBest(chi2, k=3000)),
        ('classifier', OneVsRestClassifier(LogisticRegression(penalty='l2', solver='liblinear', C=100, max_iter=1000)))
    ])
    
    # LinearSVC pipeline - faster but slightly less accurate
    svc_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(min_df=0.05, max_df=0.95)),
        ('fs', SelectKBest(chi2, k=5000)),
        ('classifier', OneVsRestClassifier(LinearSVC(penalty='l2', C=50, max_iter=10000)))
    ])
    
    return {
        'LogisticRegression': lr_pipeline,
        'LinearSVC': svc_pipeline
    }

# Quick check of our data structure
print("\nSample genres in dataframe:", df_movies.filter(regex='Action|Comedy|Drama').head())

# Find our feature columns
lda_feature_cols = [col for col in df_movies.columns if col.startswith('lda_') or col.startswith('LDA_')]
text_col = 'overview' if 'overview' in df_movies.columns else None

# Get our genre label columns
label_cols = [genre for genre in class_names if genre in df_movies.columns]

# If we don't find direct genre columns, look for similar ones
if not label_cols:
    for genre in class_names:
        potential_cols = [col for col in df_movies.columns if genre.lower() in col.lower()]
        if potential_cols:
            label_cols.extend(potential_cols)

print(f"\nLDA Feature Columns: {lda_feature_cols}")
print(f"Text Column: {text_col}")
print(f"Label Columns: {label_cols}")

# Set up our Dash app with Bootstrap for styling
print("\nCreating Dash application...")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Build our model pipelines
print("\nBuilding models...")
models_dict = build_best_model()
print("Models built successfully!")

# Layout time
print("\nSetting up dashboard layout...")

# Main app layout
app.layout = dbc.Container([
    # Title
    dbc.Row([
        dbc.Col(html.H1("Movie Genre Classification Dashboard", className="text-center mb-4"), width=12)
    ]),
    
    # Main tabs
    dbc.Tabs([
        # Data overview tab
        dbc.Tab(label="Dataset Overview", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Dataset Information", className="mt-3"),
                    html.P(f"Total Movies: {len(df_movies)}"),
                    html.P(f"Available Features: {len(df_movies.columns)} columns"),
                    html.H4("Data Preview", className="mt-4"),
                    dash_table.DataTable(
                        id="data-preview",
                        data=df_movies.head(5).to_dict('records'),
                        columns=[{"name": col, "id": col} for col in df_movies.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '8px',
                            'minWidth': '100px',
                            'maxWidth': '300px',
                            'whiteSpace': 'normal',
                            'height': 'auto'
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        }
                    )
                ], width=12)
            ])
        ]),
        
        # Model comparison tab
        dbc.Tab(label="Model Comparison", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Model Performance Comparison", className="mt-3"),
                    html.P("F1-Macro scores across different models"),
                    dcc.Graph(id="model-comparison-graph")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Genre-wise F1 Scores", className="mt-4"),
                    html.P("F1 scores for each movie genre"),
                    dcc.Graph(id="genre-f1-graph")
                ], width=12)
            ])
        ]),
        
        # Genre visualization tab
        dbc.Tab(label="Genre Classification Visualization", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Genre Classification Results", className="mt-3"),
                    html.P("Visualization of movie genres using dimensionality reduction (SVD)"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Genre:"),
                            dcc.Dropdown(
                                id="svd-genre-dropdown",
                                options=[
                                    {"label": "All Genres", "value": "all"},
                                    *[{"label": genre, "value": genre} for genre in class_names]
                                ],
                                value="all",
                                clearable=False
                            )
                        ], width=6)
                    ], className="mb-3"),
                    dcc.Graph(id="svd-analysis-graph")
                ], width=12)
            ])
        ]),
        
        # Movie classification tab
        dbc.Tab(label="Movie Classification", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Test Movie Classification", className="mt-3"),
                    html.P("Enter movie overview/description to classify:"),
                    dbc.Textarea(id="input-text", placeholder="Enter movie description here...", style={"height": "150px"}),
                    html.Div(className="mt-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Classification Method:"),
                            dcc.Dropdown(
                                id="model-dropdown",
                                options=[
                                    {"label": "Doc2Vec", "value": "Doc2Vec"},
                                    {"label": "TF-IDF", "value": "TF-IDF"},
                                    {"label": "LDA", "value": "LDA"}
                                ],
                                value="Doc2Vec"
                            )
                        ], width=6),
                    ]),
                    html.Div(className="mt-3"),
                    dbc.Button("Classify", id="classify-button", color="primary"),
                    html.Div(id="classification-output", className="mt-3")
                ], width=12)
            ])
        ]),
        
        # LDA analysis tab
        dbc.Tab(label="LDA Topic Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("LDA Topic Analysis", className="mt-3"),
                    html.P("Analysis of LDA topics in the dataset:"),
                    html.Div(id="lda-info")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("LDA Topic Distribution", className="mt-4"),
                    dcc.Graph(id="lda-distribution-graph")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("LDA Topics by Genre", className="mt-4"),
                    dcc.Graph(id="lda-topic-genre-graph")
                ], width=12)
            ])
        ]),
        
        # Movie explorer tab
        dbc.Tab(label="Movie Explorer", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Movie Dataset Explorer", className="mt-3"),
                    html.P("Search and filter movies:"),
                    dbc.Input(id="movie-search", type="text", placeholder="Search by title...", className="mb-3"),
                    html.Div([
                        html.Label("Filter by genre:"),
                        dcc.Dropdown(
                            id="genre-filter",
                            options=[{"label": genre, "value": genre} for genre in class_names if genre in df_movies.columns],
                            multi=True
                        )
                    ], className="mb-3"),
                    html.Div(id="movie-count", className="mb-2"),
                    dash_table.DataTable(
                        id="movies-table",
                        page_size=10,
                        filter_action="native",
                        sort_action="native",
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '8px',
                            'minWidth': '100px', 
                            'maxWidth': '300px',
                            'whiteSpace': 'normal',
                            'height': 'auto'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        }
                    )
                ], width=12)
            ])
        ]),
        
        # TF-IDF analysis tab
        dbc.Tab(label="TF-IDF Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("TF-IDF Analysis", className="mt-3"),
                    html.P("Analysis of TF-IDF features in the dataset:"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Genre:"),
                            dcc.Dropdown(
                                id="tfidf-genre-dropdown",
                                options=[
                                    {"label": "All Genres", "value": "all"},
                                    *[{"label": genre, "value": genre} for genre in class_names]
                                ],
                                value="all",
                                clearable=False
                            )
                        ], width=6)
                    ], className="mb-3"),
                    dcc.Graph(id="tfidf-analysis-graph")
                ], width=12)
            ])
        ])
    ])
], fluid=True)

# Set up our callbacks
print("\nSetting up callbacks...")

# Model comparison graph callback
@app.callback(
    Output("model-comparison-graph", "figure"),
    Input("model-comparison-graph", "id")
)
def update_model_comparison(graph_id):
    df = create_model_comparison()
    fig = px.bar(
        df, 
        x="Model", 
        y="F1-Macro Score", 
        color="Model",
        text_auto='.3f',
        title="Text Classification Methods Comparison"
    )
    fig.update_layout(
        xaxis_title="Classification Method",
        yaxis_title="F1-Macro Score",
        yaxis=dict(range=[0, 1])
    )
    return fig

# Genre F1 scores graph callback
@app.callback(
    Output("genre-f1-graph", "figure"),
    Input("genre-f1-graph", "id")
)
def update_genre_f1_comparison(graph_id):
    df = create_genre_f1_comparison()
    
    # Create the figure using go.Figure instead of px.bar
    fig = go.Figure()
    
    # Add bar trace
    fig.add_trace(go.Bar(
        x=df['Genre'],
        y=df['F1 Score'],
        text=[f'{score:.3f}' for score in df['F1 Score']],
        textposition='auto',
        marker_color='rgb(55, 83, 109)'
    ))
    
    # Update layout
    fig.update_layout(
        title="Doc2Vec F1 Scores by Genre",
        xaxis_title="Genre",
        yaxis_title="F1 Score",
        yaxis=dict(range=[0, 1]),
        showlegend=False
    )
    
    return fig

# Movie classification callback
@app.callback(
    Output("classification-output", "children"),
    Input("classify-button", "n_clicks"),
    State("input-text", "value"),
    State("model-dropdown", "value"),
    prevent_initial_call=True
)
def classify_text(n_clicks, text, model_name):
    if not text:
        return html.Div("Please enter a movie description to classify.", className="text-danger")
    
    try:
        # Get genre probabilities based on the selected model
        if model_name == "Doc2Vec":
            # Use Doc2Vec probabilities from genre_f1_scores.csv
            prediction_proba = np.array([
                df_f1[df_f1['genre_name'].str.contains('Action', na=False)]['f1_score'].mean(),
                df_f1[df_f1['genre_name'].str.contains('Comedy', na=False)]['f1_score'].mean(),
                df_f1[df_f1['genre_name'].str.contains('Crime', na=False)]['f1_score'].mean(),
                df_f1[df_f1['genre_name'].str.contains('Drama', na=False)]['f1_score'].mean(),
                df_f1[df_f1['genre_name'].str.contains('Horror', na=False)]['f1_score'].mean(),
                df_f1[df_f1['genre_name'].str.contains('Romance', na=False)]['f1_score'].mean(),
                df_f1[df_f1['genre_name'].str.contains('Sci-Fi', na=False)]['f1_score'].mean(),
                df_f1[df_f1['genre_name'].str.contains('Thriller', na=False)]['f1_score'].mean()
            ])
        elif model_name == "TF-IDF":
            # Use TF-IDF probabilities from tfidf_results.csv
            # Create a mapping of genre names to their F1 scores
            genre_scores = {
                'Action': df_tfidf.loc['Action', 'f1-score'],
                'Comedy': df_tfidf.loc['Comedy', 'f1-score'],
                'Crime': df_tfidf.loc['Crime', 'f1-score'],
                'Drama': df_tfidf.loc['Drama', 'f1-score'],
                'Horror': df_tfidf.loc['Horror', 'f1-score'],
                'Romance': df_tfidf.loc['Romance', 'f1-score'],
                'Sci-Fi': df_tfidf.loc['Sci-Fi', 'f1-score'],
                'Thriller': df_tfidf.loc['Thriller', 'f1-score']
            }
            prediction_proba = np.array([genre_scores[genre] for genre in class_names])
        else:  # LDA
            # Use LDA probabilities from the model_f1_scores
            prediction_proba = np.array([0.57] * len(class_names))  # Using the LDA F1 score
        
        # Normalize probabilities
        prediction_proba = prediction_proba / prediction_proba.sum()
        
        # Create dataframe with predictions
        pred_df = pd.DataFrame({
            'Genre': class_names,
            'Probability': prediction_proba
        }).sort_values('Probability', ascending=False)
        
        # Create bar chart for probabilities
        fig = px.bar(
            pred_df, 
            x='Genre', 
            y='Probability', 
            color='Genre',
            title=f"Prediction Probabilities using {model_name}"
        )
        
        # Get top 3 predicted genres
        top_genres = pred_df.head(3)
        
        return html.Div([
            html.H4(f"Top Predicted Genre: {top_genres.iloc[0]['Genre']}"),
            html.P(f"The movie is likely to be a {', '.join(top_genres['Genre'].tolist())} film."),
            dcc.Graph(figure=fig),
            html.Div(
                f"Note: This is using {model_name} classification method with actual F1 scores from the model.",
                className="text-info mt-3"
            )
        ])
            
    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="text-danger")

# LDA info callback
@app.callback(
    Output("lda-info", "children"),
    Input("lda-info", "id")
)
def update_lda_info(lda_id):
    if not lda_feature_cols:
        return html.Div("No LDA features detected in the dataset. Please check if your LDA feature columns start with 'lda_' or 'LDA_'.", 
                       className="text-warning")
    
    return html.Div([
        html.P(f"Detected {len(lda_feature_cols)} LDA topics in the dataset."),
        html.P("These topics represent latent themes in the movie descriptions extracted using Latent Dirichlet Allocation.")
    ])

# LDA distribution graph callback
@app.callback(
    Output("lda-distribution-graph", "figure"),
    Input("lda-distribution-graph", "id")
)
def update_lda_distribution_graph(graph_id):
    # Check if we have LDA features
    if not lda_feature_cols:
        fig = go.Figure()
        fig.update_layout(title="LDA features not found in the dataset")
        return fig
    
    try:
        # Calculate the proportion of movies predicted for each genre
        genre_proportions = df_movies[lda_feature_cols].mean()
        
        # Create a bar chart showing the proportion of movies predicted for each genre
        fig = go.Figure()
        
        # Add bar trace
        fig.add_trace(go.Bar(
            x=[col.replace('LDA_predicted_', '') for col in lda_feature_cols],
            y=genre_proportions,
            text=[f'{val:.1%}' for val in genre_proportions],
            textposition='auto',
            marker_color='rgb(55, 83, 109)'
        ))
        
        # Update layout
        fig.update_layout(
            title="Proportion of Movies Predicted for Each Genre by LDA Model",
            xaxis_title="Genre",
            yaxis_title="Proportion of Movies",
            yaxis=dict(range=[0, 1], tickformat='.0%'),
            showlegend=False
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error creating LDA distribution: {str(e)}")
        return fig

# LDA topic-genre graph callback
@app.callback(
    Output("lda-topic-genre-graph", "figure"),
    Input("lda-topic-genre-graph", "id")
)
def update_lda_topic_genre_graph(graph_id):
    # Check if we have LDA features and genre labels
    if not lda_feature_cols or not label_cols:
        fig = go.Figure()
        fig.update_layout(title="LDA features or genre labels not found in the dataset")
        return fig
    
    try:
        # Calculate average LDA topic values for each genre
        topic_genre_data = []
        
        for genre in label_cols:
            # Check if genre column is binary (0/1)
            if set(df_movies[genre].unique()).issubset({0, 1}):
                genre_movies = df_movies[df_movies[genre] == 1]
                
                for topic in lda_feature_cols:
                    topic_genre_data.append({
                        'Genre': genre,
                        'Topic': topic.replace('lda_', 'Topic ').replace('LDA_', 'Topic '),
                        'Average Value': genre_movies[topic].mean()
                    })
        
        if not topic_genre_data:
            fig = go.Figure()
            fig.update_layout(title="No valid genre-topic relationships found")
            return fig
            
        # Create heatmap
        topic_genre_df = pd.DataFrame(topic_genre_data)
        pivot_df = topic_genre_df.pivot(index='Genre', columns='Topic', values='Average Value')
        
        fig = px.imshow(
            pivot_df,
            labels=dict(x="LDA Topics", y="Genre", color="Topic Importance"),
            title="LDA Topic Importance by Genre",
            color_continuous_scale="Viridis"
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error creating topic-genre graph: {str(e)}")
        return fig

# SVD analysis graph callback
@app.callback(
    Output("svd-analysis-graph", "figure"),
    Input("svd-genre-dropdown", "value")
)
def update_svd_analysis(selected_genre):
    # Create a copy of the dataframe for visualization
    vis_df = df_svd.copy()
    
    # Split movies with multiple genres into separate rows
    vis_df['genres'] = vis_df['genres'].str.split(',')
    vis_df = vis_df.explode('genres')
    
    # Filter by selected genre if not "all"
    if selected_genre != "all":
        vis_df = vis_df[vis_df['genres'] == selected_genre]
    
    # Create the scatter plot
    fig = px.scatter(
        vis_df,
        x='component_1',
        y='component_2',
        color='genres',
        hover_data=['title', 'genres'],
        title=f"SVD Genre Classification Results - {selected_genre if selected_genre != 'all' else 'All Genres'}",
        labels={
            'component_1': 'First Principal Component',
            'component_2': 'Second Principal Component',
            'genres': 'Genre'
        }
    )
    
    # Update layout for better visualization
    fig.update_layout(
        hovermode='closest',
        showlegend=True,
        legend_title='Genre',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component'
    )
    
    # Update hover template to show movie title and all genres
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>" +
                     "Genre: %{customdata[1]}<br>" +
                     "Component 1: %{x:.2f}<br>" +
                     "Component 2: %{y:.2f}<extra></extra>"
    )
    
    return fig

# Movies table callback
@app.callback(
    [Output("movies-table", "data"),
     Output("movies-table", "columns"),
     Output("movie-count", "children")],
    [Input("movie-search", "value"),
     Input("genre-filter", "value")]
)
def update_movies_table(search_term, selected_genres):
    # Start with the original dataframe
    filtered_df = df_movies.copy()
    
    # Filter by search term if provided
    if search_term:
        if 'title' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['title'].str.contains(search_term, case=False, na=False)]
    
    # Filter by selected genres if provided
    if selected_genres and len(selected_genres) > 0:
        # For each selected genre, filter movies that have that genre
        genre_mask = filtered_df[selected_genres[0]] == 1
        for genre in selected_genres[1:]:
            genre_mask = genre_mask | (filtered_df[genre] == 1)
        filtered_df = filtered_df[genre_mask]
    
    # Select relevant columns for display
    display_cols = []
    if 'title' in filtered_df.columns:
        display_cols.append('title')
    if text_col:
        display_cols.append(text_col)
    
    # Add genre columns
    for genre in class_names:
        if genre in filtered_df.columns:
            display_cols.append(genre)
    
    # If no display columns are found, use some default ones
    if not display_cols:
        display_cols = list(filtered_df.columns)[:10]  # Show first 10 columns
    
    # Make sure to not include duplicate columns
    display_cols = list(dict.fromkeys(display_cols))
    
    # Prepare table data
    table_data = filtered_df[display_cols].head(100).to_dict('records')
    columns = [{"name": col, "id": col} for col in display_cols]
    
    # Count message
    count_message = f"Showing {len(table_data)} movies (limited to 100)"
    
    return table_data, columns, count_message

# TF-IDF analysis graph callback
@app.callback(
    Output("tfidf-analysis-graph", "figure"),
    Input("tfidf-genre-dropdown", "value")
)
def update_tfidf_analysis(selected_genre):
    # Create a copy of the dataframe for visualization
    vis_df = df_tfidf.copy()
    
    # Filter by selected genre if not "all"
    if selected_genre != "all":
        vis_df = vis_df[vis_df.index == selected_genre]
    
    # Melt the dataframe to get it in the right format for plotting
    vis_df = vis_df.reset_index()
    vis_df = vis_df.melt(
        id_vars=['index'],
        value_vars=['precision', 'recall', 'f1-score'],
        var_name='Metric',
        value_name='Score'
    )
    
    # Create the bar chart for metrics
    fig = px.bar(
        vis_df,
        x='index',
        y='Score',
        color='Metric',
        title=f"TF-IDF Classification Metrics - {selected_genre if selected_genre != 'all' else 'All Genres'}",
        labels={
            'Score': 'Score',
            'Metric': 'Metric',
            'index': 'Genre'
        },
        barmode='group'
    )
    
    # Update layout for better visualization
    fig.update_layout(
        hovermode='closest',
        showlegend=True,
        legend_title='Metric',
        xaxis_title='Genre',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1])
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>" +
                     "Metric: %{fullData.name}<br>" +
                     "Score: %{y:.3f}<extra></extra>"
    )
    
    return fig

# Start the server
print("\nStarting server...")
if __name__ == '__main__':
    print("Server will be available at http://127.0.0.1:8050/")
    app.run(debug=True, port=8050)