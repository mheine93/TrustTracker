# Loading packages
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')

# Configuring the web page and setting the page title and icon
st.set_page_config(
  page_title='TrustTracker',
  page_icon='👌',
  initial_sidebar_state='expanded')
  
# Function to load the dataset
@st.experimental_memo
def load_data():
    # Define the file path
    file_path = 'https://raw.githubusercontent.com/MikkelONielsen/TrustTracker/main/trust_pilot_reviews_data_2022_06.csv'
    
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(file_path)
    
    return df

# Loading the data using the defined function
df = load_data()

# Loading the model
ros_pipe_rf = pickle.load(open('data/model (4).pkl', 'rb'))

# Defining text preprocessing function for individual review predictor
def text_prepro(texts: pd.Series) -> list:
  # Creating a container for the cleaned texts
  clean_container = []
  # Using spacy's nlp.pipe to preprocess the text
  for doc in nlp.pipe(texts):
    # Extracting lemmatized tokens that are not punctuations, stopwords or non-alphabetic characters
    words = [words.lemma_.lower() for words in doc
            if words.is_alpha and not words.is_stop and not words.is_punct]

    # Adding the cleaned tokens to the container "clean_container"
    clean_container.append(" ".join(words))

  return clean_container

# Defining the ML function for the individual review predictor
def predict(placetext):
  text_ready = []
  text_ready = text_prepro(pd.Series(placetext))
  result = ros_pipe_rf.predict(text_ready)
  if result == 0:
    return "Negative sentiment"
  if result == 1:
    return "Positive sentiment"

# Defining categories to add aspect-based sentiment analysis
categories = {
    "Price": [
        "price", "cost", "expensive", "cheap", "value", "pay", "affordable",
        "pricey", "budget", "charge", "fee", "pricing", "rate", "worth", "economical"
    ],
    "Delivery": [
        "deliver", "delivery", "shipping", "dispatch", "courier", "ship", "transit",
        "postage", "mail", "shipment", "logistics", "transport", "send", "carrier", "parcel"
    ],
    "Quality": [
        "quality", "material", "build", "standard", "durability", "craftsmanship",
        "workmanship", "texture", "construction", "condition", "grade", "caliber",
        "integrity", "excellence", "reliability", "sturdiness", "performance"
    ],
    "Service": [
        "service", "support", "assistance", "help", "customer service", "care", "response",
        "satisfaction", "experience", "professionalism", "expertise", "efficiency",
        "friendliness", "availability", "flexibility", "reliability"
    ]
}

# Lemmatizing the keywords
def lemmatize_keywords(categories):
    lemmatized_categories = {}
    for category, keywords in categories.items():
        lemmatized_keywords = [nlp(keyword)[0].lemma_ for keyword in keywords]
        lemmatized_categories[category] = lemmatized_keywords
    return lemmatized_categories

# Defining function to categories the reviews according to the categories
list_lab = []
def categorize_review(text_review):
    lemmatized_review = " ".join([token.lemma_ for token in nlp(text_review.lower())])
    for category, keywords in lemmatize_keywords(categories).items():
        if any(keyword in lemmatized_review for keyword in keywords):
          list_lab.append(category)
    return list_lab


# Creating the overall company performance features

# Creating a sentiment column
df['sentiment']=np.where(df['rating']>=4,1,0) # 1=positive, 0=negative

# Defining a function that returns the overall sentiment and the sentiment for each category for all the companies
def overall_and_category_sentiment(df):
    # Calculating the mean sentiment for each company
    overall_sentiment_means = df.groupby('name')['sentiment'].mean()

    # Calculating the mean sentiment for each category for each company
    category_sentiments = {}
    for category, keywords in categories.items():
        df_category = df[df['review_text'].str.contains('|'.join(keywords))]
        category_sentiments[category] = df_category.groupby('name')['sentiment'].mean()

    # Determine overall sentiment and category sentiment
    overall = overall_sentiment_means.apply(lambda x: 'Positive' if x >= 0.5 else 'Negative')
    category_sentiment_labels = {category: sentiments.apply(lambda x: 'Positive' if x >= 0.5 else 'Negative') for category, sentiments in category_sentiments.items()}

    return overall, category_sentiment_labels

# Applying the function to our DataFrame
overall_scores, category_scores = overall_and_category_sentiment(df)

# Defining the function for the predictor
def check_company_sentiment(company_name):
    if company_name not in overall_scores:
        return f"{company_name} not found in the data."
    sentiment_info = [f"Overall sentiment for {company_name}: {overall_scores[company_name]}"]
    for category, scores in category_scores.items():
        if company_name in scores:
            sentiment_info.append(f"{category} sentiment: {scores[company_name]}")
    return sentiment_info

# Creating the app

# Setting the title and adding text
st.title('TrustTracker 👌')
st.markdown('Welcome to TrustTracker! The application where you easily can check the quality, price, service and delivery of your favorite companies.')

# Creating tabs for the different features of the application
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['About', 'Individual Reviews', 'Overall Company Performance', 'Model performance', 'Dataset', 'Visualisations'])

with tab1:
  st.header("About the application")
  st.markdown('This application is divided into 6 tabs:')
  st.markdown('*   **About the application:** The first tab shows the current page, where you can get some information about the application. You can always return to this tab if necessary.')
  st.markdown('*   **Individual Reviews:** In the second tab, you can input a review and get an Aspect-Based Sentiment Analysis of the review that show if the review regards price, quality, service or delivery and if the review is positive or negative.')
  st.markdown('*   **Overall Company Performance:** The third tab contains an overall Aspect-Based Sentiment Analysis for a company. Here you can choose a company from the list and get an output telling you if consumers overall have a positive or negative sentiment regarding the company and a separate sentiment for each of the chosen categories.')
  st.markdown('*   **Model Performance:** The fourth tab explains how the underlying Machine Learning Model performs and how the classifier works.')
  st.markdown('*   **Dataset:** In the fifth tab, you can see the origin of the dataset that has been used to build the application.')
  st.markdown('*   **Visualisations:** Lastly, for those who are curious about the distribution of the dataset, the application includes an exploratory data analysis of the different variables.')
        
with tab2:
  st.header('Analyse Individual Reviews')
  st.markdown('This tab includes an Aspect-Based Sentiment Analysis for individual reviews. The classifier is built using TF-IDF and Random Forests.')
  review_txt = st.text_input('Enter your review here')
  if st.button('Analyse Aspect-Based Sentiment'):
    category = categorize_review(review_txt)
    sentiment = predict(review_txt)
    st.write(f'This review regards: {", ".join(category)}')
    st.write(f'The sentiment of the review is: {sentiment}')

with tab3:
  st.header('Analyse Overall Company Performance')
  st.markdown('This tab includes an overall Aspect-Based Sentiment Analysis for all the reviews of the companies in the list. This classifier is also built using TF-IDF and Random Forests.')

  # Adding a selectbox
  selected_company = st.selectbox('Select company:', df['name'].unique())

  # Adding an analyzing button
  if st.button('Analyse Overall Aspect-Based Sentiment of Selected Company'):
    result = check_company_sentiment(selected_company)
    st.write(result[0])

    # Defining a list of indices to display (1, 2, 3, 4 in this case)
    indices_to_display = [1, 2, 3, 4]

    # Looping through the indices and displaying only if they are in the list
    for index in indices_to_display:
        if index < len(result):
            st.write(result[index])

with tab4:
  st.header('Model performance')
  st.markdown('This tab is for those, who are curious about how the underlying classification model works. Scroll through the page to get more information.')

  # Confusion Matrix
  st.subheader("Confusion Matrix")
  st.markdown('In this confusion matrix, we can see how the Random Forest model has 789 True Positives and 814 True Negatives. It only has 7 False Positive predictions and 25 False Negative predictions, which shows that the classifier is performing quite well.')
  st.image('images/rfconfusionmatrix.png')

  # Yellowbrick FreqDistVisualizer
  st.subheader("Yellowbrick FreqDistVisualizer")
  st.markdown('Here we can see that service is an often occuring word over the whole corpus. We can also see that the reviews in general are positive, as the words great and good are in the top of the distribution.')
  st.image('images/svmyellowbrick.png')

  # LIME 
  st.subheader("LIME Text Explainer")
  st.markdown('Here we can see an example of the LIME Text Explainer, which shows that the words "cup" and "chicken" are dragging the review towards being negative.')
  st.image('images/lime.png')

  # WordClouds
  st.subheader("Word Clouds")
  st.markdown('')
  st.image('images/wordclouds.png')

with tab5:
  st.header('Dataset')
  st.markdown('The dataset has been loaded from the following link: https://www.kaggle.com/datasets/crawlfeeds/trustpilot-reviews-dataset.')
  # Display dataset overview
  st.subheader("Dataset Overview")
  st.dataframe(df.head())
  
with tab6:
  st.header('Visualisations')

  # Reviews by number of companies
  st.subheader("Reviews by Number of Companies")
  st.image('images/reviewsbycompanies.png')

  # Reviews by year
  st.subheader("Reviews by Year")
  st.image('images/reviewsbyyear.png')

  # Reviews by month in 2022
  st.subheader("Reviews by Month in 2022")
  st.image('images/reviewsbymonth.png')
  
  # Reviews by rating
  st.subheader("Reviews by Rating")
  st.image('images/reviewsbyrating.png')

  # Reviews by user
  st.subheader("Reviews by Consumer Name")
  st.image('images/reviewsbyauthor.png')
