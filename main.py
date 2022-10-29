from itertools import groupby
from textblob import TextBlob
import pandas as pd
import streamlit as st 
import cleantext
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 




st.header('Sentiment Analysis')
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', round(blob.sentiment.polarity,2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))


    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,
                                 stopwords=True ,lowercase=True ,numbers=True , punct=True))

with st.expander('Analyze CSV'):
    uploadfile = st.file_uploader('Upload file')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

#
    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

#
    if uploadfile:
        df = pd.read_csv(uploadfile)

        df['score'] = df['tweet'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head(10))
        
    

        most_positive = df[df.score > 0.8].tweet.head(10)
        most_negative = df[df.score < -0.8].tweet.head(10)
        
        option = ["most negative","most positive"]
        option_selected = st.selectbox("what would u like",
                                       options= option)
        if option_selected == "most negative":
            st.write(most_negative)
        elif option_selected == "most positive":
            st.write(most_positive)

        fig = plt.figure(figsize=(19, 7))
        sns.histplot(data=df, x="score")    
        plt.title("Sentiment Analysis")
        st.pyplot(fig)
        


        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )