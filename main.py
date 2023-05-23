import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval
from streamlit_lottie import st_lottie

hide_menu="""
<style>
#MainMenu {
    visibility:hidden;
}

footer{
    visibility:hidden;   
}
</style>
"""
hotel = pickle.load(open('hotels.pkl','rb'))
hotel_list = hotel['city'].values
unique_cities = list(set(hotel_list))

# st.markdown(hide_menu,unsafe_allow_html=True)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_101rlWCIdG.json")


def citybased(city):
    hotel['city']=hotel['city'].str.lower()
    citybase = hotel[hotel['city']==city.lower()]
    citybase = citybase.sort_values(by='starrating',ascending=False)
    citybase.drop_duplicates(subset='hotelcode',keep='first',inplace=True)
    if(citybase.empty==0):
        hname = citybase[['hotelname','price','starrating','address','description','url']]
        return hname.head(10)
    else:
        print('No Hotels Available')

def pop_citybased(city,number):
    hotel['city'] = hotel['city'].str.lower()
    popbased = hotel[hotel['city'] == city.lower()]
    popbased = popbased[popbased['guests_no']==number].sort_values(by='starrating',ascending=False)
    popbased.drop_duplicates(subset='hotelcode', keep='first', inplace=True)
    if popbased.empty==True:
        print("Sorry No Hotel Available\n tune your constraints")
    else:
        return popbased[['hotelname','price','roomtype','guests_no','starrating','address','description', 'url']].head(10)


def requirementbased(city,number,features):
    hotel['city'] = hotel['city'].str.lower()
    hotel['description'] = hotel['description'].str.lower()
    features = features.lower()
    features_tokens = word_tokenize(features)
    sw = stopwords.words('english')
    lemm = WordNetLemmatizer()
    f1_set = {w for w in features_tokens if not w in sw}
    f_set = set()
    for se in f1_set:
        f_set.add(lemm.lemmatize(se))
    reqbased = hotel[hotel['city'] == city.lower()]
    reqbased = reqbased[reqbased['guests_no'] == number].sort_values(by='starrating', ascending=False)
    reqbased = reqbased.set_index(np.arange(reqbased.shape[0]))
    cos=[];

    for i in range(reqbased.shape[0]):
        temp_tokens = word_tokenize(reqbased['description'][i])
        temp1_set = {w for w in temp_tokens if not w in sw}
        temp_set = set()
        for se in temp1_set:
            temp_set.add(lemm.lemmatize(se))
        rvector = temp_set.intersection(f_set)
        cos.append(len(rvector))
    reqbased['similarity'] = cos
    reqbased = reqbased.sort_values(by='similarity',ascending=False)
    reqbased.drop_duplicates(subset='hotelcode',keep='first',inplace=True)
    return reqbased[['hotelname','roomtype','price','guests_no','starrating','address','description','similarity','url']].head(10)

def recommender(city,number,features,price):
    hotel['city'] = hotel['city'].str.lower()
    hotel['description'] = hotel['description'].str.lower()
    features = features.lower()
    features_tokens = word_tokenize(features)
    sw = stopwords.words('english')
    lemm = WordNetLemmatizer()
    f1_set = {w for w in features_tokens if not w in sw}
    f_set = set()
    for se in f1_set:
        f_set.add(lemm.lemmatize(se))
    reqbased = hotel[hotel['city'] == city.lower()]
    reqbased = reqbased[reqbased['guests_no'] == number]
    reqbased = reqbased[reqbased['price'] <= price].sort_values(by='starrating', ascending=False)
    reqbased = reqbased.set_index(np.arange(reqbased.shape[0]))
    cos=[];

    for i in range(reqbased.shape[0]):
        temp_tokens = word_tokenize(reqbased['description'][i])
        temp1_set = {w for w in temp_tokens if not w in sw}
        temp_set = set()
        for se in temp1_set:
            temp_set.add(lemm.lemmatize(se))
        rvector = temp_set.intersection(f_set)
        cos.append(len(rvector))
    reqbased['similarity'] = cos
    reqbased = reqbased.sort_values(by='similarity',ascending=False)
    reqbased.drop_duplicates(subset='hotelcode',keep='first',inplace=True)
    return reqbased[['hotelname','roomtype','price','guests_no','starrating','address','description','similarity','url']].head(10)





st.set_page_config(page_title="JetSetWiz",page_icon=":desert_island:",layout="wide")

with st.container():
    l , r,e,t,y,u,i,o, = st.columns(8)
    with l:
        st.subheader("JetSetWiz.com")
    with r:
        st_lottie(lottie_coding,height=50,key="coding")
        st.write(" ")
with st.container():

    st.write("       ")
    st.write("       ")
    st.write("----")


with st.container():
    st.subheader("Search hotels in Nice")
    st.write("Navigate the World of Hotels, Your Trusty Recommender")
    one_col , two_col, third_col, fourth_col = st.columns(4)
    with one_col:
        number = st.slider('Number of Guests?', 0, 24, 2)
    
    with two_col:
        cities = st.selectbox('Select/Type City?',unique_cities,index=4002)
    with third_col:
        price_no = st.number_input('Budget?($/day)',step=1,value=70)
    with fourth_col:
        st.write(" ")

with st.container():
    description = st.text_input('Describe the type of room')   



def output():
        if data is not None:
            hotels = data[['hotelname', 'address', 'price','starrating','roomtype','description','url']]
            lent = len(hotels)
            if lent > 0:
                for i, row in hotels.iterrows():
                    st.subheader(f"{row['hotelname']}")
                    st.container()
                    left, right = st.columns(2)
                    with left:
                        st.write(f" :round_pushpin:   **Address**: {row['address']}")
                    with right:
                        st.write(f" :star:  **Rating**: {row['starrating']}/5")
                    st.container()
                    left, right = st.columns(2)
                    with left:
                        st.write(f"     **Price**: :green[{row['price']}]")
                    with right:
                        st.subheader("Description:")
                        st.write(row['description'])
                    left, right = st.columns(2)
                    with left:
                        st.write(f"    ")
                    with right:
                        st.markdown(f"[Book now..]({row['url']})")
            else:
                st.write("No hotels available based on the given criteria.")
        else:
            st.write("Error: Failed to retrieve hotel data.")


with st.container():
    if cities and number and description and price_no:
        data = recommender(cities, number, description, price_no)
        output()
    elif cities and number and description:
        data = requirementbased(cities, number, description)
        output()
    elif cities and number:
        data = pop_citybased(cities, number)
        output()
    elif cities:
        data = citybased(cities)
        output()
    else:
        st.write("Error: Failed to retrieve hotel data.")
