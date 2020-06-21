from clio_lite import clio_search,clio_search_iter,clio_keywords
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import pandas as pd


st.title('Contextual Search for Multiple Scelrosis Records on PubMed')
index = st.text_input('Database for search', 'ms')
query = st.text_input('Search term', 'muitiple sclerosis')
maxr = st.text_input('max results', 25)
url = "http://127.0.0.1:9200"
seoptions = st.multiselect(
     'Which fileds to search?',
     ["title", "abstract", 'keywords'])
total, docs = clio_search(url=url, index=index, query=query, fields=seoptions, limit=maxr)
keywords = clio_keywords(url=url, index=index, query=query,
                         fields=seoptions,
                         )
st.markdown('Relted Key Phrases:')
klist = ', '.join([kw['key'] for kw in keywords])
st.text(klist)
st.markdown('Results Table')
st.table(pd.DataFrame(docs, columns=['PMID', 'title', 'keywords', 'pubdate']))
