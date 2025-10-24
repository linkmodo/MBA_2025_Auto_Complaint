import streamlit as st
from mba_analysis import main

if __name__ == '__main__':
    st.set_page_config(page_title='Market Basket Analysis', layout='wide')
    main()
