import streamlit as st
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

pages = {
    "Анализ и модель": analysis_and_model_page,
    "Презентация": presentation_page,
}

# Отображение навигации
current_page = st.sidebar.selectbox("Выберите страницу", options=list(pages.keys()))
pages[current_page]()  # вызываем функцию для выбранной страницы
