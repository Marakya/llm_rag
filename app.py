import base64
import traceback
import streamlit as st
import os
import pickle as pkl

from io import BytesIO
from PyPDF2 import PdfMerger

from utils.config import *
from utils.model import get_collections, insert_data, translate_to_english, get_retriever


st.title("Multimodal RAG Q&A")

tab1, tab2 = st.tabs(["📂 Загрузка данных", "Использование коллекции"])
with tab1:
    uploaded_files = st.file_uploader(
        "Выберите файл(ы)", 
        type=["pdf", "docx", "txt", "png", "jpg"], 
        accept_multiple_files=True
    )
    collection_name = st.text_input("Введите название коллекции")

    if st.button("Добавить в коллекцию"):
        if not uploaded_files:
            st.warning("Пожалуйста, загрузите хотя бы один файл.")
        elif not collection_name.strip():
            st.warning("Введите название коллекции.")
        else:
            # имя первого файла
            first_file = uploaded_files[0]
            fname = first_file.name  
            base_name = os.path.splitext(fname)[0]

            fpath = os.path.abspath(f"./data/{base_name}")
            # fpath_img = os.path.join(fpath, "figures")
            fpath_img = IMAGE_DIR
            os.makedirs(fpath, exist_ok=True)

            # если один файл → просто сохраняем его
            if len(uploaded_files) == 1:
                with open(f"{fpath}/{fname}", "wb") as f:
                    f.write(first_file.getbuffer())
            else:
                # несколько файлов → объединяем PDF в один
                pdfs = [file for file in uploaded_files if file.name.lower().endswith(".pdf")]
                if pdfs:
                    merger = PdfMerger()
                    for pdf in pdfs:
                        merger.append(pdf)
                    merged_pdf_path = os.path.join(fpath, fname)  
                    merger.write(merged_pdf_path)
                    merger.close()
               

            with st.spinner("Обрабатываем и сохраняем данные..."):
                insert_data(fpath, fpath_img, fname, collection_name)

            st.success(f"✅ Коллекция '{collection_name}' успешно добавлена!")

# --- Вкладка 2: Использование коллекции ---
with tab2:
    st.subheader("Задайте вопрос по коллекции")

    collections = get_collections()

    if not collections:
        st.info("Коллекций пока нет. Добавьте хотя бы одну во вкладке 'Загрузка данных'.")
    else:
        selected_collection = st.selectbox("Выберите коллекцию", collections)
        query = st.text_input("Введите вопрос:")

        if st.button("Получить ответ", key="ask_btn"):
            if query.strip() == "":
                st.warning("Введите вопрос.")
            else:
                try:
                    # Получаем retriever и RAG-цепочку
                    retriever, rag_chain = get_retriever(selected_collection, k=5)
                
                    
                    with st.spinner("Модель обрабатывает запрос..."):
                        query_eng = translate_to_english(query) 
                        # Получаем релевантные документы
                        docs = retriever.get_relevant_documents(query_eng)
                        print('docs', docs)
                        # Генерируем ответ с помощью RAG-цепочки
                        answer = rag_chain.invoke(query_eng)

                    st.subheader("Ответ модели:")
                    st.write(answer)

                    st.subheader("Релевантные документы:")

                    for i, doc in enumerate(docs):
                        st.markdown(f"**Документ {i+1}:**")

                        doc_type = doc.metadata.get("type", "unknown")
                        if doc_type in ("image", "image_summary"):
                            img_path = doc.metadata.get("image_path")
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, caption=f"Изображение {i+1}")
                            else:
                                # fallback на base64 если путь не найден
                                img_b64 = doc.metadata.get("image_b64")
                                if img_b64:
                                    img_data = BytesIO(base64.b64decode(img_b64))
                                    st.image(img_data, caption=f"Изображение {i+1}")
                                else:
                                    st.write("Изображение недоступно")
                        else:
                            # Для текстов и таблиц просто выводим контент
                            st.write(doc.page_content)

                        st.markdown("---")

                except Exception as e:
                    st.error(f"Ошибка при обработке запроса: {str(e)}")
                    st.error(traceback.format_exc())
