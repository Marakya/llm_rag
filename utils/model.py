import os
import io
import re
import uuid
import base64
import chromadb
import pickle as pkl


from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.schema import Document


from utils.processing import get_data, generate_img_summaries
from utils.config import *


def get_retriever(collection_name: str, k: int = 5):
    """
    Создаёт retriever для заданной коллекции с поддержкой мультимодальных данных.
    
    Аргументы:
    - collection_name: имя коллекции в Chroma
    - k: количество релевантных документов для поиска
    
    Возвращает:
    - retriever: объект MultiVectorRetriever для поиска по векторной БД
    - chain_multimodal_rag: RAG-цепочка для генерации ответов с мультимодальными данными
    """
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vectorstore = Chroma(collection_name=collection_name,
                         embedding_function=embeddings,
                         persist_directory=PERSIST_DIR)

    # Загружаем docstore, где хранится исходный контент и метаданные
    if os.path.exists(DOCSTORE_FILE):
        with open(DOCSTORE_FILE, 'rb') as f:
            docstore_data = pkl.load(f)
    else:
        docstore_data = {}

    store = InMemoryStore()

    # Пробегаем все документы в docstore и добавляем их в store
    for doc_id, doc in docstore_data.items():
        # Обрабатываем разные форматы данных
        if isinstance(doc, Document):
            # Если это уже Document объект
            store.mset([(doc_id, doc)])
        elif isinstance(doc, dict):
            # Если это словарь (старый формат)
            store.mset([(
                doc_id,
                Document(
                    page_content=doc.get("content", ""),
                    metadata={
                        "type": doc.get("type", "unknown"),
                        "doc_id": doc_id,
                        "image_path": doc.get("image_path"),
                        "image_b64": doc.get("image_b64")
                    }
                )
            )])
        else:
            # Для других форматов
            store.mset([(
                doc_id,
                Document(
                    page_content=str(doc),
                    metadata={"type": "unknown", "doc_id": doc_id}
                )
            )])
    # MultiVectorRetriever ищет по векторной базе (Chroma) и возвращает исходные данные из docstore
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id", # ключ для сопоставления документов
        search_kwargs={"k": k}
    )
    # Создаём RAG-цепочку для мультимодального поиска и генерации ответов
    chain_multimodal_rag = multi_modal_rag_chain(retriever)
    return retriever, chain_multimodal_rag


# def get_collections() -> list:
#     return [d for d in os.listdir(PERSIST_DIR) if os.path.isdir(os.path.join(PERSIST_DIR, d))]


def get_collections() -> list:
    """
    Выводит список всех коллекций
    """
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return [col.name for col in client.list_collections()]



def save_collection(collection_name, texts, tables, image_summaries, images_b64):
    """
    Сохраняем данные в виде Document с методанными в векторную БД, а также в docstore для последующего поиска
    collection_name - имя коллекции
    texts - список текстов
    tables - список таблиц
    image_summaries - список описаний изображений
    images_b64 - список изображений в формате base64
    """
    # Проверяем созданы ли папки для хранения данных
    os.makedirs(PERSIST_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Инициализируем БД и эмбеддер
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vectorstore = Chroma(collection_name=collection_name,
                         embedding_function=embeddings,
                         persist_directory=PERSIST_DIR)

    # Загружаем docstore для 
    if os.path.exists(DOCSTORE_FILE):
        with open(DOCSTORE_FILE, 'rb') as f:
            docstore = pkl.load(f)
    else:
        docstore = {}

    docs_to_add = []

    def add_docs(content_list, doc_type):
        for i, content in enumerate(content_list):
            doc_id = str(uuid.uuid4())
            docs_to_add.append(Document(page_content=content, metadata={"doc_id": doc_id, "type": doc_type}))
            docstore[doc_id] = {"content": content, "type": doc_type}

    # Добавляем тексты и таблицы для последующего сохранения в хранилища
    add_docs(texts, "text")
    add_docs(tables, "table")

    # Изображения: сохраняем как файлы и добавляем summary в Chroma
    for i, (summary, img_b64) in enumerate(zip(image_summaries, images_b64)):
        doc_id = str(uuid.uuid4())
        # Сохраняем изображение как файл
        img_path = os.path.join(IMAGE_DIR, f"{collection_name}_{i}.jpg")
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img_b64))
        # Добавляем summary в Chroma
        docs_to_add.append(Document(page_content=summary, metadata={"doc_id": doc_id, "type": "image_summary"}))
        # В docstore сохраняем ссылку на файл и base64
        docstore[doc_id] = {"content": summary, "type": "image_summary", "image_path": img_path, "image_b64": img_b64}

    # Добавляем все документы в Chroma
    if docs_to_add:
        vectorstore.add_documents(docs_to_add)
        vectorstore.persist()

    # Сохраняем docstore
    with open(DOCSTORE_FILE, 'wb') as f:
        pkl.dump(docstore, f)

    print(f"✅ Коллекция '{collection_name}' сохранена с изображениями")



def insert_data(fpath: str, fpath_img: str, fname: str, collection_name: str):
    """
    Основная функция для обработки и сохранения загруженных данных в коллекцию.

    Пошагово:
    1) Извлекаем текстовые данные и таблицы из загруженного файла и сохраняем изображения на диск.
       Используется функция get_data(fpath, fname, fpath_img), которая возвращает:
       - texts: список текстовых блоков
       - tables: список таблиц
    2) Генерируем описания для всех изображений в виде текста и формируем их base64-кодированные версии.
       Используется функция generate_img_summaries(IMAGE_DIR), которая возвращает:
       - img_base64_list: список изображений в формате base64
       - image_summaries: список текстовых описаний изображений
    3) Сохраняем все данные в коллекцию:
       - текстовые блоки
       - таблицы
       - описания изображений
       - сами изображения (base64 + путь на диске)

    Аргументы:
        
        - fpath (str): путь к папке с загруженным файлом
        - fpath_img (str): путь к папке для сохранения изображений
        - fname (str): имя загруженного файла
        - collection_name (str): имя коллекции для сохранения данных

    Выход:
        Сохраненная коллекция в PERSIST_DIR

    """
    texts, tables = get_data(fpath, fname, fpath_img)
    img_base64_list, image_summaries = generate_img_summaries(IMAGE_DIR)

    # Вызываем функцию сохранения данных в коллекцию 
    save_collection(
        collection_name,
        texts,
        tables,
        image_summaries,
        img_base64_list,
    )


def looks_like_base64(sb):
    """Проверяет, похожа ли переданная строка на base64-кодированную строку."""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None



def is_image_data(b64data):
    """
    Проверяет, является ли base64-строка изображением, анализируя сигнатуру (header) файла.

    Аргументы:
    b64data : str
        Строка с данными в формате base64.

    Возвращает:
    bool
        True, если строка представляет собой изображение (JPEG, PNG, GIF, WEBP), иначе False.

    Логика:
    - Каждое изображение в бинарном формате начинается с уникального набора байтов (magic numbers),
      называемого сигнатурой файла.
      Например:
        JPEG: b'\xff\xd8\xff'
        PNG:  b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a'
        GIF:  b'\x47\x49\x46\x38'
        WEBP: b'\x52\x49\x46\x46'
    - Функция декодирует base64, берёт первые 8 байт и проверяет их на совпадение с известными сигнатурами.
    - Если сигнатура совпала, возвращается True, иначе False.
    - В случае любой ошибки при декодировании возвращается False.
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Декодируем и берём первые 8 байт
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Изменяет размер изображения, закодированного в Base64.

    Аргументы:
    base64_string : str
        Строка с изображением в формате Base64.
    size : tuple(int, int), по умолчанию (128, 128)
        размеры изображения (ширина, высота).

    Возвращает:
    str
        Новое изображение, закодированное в Base64.

    Логика работы:
    1. Декодируем строку Base64 в бинарные данные.
    2. Загружаем изображение через PIL из байтового потока.
    3. Меняем размер изображения на указанный размер с помощью фильтра LANCZOS для качественного ресайза.
    4. Сохраняем изменённое изображение в байтовый буфер с исходным форматом.
    5. Кодируем содержимое буфера обратно в Base64 и возвращаем как строку.

    Когда нужно уменьшить изображение перед передачей модели, чтобы ускорить обработку и уменьшить потребление памяти.
    """
    # Декодируем строку Base64
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Изменяем размер изображения
    resized_img = img.resize(size, Image.LANCZOS)

    # Сохраняем изменённое изображение в байтовый буфер
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Кодируем изменённое изображение в Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs: list) -> dict:
    """
    Разделяет документы на тексты и изображения в формате base64.

    Аргументы:
    docs : list
        Список документов, полученных от retriever. Каждый элемент может быть объектом Document
        или строкой (например, base64 изображения или текст).

    Возвращает:
    dict
        Словарь с ключами:
            - "images": список base64 изображений (уже изменённых размера для отображения)
            - "texts": список текстовых документов
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Если документ — объект Document, берем его page_content
        if isinstance(doc, Document):
            doc = doc.page_content
        # Проверяем, похоже ли содержимое на base64 изображение
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            # Всё остальное считаем текстом
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """
    Формирует входные сообщения для модели на основе текста и изображений.

    Аргументы:
    data_dict : dict
        Словарь с ключами:
        - "context": {"texts": [...], "images": [...]}
        - "question": вопрос пользователя

    Возвращает:
    list
        Список с одним объектом HumanMessage, содержащим все тексты и изображения.
    """

    # Объединяем все тексты в один блок
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Если есть изображения, добавляем их в список сообщений
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # Формируем текстовое сообщение с инструкцией для модели
    text_message = {
        "type": "text",
        "text": (
            "You are a financial analyst, and your task is to provide investment advice.\n"
            "You will be given texts, tables, and images, usually charts or graphs.\n"
            "Use this information to provide investment guidance related to the user's question.\n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Texts and/or tables:\n"
            f"{formatted_texts}"
            "Generate your answer in Russian."
        ),

    }

    messages.append(text_message)
    # Возвращаем как список с объектом HumanMessage (требование LangChain)
    return [HumanMessage(content=messages)]


def limit_docs(docs: list, k: int = 5) -> list:
    """
    Ограничивает количество документов до k
    """
    return docs[:k]


def multi_modal_rag_chain(retriever):
    """
    Создает мультимодальную RAG-цепочку (Retrieval-Augmented Generation).
    
    Эта цепочка берет релевантные документы (тексты и изображения), 
    обрабатывает их и передает в LLM для генерации ответа.
    """

    
    model = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=1024, api_key=OPENAI_API_KEY)

    # Строим цепочку обработки
    # 'context' — это документы, извлеченные retriever
    # Сначала ограничиваем количество документов до 3 с помощью limit_docs
    # Затем разделяем документы на тексты и изображения для отдельной обработки
    chain = (
        {
            "context": retriever | RunnableLambda(lambda docs: limit_docs(docs, k=3)) | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(), # 'question' передается без изменений
        }
        | RunnableLambda(img_prompt_func) # Формируем готовый prompt для модели с текстами и изображениями
        | model # Передаем в LLM для генерации ответа
        | StrOutputParser()  # Конвертируем результат модели в строку
    )

    return chain

def translate_to_english(text: str) -> str:
    """Перевод вопроса на английский"""
    translator = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        temperature=0
    )
    response = translator.invoke(f"Translate the following question from Russian to English (business/finance style) - without changing main idea: {text}")
    return response.content
