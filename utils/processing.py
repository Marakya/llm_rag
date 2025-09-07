import os
import time
import base64
import pickle as pkl

from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

from utils.config import *


def extract_pdf_elements(path: str, fpath: str) -> list:
    """
    Извлечение данных из PDF
    """
    return partition_pdf(
        filename=path,
        extract_images_in_pdf=True,  # включаем извлечение картинок
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=fpath,
        languages=["ru", "en"],
    )


def categorize_elements(raw_pdf_elements: list) -> tuple[list, list, list]:
    """
    Извлекаем элементы из PDF (таблицы, текст и картинки в отдельные списки)
    """
    tables = []
    texts = []
    images = []
    for element in raw_pdf_elements:
        etype = str(type(element))
        if "unstructured.documents.elements.Table" in etype:
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in etype:
            texts.append(str(element))
        elif "unstructured.documents.elements.Image" in etype:
            images.append(element) 
    return texts, tables, images


def get_data(fpath: str, fname: str, fpath_img: str) -> tuple[str, str]:
    """
    Извлекаем данные из PDF и возвращаем текст,таблицы, картинки. Сохраняем картинки в файлы.
    Разбиваем тексты на чанки.

    fpath - путь к папке с PDF
    fpath_img - путь к папке для сохранения изображений
    fname - имя PDF файла
    """

    file_path = f"{fpath}/{fname}"
    raw_pdf_elements = extract_pdf_elements(file_path, fpath_img)
    texts, tables, images = categorize_elements(raw_pdf_elements)

    # Сохраняем картинки в файлы
    for i, img in enumerate(images, start=1):
        img_path = os.path.join(fpath_img, f"image_{i}.jpg")
        with open(img_path, "wb") as f:
            f.write(img.data)
        print(f"Сохранено: {img_path}")

    # Разбивка текста на чанки
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0
    )
    joined_texts = " ".join(texts)
    texts_4k_token = text_splitter.split_text(joined_texts)
    return texts_4k_token, tables



def encode_image(image_path):
    """
    Получение base64 строки

    image_path - путь к изображению
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """
    Вызов модели для создания описания изображения

    img_base64 - base64 строка изображения
    prompt - текстовый запрос для модели
    """
    chat = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=OPENAI_API_KEY)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(fpath):
    """
    Генерируем описание картинок и base64 закодированные строки для изображений
    path: путь к картинкам .jpg
    """

    # Сохранение base64 encoded images
    img_base64_list = []

    # Сохранение описаний изображений
    image_summaries = []

    prompt = """Ты — ассистент, задача которого состоит в том, чтобы создавать краткие описания изображений для последующего поиска.&#x20;
    Эти описания будут преобразованы в эмбеддинги и использоваться для нахождения исходного изображения.&#x20;
    Составь сжатое и точное описание изображения, оптимизированное для поиска.
    """

    for i, img_file in enumerate(sorted(os.listdir(fpath))):
      if img_file.endswith(".jpg"):
          img_path = os.path.join(fpath, img_file)
          base64_image = encode_image(img_path)
          img_base64_list.append(base64_image)

          try:
              summary = image_summarize(base64_image, prompt)
              image_summaries.append(summary)
          except Exception as e:
              print(f"Ошибка при обработке {img_file}: {e}")
              time.sleep(60)  
              summary = image_summarize(base64_image, prompt)
              image_summaries.append(summary)

          # Добавим небольшую паузу между запросами, чтобы снизить нагрузку на лимит
          time.sleep(1)

    return img_base64_list, image_summaries