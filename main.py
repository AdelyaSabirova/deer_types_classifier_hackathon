import streamlit as st
import os
from classifier import CLASS_p
from PIL import Image
import uuid
from datetime import datetime
import plotly.graph_objs as go
from tqdm import tqdm
import time

st.title("Цифровой прорыв ikanam_chipi_chipi")
st.write("Классификация парнокопытных")

# Порог уверенности модели классификации
confidence_threshold = 0.80

def load_images_from_folder(folder):
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('jpeg', 'jpg', 'png')):
                image_files.append(os.path.join(root, file))
    return image_files

def save_image_to_folder(image, category, original_filename, base_folder):
    folder_path = os.path.join(base_folder, category)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = original_filename
    while os.path.exists(os.path.join(folder_path, unique_filename)):
        unique_filename = f"{uuid.uuid4()}{file_extension}"

    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(os.path.join(folder_path, unique_filename))

def plot_histogram(class_counts):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    fig = go.Figure(data=[go.Bar(x=classes, y=counts, marker_color=['red', 'green', 'blue', 'gray'])])
    fig.update_layout(title='Количество файлов в каждом классе', xaxis_title='Классы', yaxis_title='Количество файлов')
    st.plotly_chart(fig)

uploaded_files = st.file_uploader("Загрузите изображения или папки (только JPEG и PNG)", type=["jpeg", "jpg", "png"], accept_multiple_files=True)

folder_uploaded = st.text_input("Или введите путь к папке с изображениями:")

submit_button = st.button('Готово')

if submit_button:
    all_files = []
    error_messages = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            all_files.append(uploaded_file)

    if folder_uploaded:
        folder_files = load_images_from_folder(folder_uploaded)
        all_files.extend(folder_files)

    if all_files:
        classifier = CLASS_p()
        base_folder = datetime.now().strftime("Classifier_Result_%Y_%m_%d_%H:%M:%S")
        os.makedirs(base_folder, exist_ok=True)
        
        class_counts = {'Кабарга': 0, 'Косуля': 0, 'Олень': 0, 'Неопределено': 0}
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_files = len(all_files)

        start_time = time.time()
        
        for i, file in enumerate(tqdm(all_files)):
            try:
                if isinstance(file, str):
                    with open(file, "rb") as f:
                        image = Image.open(f).convert('RGB')  # Преобразование изображения к формату RGB
                    original_filename = os.path.basename(file)
                else:
                    with open(file, "rb") as f:
                        image = Image.open(f).convert('RGB')  # Преобразование изображения к формату RGB
                    original_filename = file.name
                
                category, confidence = classifier.predict(image)
                if confidence >= confidence_threshold:
                    save_image_to_folder(image, category, original_filename, base_folder)
                    class_counts[category] += 1
                else:
                    save_image_to_folder(image, 'Неопределено', original_filename, base_folder)
                    class_counts['Неопределено'] += 1

                elapsed_time = time.time() - start_time
                avg_time_per_image = elapsed_time / (i + 1)
                remaining_time = avg_time_per_image * (total_files - (i + 1))
                
                # Форматирование оставшегося времени
                if remaining_time >= 60:
                    minutes = int(remaining_time // 60)
                    seconds = int(remaining_time % 60)
                    time_str = f"{minutes} минут {seconds} секунд"
                else:
                    seconds = int(remaining_time)
                    time_str = f"{seconds} секунд"
                
                # Вычисление скорости обработки
                processing_speed = 1 / avg_time_per_image
                
                progress_bar.progress((i + 1) / total_files)
                progress_text.text(
f"""Обработка изображения {i + 1} из {total_files}. Осталось: {total_files - (i + 1)} изображений. 
Примерное оставшееся время: {time_str}. 
Скорость обработки: {processing_speed:.2f} изображений/сек.""")

            except Exception as e:
                error_message = f"""Произошла ошибка при обработке файла: {file}. 
Error: {str(e)}"""
                error_messages.append(error_message)

        st.success(
f'''Изображения успешно классифицированы и распределены по папкам! 
Папка: {base_folder}''')
        
        plot_histogram(class_counts)

        if error_messages:
            with st.expander("Список файлов неверного формата"):
                for error in error_messages:
                    st.error(error)
    else:
        st.error('Пожалуйста, загрузите хотя бы одно изображение или укажите путь к папке.')