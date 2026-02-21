import streamlit as st
import docx
from langchain_core.messages import SystemMessage, HumanMessage
from prompts import SYSTEM_PROMPT_v1, SYSTEM_PROMPT_v2, USER_PROMT, RULES, PROMPT_SENTIMENT
from langchain_deepseek import ChatDeepSeek
from markdown_pdf import MarkdownPdf, Section
from io import BytesIO
from dotenv import load_dotenv
from loguru import logger 
import plotly.io as pio
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()
import os
import json
import plotly.express as px
import pandas as pd
from datetime import datetime
API_DEEPSEEK=os.getenv("API_DEEPSEEK")
import time

import json
import re
import pandas as pd
from typing import Optional, List, Dict, Any
import ast

def parse_llm_json_to_df(text: str) -> pd.DataFrame:
    """
    Парсит JSON из текста, сгенерированного LLM, с запасными методами.
    
    Args:
        text: Строка с JSON (возможно, с лишним текстом до/после)
    
    Returns:
        pandas.DataFrame с распарсенными данными
    """
    
    def try_parse_json(json_str: str) -> Optional[List[Dict[str, Any]]]:
        """Пробует распарсить JSON стандартным методом"""
        print("def try_parse_json(json_str: str)")
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]  # Если это объект, оборачиваем в список
        except:
            pass
        return None
    
    def try_parse_ast(json_str: str) -> Optional[List[Dict[str, Any]]]:
        """Пробует распарсить как Python literal (более либерально чем JSON)"""
        print("def try_parse_ast(json_str: str)")
        try:
            # Заменяем true/false/null на Python-совместимые значения
            cleaned = json_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            data = ast.literal_eval(cleaned)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except:
            pass
        return None
    
    def find_json_by_brackets(text: str) -> Optional[str]:
        """Находит JSON по балансу квадратных скобок"""
        print("def find_json_by_brackets(text: str)")
        stack = []
        start_idx = -1
        results = []
        
        for i, char in enumerate(text):
            if char == '[':
                if not stack:  # Первая открывающая скобка
                    start_idx = i
                stack.append(char)
            elif char == ']':
                if stack:
                    stack.pop()
                    if not stack and start_idx != -1:  # Нашли закрывающую скобку для первого уровня
                        results.append(text[start_idx:i+1])
                        start_idx = -1
        
        if results:
            # Берем самый длинный результат (наиболее вероятный полный JSON)
            return max(results, key=len)
        
        return None
    
    def extract_json_objects(text: str) -> Optional[str]:
        """Извлекает массив JSON объектов, даже если они разбросаны по тексту"""
        print("def extract_json_objects(text: str)")
        # Ищем все объекты в тексте
        objects = []
        brace_count = 0
        start_idx = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    # Нашли полный объект
                    obj_str = text[start_idx:i+1]
                    try:
                        # Проверяем, что это валидный JSON объект
                        json.loads(obj_str)
                        objects.append(obj_str)
                    except:
                        # Если не валидный, пробуем очистить
                        try:
                            cleaned = clean_json_string(obj_str)
                            json.loads(cleaned)
                            objects.append(cleaned)
                        except:
                            pass
                    start_idx = -1
        
        if objects:
            # Оборачиваем все найденные объекты в массив
            return '[' + ','.join(objects) + ']'
        
        return None
    
    def clean_json_string(json_str: str) -> str:
        """Очищает JSON строку от распространенных проблем"""
        print("def clean_json_string(json_str: str)")
        # Удаляем лишние пробелы и переносы строк
        json_str = json_str.strip()
        
        # Исправляем кавычки
        json_str = json_str.replace('"', '"').replace('"', '"')
        
        # Удаляем комментарии
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
        
        # Убираем trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str
    
    def fix_quotes_and_commas(json_str: str) -> str:
        """Исправляет проблемы с кавычками и запятыми"""
        print("def fix_quotes_and_commas(json_str: str)")
        # Заменяем одинарные кавычки на двойные, но не внутри строк
        result = []
        in_string = False
        escape_next = False
        
        for char in json_str:
            if char == '\\' and not escape_next:
                escape_next = True
                result.append(char)
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                result.append(char)
            elif char == "'" and not in_string and not escape_next:
                # Заменяем одинарную кавычку на двойную только вне строк
                result.append('"')
            else:
                result.append(char)
            
            escape_next = False
        
        return ''.join(result)
    
    # Основная логика
    data = None
    
    # Шаг 1: Пробуем распарсить как есть
    data = try_parse_json(text)
    
    # Шаг 2: Пробуем найти JSON по квадратным скобкам
    if data is None:
        extracted = find_json_by_brackets(text)
        if extracted:
            data = try_parse_json(extracted)
    
    # Шаг 3: Пробуем найти и собрать все объекты
    if data is None:
        extracted = extract_json_objects(text)
        if extracted:
            data = try_parse_json(extracted)
    
    # Шаг 4: Пробуем очистить и исправить
    if data is None:
        cleaned = clean_json_string(text)
        data = try_parse_json(cleaned)
    
    # Шаг 5: Пробуем через ast.literal_eval
    if data is None:
        fixed = fix_quotes_and_commas(text)
        data = try_parse_ast(fixed)
        
        # Если не получилось, пробуем найти массив в исправленном тексте
        if data is None:
            extracted = find_json_by_brackets(fixed)
            if extracted:
                data = try_parse_ast(extracted)
    
    # Если ничего не помогло, возвращаем пустой DataFrame
    if data is None:
        return None
    
    # Преобразуем в DataFrame
    df = pd.DataFrame(data)
    
    return df




def generate_pdf(markdown_content, filename, df = None):
    pdf = MarkdownPdf()
    pdf.meta["title"] = 'Отчет'
    pdf.meta["author"] = 'AI Assistant'
    pdf_content = f"Отчет: {filename}\n\n "

    pdf_content += "Анализ аналитического отчета\n\n"

    # Добавляем текстовый контент
    pdf_content += markdown_content


    if df is not None and not df.empty:
        pdf_content += "*Анализ сентимента*"
        # Создаем фигуру и оси
        fig, ax = plt.subplots(figsize=(12, 6))

        # Создаем точечную диаграмму
        companies = range(len(df))
        ax.scatter(companies, df['sentiment'], 
                s=225,  # Размер точек (15^2)
                c='darkgrey',
                edgecolors='darkgrey',
                linewidths=1,
                zorder=2)

        # Добавляем горизонтальную линию на отметке 5
        ax.axhline(y=5, color='black', linestyle='-', linewidth=1, zorder=1)

        # Добавляем текст для линии
        ax.text(0.02, 5.1, 'нейтральный сентимент', 
                transform=ax.get_yaxis_transform(),
                fontsize=10,
                verticalalignment='bottom')

        # Настраиваем подписи
        ax.set_xlabel('Компания', fontsize=12)
        ax.set_ylabel('Сентимент', fontsize=12)
        ax.set_title('Сентимент по компаниям', fontsize=14, pad=15)

        # Устанавливаем метки на оси X
        ax.set_xticks(companies)
        ax.set_xticklabels(df['company'], rotation=45, ha='right')

        # Устанавливаем диапазон для оси Y
        ax.set_ylim(0, 10)

        # Добавляем сетку для лучшей читаемости
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Убираем легенду
        ax.legend_.remove() if ax.legend_ else None

        # Сохраняем в base64
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        pdf_content += f"![График сентимента](data:image/png;base64,{img_base64})\n\n"


        pdf_content += df.to_markdown()

    pdf.add_section(Section(pdf_content, toc=False))
    return pdf

deepseek_llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=st.secrets["MY_LLM"],
    temperature=1,
    streaming=True
)

deepseek_llm_not_streaming = ChatDeepSeek(
    model="deepseek-chat",
    api_key=st.secrets["MY_LLM"],
    temperature=1
)

# deepseek_llm = ChatDeepSeek(
#     model="deepseek-reasoner",
#     api_key=API_DEEPSEEK,
#     temperature=1,
#     max_tokens=32000,
#     reasoning_effort="medium",
#     streaming=True
# )

st.subheader('ИИ-помощник «Цензор»')

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "text": "Заргузите документ, который необходимо проверить"}]

for i, message in enumerate(st.session_state.messages):
    with st.chat_message("assistant", avatar=":material/priority_high:"):
        # Отображаем текстовый контент
        if 'text' in message:
            st.write(message['text'])

# React to user input
user_input = st.chat_input('Введите дополнительные инструкции или оставьте поле пустым', accept_file=True, accept_audio=False)
if user_input:
    if user_input.files:
        doc = docx.Document(user_input.files[0])
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        content = '\n'.join(full_text)
        # Display user message
        with st.chat_message("user", avatar=":material/person_pin:"):
            # st.write(user_input)
            st.write(f"{user_input.get("text", " ")}\n\n Прикрепленный файл: {user_input.files[0].name}")
            # st.markdown(prompt)

        with st.chat_message("ai", avatar=":material/android:"):
            temp_message = st.empty()
            temp_message.write("⏳ Обработка запроса...")

            ### Готовим сентимент #####

            messages_sentiment = [HumanMessage(content=PROMPT_SENTIMENT.format(report_text = content))]
            response_sentiment = deepseek_llm.invoke(messages_sentiment)
            temp_message.empty()
            try:
                # companies_sentiment = json.loads(response_sentiment.content)
                df_companies_sentiment = parse_llm_json_to_df(response_sentiment.content)

                if df_companies_sentiment is None:
                    st.warning('Не удалось прочитать JSON объект')
                else:
                    st.write("✅ Готов расчет сентимента")
                    # Создаем точечную диаграмму
                    fig = px.scatter(df_companies_sentiment, 
                                    x='company', 
                                    y='sentiment',
                                    title='Сентимент по компаниям',
                                    labels={'company': 'Компания', 'sentiment': 'Сентимент'},
                                    size=[20] * len(df_companies_sentiment),  # Размер точек
                                    color_discrete_sequence=['darkgrey'])

                    # Добавляем горизонтальную линию на отметке 5
                    fig.add_hline(y=5, 
                                line_dash="solid", 
                                line_color="black",
                                line_width=1,
                                annotation_text="нейтральный сентимент",
                                annotation_position="top left")

                    # Настраиваем отображение
                    fig.update_layout(
                        xaxis_title="Компания",
                        yaxis_title="Сентимент",
                        showlegend=False,
                        yaxis=dict(
                            range=[0, 10]  # Устанавливаем диапазон для лучшей видимости
                        )
                    )

                    # Настраиваем внешний вид точек
                    fig.update_traces(
                        marker=dict(
                            size=15,  # Размер точек
                            line=dict(width=1, color='darkgrey')  # Обводка точек
                        )
                    )

                    # Показываем график
                    st.plotly_chart(fig)
            except:
                st.warning('Не удалось рассчитать сентимент')
                df_companies_sentiment = None
                st.write(response_sentiment.content)

            ### Анализируем текст ####
            temp_message = st.empty()
            temp_message.write("⏳ Анализ отчета...")
            system_instructions = SYSTEM_PROMPT_v2.format(rules=RULES, date = datetime.now().strftime("%Y-%m-%d"))
            user_instructions = USER_PROMT.format(additional_instructions = user_input.get("text", " "),
                                                   analytical_report = content)
            # logger.info(f"Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} \n\n SYSTEM_PROMPT: \n\n {system_instructions} \n")
            messages = [
            SystemMessage(content=system_instructions),
            HumanMessage(content=user_instructions)]
            
            # st.write(messages)
            

            def generate_response():
                for chunk in deepseek_llm.stream(messages):
                    if chunk.content:
                        yield chunk.content
            if generate_response:
                temp_message.empty()
                st.write("✅ Ответ готов")

            response = st.write_stream(generate_response)
            download_content = generate_pdf(response, user_input.files[0].name, df_companies_sentiment)

            # Сохраняем в буфер
            buffer = BytesIO()
            download_content.save(buffer)
            buffer.seek(0)
            st.download_button(
                label="Скачать PDF",
                data=buffer.getvalue(),
                file_name=f"{os.path.splitext(user_input.files[0].name)[0][:25]}_результаты_проверки.pdf",
                mime="application/pdf",
                key="download_pdf",
                on_click="ignore",
                icon = ":material/download:"
            )
    else:
        st.warning("Пожалуйста, загрузите документ")



