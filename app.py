import PyPDF2  # Importa a biblioteca PyPDF2 para ler arquivos PDF
import streamlit as st  # Importa a biblioteca Streamlit para criar a interface web
import google.generativeai as genai  # Importa a biblioteca do Google Generative AI para usar modelos de linguagem
import pymupdf  # Importa a biblioteca PyMuPDF (fitz) para ler arquivos PDF
import fitz  # Importa a biblioteca fitz para trabalhar com documentos PDF
import time
from pathlib import Path
import joblib
import os

BADGE = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRsKDP3usN9gwoHxGMIWRYI22xNl5aJ5elOjA&s"
LOGO="https://fcmsantacasasp.edu.br/wp-content/themes/fcmscsp/content/imgs/faculdade-de-ciencias-medicas-da-santa-casa-de-sao-paulo.png"
INSTRUCOES='Você é um assistente de estudo do curso de medicina da Santa Casa de Misericódia de São Paulo. Seu propósito é responder a dúvidas sobre os documentos anexados. Não serve para propósito de diagnósticos e consultas, portanto é usado dentro de sala de aula. Não oferece risco as pessoas pois é usado em um ambiente controlado e seguro.'
STREAM_RESPONSE=False
HABILITAR_CHAT=True

st.session_state.temperatura=0.10
st.session_state.topP=0.95
st.session_state.topK=64
st.session_state.modelo="gemini-1.5-flash"

generation_config = {
    "temperature": st.session_state.temperatura,  # Define a temperatura para a geração de texto (menor = mais previsível)
    "top_p": st.session_state.topP,  # Define a probabilidade de escolha das palavras (maior = mais palavras prováveis)
    "top_k": st.session_state.topK,  # Define o número de palavras candidatas para escolher (maior = mais opções)
    "max_output_tokens": 8192,  # Define o número máximo de tokens na saída
    "response_mime_type": "text/plain",  # Define o tipo de mídia da resposta
}

model = genai.GenerativeModel(
    model_name=st.session_state.modelo,  # Define o modelo de linguagem a ser usado (Gemini 1.5 Flash)
    generation_config=generation_config,  # Define a configuração de geração de texto
    # safety_settings = Adjust safety settings  # Ajusta as configurações de segurança (opcional)
    # See https://ai.google.dev/gemini-api/docs/safety-settings  # Link para a documentação das configurações de segurança
    system_instruction=INSTRUCOES,
)

# Inicializar a sessão de chat (fora da função para ser persistente)
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(
        history=[
                {
                "role": "user",
                "parts": [
                    ""
                ],
                },
            ]
    )
    
def getTokens(prompt):
    """
    Conta o número de tokens em um prompt.

    Args:
        prompt (str): O prompt a ser contado.

    Returns:
        int: O número de tokens no prompt.
    """
    if(st.session_state.apiKeyGoogleAiStudio != ""):
        genai.configure(api_key=st.session_state.apiKeyGoogleAiStudio)
        return model.count_tokens(prompt).total_tokens
    else:
        return 0
    
def clear_chat_history():
    """
    Limpa o histórico de mensagens do chat.
    """
    try:
        st.session_state.messages = [
            {"role": "assistant", "content": "Faça o upload de um livro ou apostila e faça perguntas para a IA ou qualquer outra solicitação como um questionário ou resumo."}]
        st.session_state.chat_session=model.start_chat(
            history=[
                {
                "role": "user",
                "parts": [
                    ""
                ],
                },
            ]
        )
        st.session_state.docsEnviados=False
    except:
        st.error("Erro ao limpar o histórico do chat.",icon="❌")
        
def get_gemini_reponse(prompt='',raw_text=''):
    """
    Obtém uma resposta do modelo Gemini.

    Args:
        prompt (str): O prompt a ser enviado para o modelo.
        raw_text (str): O texto bruto do arquivo PDF.

    Returns:
        str: A resposta do modelo Gemini.
    """
    contexto=raw_text
    response = st.session_state.chat_session.send_message(contexto+prompt,stream=STREAM_RESPONSE)
    return response

# read all pdf files and return text
def get_pdf_text(pdf_docs):
    """
    Lê o texto de todos os arquivos PDF fornecidos.

    Args:
        pdf_docs (list): Uma lista de arquivos PDF.

    Returns:
        str: O texto extraído de todos os arquivos PDF.
    """
    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PyPDF2.PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        with open("material.txt", "w",encoding="utf-8") as arquivo:
            arquivo.write(text)
        return text
    except:
        st.error("Erro ao converter arquivo PDF para texto",icon="❌")

def get_pdf_text_v2(pdf_docs):
    """
    Lê o texto de todos os arquivos PDF fornecidos usando PyMuPDF.

    Args:
        pdf_docs (list): Uma lista de arquivos PDF.

    Returns:
        str: O texto extraído de todos os arquivos PDF.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_bytes = pdf.getvalue()
        # Open the PDF with PyMuPDF (fitz) using the bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
    return text

def changeConfigModel():
    generation_config = {
    "temperature": st.session_state.temperatura,  # Define a temperatura para a geração de texto (menor = mais previsível)
    "top_p": st.session_state.topP,  # Define a probabilidade de escolha das palavras (maior = mais palavras prováveis)
    "top_k": st.session_state.topK,  # Define o número de palavras candidatas para escolher (maior = mais opções)
    "max_output_tokens": 8192,  # Define o número máximo de tokens na saída
    "response_mime_type": "text/plain",  # Define o tipo de mídia da resposta
    }
    global model 
    model = genai.GenerativeModel(
    model_name=st.session_state.modelo,  # Define o modelo de linguagem a ser usado (Gemini 1.5 Flash)
    generation_config=generation_config,  # Define a configuração de geração de texto
    # safety_settings = Adjust safety settings  # Ajusta as configurações de segurança (opcional)
    # See https://ai.google.dev/gemini-api/docs/safety-settings  # Link para a documentação das configurações de segurança
    system_instruction=INSTRUCOES,  # Define a instrução do sistema para o modelo de linguagem
    )

def sidebar():
    st.logo(LOGO, link=None, icon_image=LOGO)  # Exibe o logotipo azul do SENAI
    with st.sidebar:
        st.link_button("Ajuda?",'https://aistudio.google.com/app/apikey')     
        st.title("Configurações:")
        st.session_state.apiKeyGoogleAiStudio = st.text_input("Chave de API Google AI Studio:", "", type='password',help="Obtenha sua chave de API em https://ai.google.dev/aistudio")  # Campo de entrada para a chave API
        st.write(f"**Total de Tokens**: {getTokens(st.session_state.chat_session.history)}"+"/1.048.576") 
        with st.expander("Configurações avançadas"):
            st.selectbox("Selecione o modelo de linguagem",("gemini-1.5-flash","gemini-1.5-pro","gemini-1.5-pro-exp-0801"),on_change=changeConfigModel,help="**Gemini 1.5 Pro**\n2 RPM (requisições por minuto)\n32.000 TPM (tokens por minuto)\n50 RPD (requisições por dia)\n\n**Gemini 1.5 Flash**\n15 RPM (requisições por minuto)\n1 milhão TPM (tokens por minuto)\n1.500 RPD (requisições por dia)",disabled=True)
            st.session_state.temperatura=st.slider("Temperatura",0.05,2.0,0.10,0.05,help="**Temperatura**: Imagine a temperatura como um controle de criatividade do modelo. Em temperaturas mais altas, o Gemini se torna mais aventureiro, explorando respostas menos óbvias e mais criativas. Já em temperaturas mais baixas, ele se torna mais conservador, fornecendo respostas mais diretas e previsíveis. É como ajustar o termostato de um forno: quanto mais alto, mais quente e mais chances de algo queimar; quanto mais baixo, mais frio e mais seguro.",on_change=changeConfigModel)
            st.session_state.topP=st.slider("Top P",0.05,1.0,0.95,0.05,help="Pense no **TopP** como um filtro que controla a variedade das palavras que o Gemini pode usar. Um valor de TopP mais baixo significa que o modelo se concentrará em um conjunto menor de palavras mais prováveis, resultando em respostas mais coerentes e focadas. Por outro lado, um valor mais alto permite que o modelo explore um vocabulário mais amplo, o que pode levar a respostas mais diversas e inesperadas. É como escolher um dicionário: um dicionário menor oferece menos opções, mas as palavras são mais conhecidas; um dicionário maior oferece mais opções, mas pode ser mais difícil encontrar a palavra certa.",on_change=changeConfigModel)
            st.session_state.topK=st.slider("Top K",1,100,64,1,help="O **TopK** é semelhante ao TopP, mas funciona de uma forma ligeiramente diferente. Em vez de filtrar as palavras com base em suas probabilidades cumulativas, o TopK simplesmente seleciona as K palavras mais prováveis a cada passo da geração de texto. Isso significa que o TopK pode levar a resultados mais imprevisíveis, especialmente para valores baixos de K. É como escolher um número limitado de opções de um menu: um número menor de opções restringe suas escolhas, enquanto um número maior oferece mais flexibilidade.",on_change=changeConfigModel)
        pdf_docs=None
        pdf_docs = st.file_uploader("Carregue seus arquivos PDF e clique no botão \"Processar documentos:\"", type='.pdf', accept_multiple_files=True, help='Faça o upload de um livro ou apostila e faça perguntas para a IA ou qualquer outra solicitação como um questionário ou resumo.')  # Carregador de arquivos PDF
        if st.button("Processar documentos"):
            try:
                if pdf_docs == []:  
                    st.warning("Insira um ou mais documentos para análise", icon="⚠️")
                else:
                    with st.spinner("Processando..."):
                        st.session_state.docs_raw = get_pdf_text(pdf_docs)  # Lê o texto dos arquivos PDF e armazena na sessão                
                        st.success("Concluído",icon="✅")  # Exibe uma mensagem de sucesso
            except:
                st.error("Falha ao processar documentos",icon="❌")
        st.sidebar.button('Limpar histórico do chat', on_click=clear_chat_history)  # Botão para limpar o histórico do chat
        st.sidebar.link_button("Reportar Bug",'mailto:lucastadeusalomao@gmail.com')
        
def main():
    st.set_page_config(
        page_title="Assistente Virtual do Estudante de Medicina",  # Define o título da página
        page_icon="⚕️",  # Define o ícone da página
        menu_items={'Get Help': 'https://aistudio.google.com/app/apikey',  # Define os itens do menu
                   'Report a bug': "mailto:lucastadeusalomao@gmail.com",
                   'About': "Criado por Lucas Salomão"}
    )
    
    if 'docs_raw' not in st.session_state:
        st.session_state.docs_raw = ''
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Faça o upload de um livro ou apostila e faça perguntas para a IA ou qualquer outra solicitação como um questionário ou resumo."})  # Mensagem inicial do assistente
    if "docsEnviados" not in st.session_state:
        st.session_state.docsEnviados=False
        
    sidebar()
    st.image(BADGE, width=100)  # Exibe o logotipo sidebar
    
    # Main content area for displaying chat messages
    st.title("Assistente Virtual do Estudante de Medicina")  # Título da página
    st.write("Bem vindo ao assistente virtual do estudante de medicina!")  # Mensagem de boas-vindas
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])  # Exibe as mensagens do chat
    
    if(HABILITAR_CHAT):
        ##Testando prompt controlado
        if prompt := st.chat_input(placeholder="Faça alguma pergunta ou solicitação"):
            if(st.session_state.apiKeyGoogleAiStudio==""):
                st.warning("Por favor insira a chave de API",icon="🚨")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
        
        # Display chat messages and bot response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    genai.configure(api_key=st.session_state.apiKeyGoogleAiStudio)
                    if(st.session_state.docsEnviados):
                        response = get_gemini_reponse(prompt)
                    else:
                        response = get_gemini_reponse(prompt,st.session_state.docs_raw)
                        st.session_state.docsEnviados=True
                    placeholder = st.empty()
                    full_response = ''
                    if(STREAM_RESPONSE):
                        for chunk in response:
                            for ch in chunk.text.split(' '):
                                full_response += ch + ' '
                                time.sleep(0.05)
                                # Rewrites with a cursor at end
                                placeholder.markdown(full_response,unsafe_allow_html=True)
                    else:
                        placeholder.markdown(response.text,unsafe_allow_html=True)  # Exibe a resposta no placeholder
            if response.text is not None:
                message = {"role": "assistant", "content": response.text}
                st.session_state.messages.append(message)
                st.rerun()

if __name__ == "__main__":
    main()