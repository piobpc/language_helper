# Na początek importujemy niezbędne biblioteki
import streamlit as st
from dotenv import dotenv_values
from openai import OpenAI
import instructor
from pydantic import BaseModel
from typing import List #do analizy gramatycznej (pomoc GPT)
from io import BytesIO
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid 
from bs4 import BeautifulSoup

# dodajemy Qdrant i kolekcję o odpowiedniej nazwie
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
QDRANT_COLLECTION_NAME = "pomocnik_jezykowy"

# pobieranie klucza OpenAI
def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

env = dotenv_values('.env')
### Secrets using Streamlit Cloud Mechanism
# https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']
###

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

if not st.session_state["openai_api_key"]:
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
    else:
        # pokazujemy pole input tylko gdy klucz jest pusty
        api_key_input = st.text_input(
            "Aby skorzystać z Pomocnika Językowego, podaj najpierw swój klucz API OpenAI",
            type="password",
            placeholder="sk-..."
        )
        if api_key_input:
            st.session_state["openai_api_key"] = api_key_input
            st.rerun()  # odświeżanie aplikacji, żeby schować input
        else:
            st.stop()  # bez klucza zatrzymujemy dalsze działanie

openai_client = get_openai_client()
instructor_openai_client = instructor.from_openai(openai_client)

# sprawdzanie dostępu do klucza OpenAI
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
    else:
        st.info("Podaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

# klasa dla tokenu z analizą gramatyczną
class AnalizaTokenowa(BaseModel):
    token: str
    partie_tekstu: str
    wyjasnienie: str

# klasa zwracana przez model
class AnalizaGramatyczna(BaseModel):
    zdania: str
    tokeny: List[AnalizaTokenowa]

def _format_note_html(tłumaczenie: str, tokeny: List[AnalizaTokenowa]) -> str:    # <-- GPT
    html = [f"<p><strong>{tłumaczenie}</strong></p>"]                            # <-- GPT
    for t in tokeny:                                                             # <-- GPT
        html.append(f"<span style='color:#5dade2;font-weight:bold'>{t.token}</span> — {t.wyjasnienie}<br/>")   # <-- GPT
    return "\n".join(html)     # <-- GPT

# generowanie mowy
def generowanie_audio(html_or_text: str, voice: str = "alloy"):
    # zamień HTML → tekst
    text = BeautifulSoup(html_or_text, "html.parser").get_text(" ", strip=True)
    resp = openai_client.audio.speech.create(
        model="tts-1",
        voice=voice,
        response_format="mp3",
        input=text,
    )
    # odtworzenie w Streamlit
    st.audio(BytesIO(resp.read()), format="audio/mp3")

#
# BAZA DANYCH 
#
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
    url=env["QDRANT_URL"],
    api_key=env["QDRANT_API_KEY"],
)

def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        print("Tworzę kolekcję")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
    else:
        print("Kolekcja już istnieje")

assure_db_collection_exists()  # <-- GPT

def get_embedding(text):
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )

    return result.data[0].embedding

def add_note_to_db(note_text):
    assure_db_collection_exists() 
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        exact=True,
    )
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        wait=True, # <-- GPT
        points=[
            PointStruct(
                id=str(uuid.uuid4()),    # <-- GPT
                vector=get_embedding(text=note_text),
                payload={
                    "text": note_text,
                },
            )
        ]
    )

def list_notes_from_db(query=None):
    qdrant_client = get_qdrant_client()
    if not query:
        notes = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=10)[0]
        result = []
        for note in notes:
            result.append({
                "text": note.payload["text"],
                "score": None,
            })

        return result

    else:
        notes = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embedding(text=query),
            limit=10,
        )
        result = []
        for note in notes:
            result.append({
                "text": note.payload["text"],
                "score": note.score,
            })

        return result
    
def _save_current_note():
    """Zapisuje aktualny st.session_state['note_text'] do Qdrant.""" # <-- GPT
    if st.session_state.get("note_text"):                   # <-- GPT
        add_note_to_db(st.session_state["note_text"])       # <-- GPT
        st.toast("Notatka zapisana ✅", icon="📒")        # <-- GPT
        

# inicjalizacja session state
if "note_text" not in st.session_state:
    st.session_state["note_text"] = ""

# inicjujemy także flagi dla przycisków „perm”     # <-- (GPT)
if "save_tab1_perm" not in st.session_state:      # <-- (GPT)
    st.session_state["save_tab1_perm"] = False    # <-- (GPT)
if "save_tab2_perm" not in st.session_state:      # <-- (GPT)
    st.session_state["save_tab2_perm"] = False    # <-- (GPT)

#
# MAIN
#

# tworzymy interfejs aplikacji
st.title(':tiger: Pomocnik Językowy')

# dorzucamy pomocnicze pole jako wskazówki do użycia aplikacji
with st.expander("ℹ️ Jak korzystać z aplikacji?", expanded=False):
    with st.container():
        st.markdown(
            """
            <style>
            .custom-box {
                background-color: transparent;
                padding: 20px;
                color: white;
            }
            </style>
            <div class="custom-box">
                <h4>Cześć!</h4>
                <h4>Jestem Twoim osobistym Pomocnikiem Językowym!</h4>
                <p>1. Mogę dla Ciebie przetłumaczyć tekst na wybrany język obcy. </p>
                <p>2. Mogę również pomóc w sprawdzeniu pisowni dowolnie wpisanego języka. </p2>
                <p>3. Otrzymasz ode mnie analizę w postaci notatki tekstowej, którą możesz sobie zedytować i zapisać na później.  </p3>
                <p>4. Możesz również odsłuchać analizę w formie dźwiękowej.</p4>
                <h3> Miłego korzystania!</h3>
                </div>
            """,
            unsafe_allow_html=True
        )

tab1, tab2, tab3 = st.tabs(["Tłumaczenie z polskiego na inny język", "Sprawdzanie pisowni w języku obcym", "Zapisane tłumaczenia"]) 

with tab1: 
    tekst_pl = st.text_area('Wprowadź tekst po polsku do przetłumaczenia', height=120)

# dodajemy pole wyboru języka docelowego 
    jezyki = ['angielski', 'niemiecki', 'francuski', 'hiszpański', 'włoski', 'portugalski', 
            'szwedzki', 'norweski', 'duński', 'ukraiński', 'rosyjski', 'turecki', 'mandaryński']
    wybor = st.selectbox('Wybierz język docelowy: ', jezyki)
# tłumaczenie PL-wybrany język
    if st.button("Przetłumacz"):
        with st.spinner("Już tłumaczę..."):
            res_pl_obcy = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"Jesteś tłumaczem. 'Przetłumacz {tekst_pl} na ten język: {wybor}. Nie pisz nic więcej."},
                    {"role": "user", "content": tekst_pl}
                ]
            )
            st.success("Tłumaczenie:")
            output = res_pl_obcy.choices[0].message.content.strip()
            st.write(output)

        with st.spinner("Analiza tekstu..."):
            res_pl_obcy_komentarz = instructor_openai_client.chat.completions.create(
                model="gpt-4o",
                temperature=0, 
                response_model=AnalizaGramatyczna,
                messages=[
                    {"role": "system", "content": f"""Jesteś nauczycielem gramatyki angielskiej. Rozbij zdanie na części mowy i opisz każdy wyraz. 
                    Używaj tylko polskich nazw części mowy, np. rzeczownik, czasownik, przymiotnik, zaimek, pomiń znaki interpunkcyjne. Użyj prostych opisów do zrozumienia"""},
                    {"role": "user", "content": f"""Przeanalizuj gramatycznie {output} i zwróć analizę {output} w języku polskim, bez trudnych słów typu 'gerundium' itd"""}
                ]
            )
            st.success("Analiza:")

# pętla na wypisywanie analizy:
            for token in res_pl_obcy_komentarz.tokeny:
                st.markdown(f"<span style='color:#5dade2; font-weight:bold'>{token.token}</span> — {token.wyjasnienie}", unsafe_allow_html=True)
                st.text("")

# zamiana formatu analizy na jeden duży string, aby móc przekownertować na audio
            analiza_str1 = _format_note_html(output, res_pl_obcy_komentarz.tokeny)  

            st.session_state["note_text"] = analiza_str1  # <-- GPT
            st.session_state["note_source"] = "tab1"       # <-- GPT

            st.markdown("**Analiza w formie audio:**")
            generowanie_audio(
                analiza_str1,
                voice="alloy",
            )

            if st.session_state.get("note_text"):                         # <-- GPT
                st.button("💾 Zapisz tłumaczenie", key="save_tab1",          # <-- GPT
                on_click=_save_current_note)                  # <-- GPT

with tab2: 
# dodajemy alternatywny input dla j. obcego
    tekst_obcy = st.text_area('Wprowadź tekst w j. obcym do sprawdzenia', height=120)

# sprawdzanie błędów
    if st.button("Sprawdź"):
        with st.spinner("Już sprawdzam..."):
            res_korekta = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"""Jesteś korektorem. Sprawdź wprowadzony 
                    tekst {tekst_obcy} pod kątem błędów i pokaż poprawiony tekst. Nie pokazuj nic innego poza poprawionym tekstem {tekst_obcy}."""},
                    {"role": "user", "content": f"""Wypisz poprawiony {tekst_obcy}"""}
                ]
            )
            st.success("Po sprawdzeniu:")
            output2 = res_korekta.choices[0].message.content.strip()
            st.write(output2)

        with st.spinner("Analiza tekstu..."):
            res_korekta_komentarz = instructor_openai_client.chat.completions.create(
                model="gpt-4o",
                temperature=0, 
                response_model=AnalizaGramatyczna,
                messages=[
                    {"role": "system", "content": f"""Jesteś nauczycielem gramatyki angielskiej. Rozbij zdanie na części mowy i opisz każdy wyraz. 
                    Używaj tylko polskich nazw części mowy, np. rzeczownik, czasownik, przymiotnik, zaimek, pomiń znaki interpunkcyjne. Użyj prostych opisów do zrozumienia"""},
                    {"role": "user", "content": f"""Przeanalizuj gramatycznie {output2} i zwróć analizę {output2} w języku polskim, bez trudnych słów typu 'gerundium' itd"""}
                ]
            )

            st.success("Analiza:")
# pętla na wypisywanie analizy
            for token in res_korekta_komentarz.tokeny:
                st.markdown(f"<span style='color:#5dade2; font-weight:bold'>{token.token}</span> — {token.wyjasnienie}", unsafe_allow_html=True)
                st.text("")
            
# zamiana formatu analizy na jeden duży string, aby móc przekownertować na audio
            analiza_str2 = _format_note_html(output2, res_korekta_komentarz.tokeny)  

            st.session_state["note_text"] = analiza_str2  # <-- GPT
            st.session_state["note_source"] = "tab2"    # <-- GPT

            st.markdown("**Analiza w formie audio:**")
            generowanie_audio(
                analiza_str2,
                voice="alloy",
            )

            if st.session_state.get("note_text"):                # <-- GPT
                st.button("💾 Zapisz tłumaczenie", key="save_tab2", # <-- GPT
                on_click=_save_current_note)                  # <-- GPT


with tab3:
    query = st.text_input("🔍 Wyszukaj tłumaczenie")

    if query:
        similar_notes = list_notes_from_db(query=query)

        if similar_notes:
# wyświetl pierwszą (najlepszą) notatkę
            st.markdown("### Najlepiej dopasowana notatka:")
            with st.container():
                st.markdown(similar_notes[0]["text"], unsafe_allow_html=True)

# jeśli jest więcej notatek, pokaż je jako "Inne notatki"
            if len(similar_notes) > 1:
                st.markdown("### Inne notatki:")
                for i, note in enumerate(similar_notes[1:], start=1):
                    with st.container():
                        st.markdown(note["text"], unsafe_allow_html=True)
                    if i < len(similar_notes[1:]):                 # żeby nie dodawać pustki za ostatnią
                        st.markdown("<br>", unsafe_allow_html=True)  # jednoliniowy odstęp
        else:
            st.warning("Brak pasujących notatek.")
    else:
        st.info("Wpisz słowo lub zdanie, aby zobaczyć podobne notatki.")
                
