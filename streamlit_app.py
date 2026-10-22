import streamlit as st
from groq import Groq
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

# --- Configuration -----------------------------------------------------------
DB_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # identique a ingest.py
LLM_MODEL = "openai/gpt-oss-120b"     # redige les reponses
REWRITE_MODEL = "openai/gpt-oss-20b"  # plus petit et plus rapide : reformule les questions
K = 5                 # nombre d'extraits recuperes dans la base
HISTORY_TURNS = 3     # nombre d'echanges precedents gardes en memoire
MAX_QUESTIONS = 20    # limite par visiteur, pour eviter les abus une fois en ligne

SYSTEM_PROMPT = """Tu es Logi, un assistant IA en version beta qui aide tous les etudiants, \
avec une attention particuliere pour les etudiants etrangers, a comprendre leurs demarches \
de logement en France : aides au logement de la CAF (APL, ALS, ALF), garantie Visale et logement CROUS.

Regles :
- Reponds toujours en francais, de facon claire et concise. N'utilise ni titres ni \
lignes de separation (---) : des paragraphes courts, et des listes a puces seulement \
pour les conditions et les etapes.
- Pour toute information sur le logement (conditions, montants, demarches, delais, documents), \
appuie-toi UNIQUEMENT sur les extraits de documents officiels fournis.
- L'etudiant ne voit pas ces extraits : parle de "mes sources" ou "les documents officiels \
que je consulte", jamais du "texte fourni" ou du "contexte".
- Si l'information n'est pas dans les extraits, dis-le simplement et oriente vers \
le site officiel concerne (caf.fr, visale.fr, service-public.gouv.fr, lescrous.fr).
- N'invente jamais de condition, de montant, de delai ou de document.
- Si l'etudiant te salue ou te demande qui tu es et ce que tu fais, reponds brievement \
a partir de cette presentation, sans utiliser les extraits. Precise que tu ne remplaces \
pas les organismes officiels.
- Si l'etudiant demande de reformuler, simplifier, resumer ou illustrer une reponse \
precedente, travaille a partir de cette reponse. Un exemple doit rester fidele aux sources \
et etre presente comme une simple illustration.
- Termine par une ligne en gras "Vous pourriez aussi vous demander :" suivie de 2 \
questions de suivi maximum, UNIQUEMENT si les extraits permettent d'y repondre. \
Sinon, n'en propose aucune."""

REWRITE_PROMPT = """Tu recois l'historique d'une conversation et la derniere question d'un etudiant.
Reecris cette derniere question pour qu'elle soit autonome et comprehensible sans l'historique \
(remplace "ca", "la definition", "et pour lui"... par le sujet exact).
Si elle est deja autonome, renvoie-la telle quelle.
Ne reponds pas a la question. Renvoie uniquement la question reecrite, sur une seule ligne."""

# --- Interface ---------------------------------------------------------------
st.set_page_config(page_title="Assistant Logement", page_icon="🏠", layout="centered")
st.title("🏠 Assistant Logement")
st.caption("Votre assistant IA basé sur les sources officielles (CAF, Visale, CROUS, Service-Public)")
st.info("Version bêta : les réponses peuvent contenir des erreurs. "
        "Vérifiez toujours sur les sites officiels avant toute démarche.")


# --- Chargement des ressources (une seule fois) ------------------------------
@st.cache_resource
def load_vector_store():
    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


@st.cache_resource
def load_client():
    # La cle est lue dans les "secrets" : jamais ecrite dans le code.
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def reformuler(client, question, historique):
    """Transforme une question de suivi ("resume la definition") en question autonome
    ("resume la definition de la CAF"), pour que la recherche trouve les bons extraits."""
    if not historique:
        return question
    conversation = "\n".join(
        f"{'Etudiant' if m['role'] == 'user' else 'Assistant'} : {m['content'][:600]}"
        for m in historique
    )
    try:
        resp = client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"Historique :\n{conversation}\n\n"
                                            f"Derniere question : {question}"},
            ],
            temperature=0,
            max_tokens=500,
            reasoning_effort="low",
        )
        reecrite = (resp.choices[0].message.content or "").strip()
        return reecrite or question
    except Exception:
        return question  # en cas de probleme, on garde la question d'origine


def stream_answer(client, question, context, historique):
    """Envoie l'historique, les extraits et la question a Groq ; renvoie la reponse mot par mot."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": m["content"]} for m in historique]
    messages.append({"role": "user", "content": f"Extraits de documents officiels :\n{context}\n\n"
                                                f"Question de l'etudiant : {question}"})
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=2000,
        reasoning_effort="low",
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


vector_store = load_vector_store()
client = load_client()

# --- Historique de la conversation ------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Bonjour ! Je suis Logi, votre Assistant logement**. Posez-moi vos questions sur "
            "les **aides de la CAF**, la **garantie Visale** ou le **logement CROUS** : "
            "je vous réponds à partir de sources officielles.\n\n"
            "Par exemple : *« Quel loyer maximum pour Visale en Île-de-France ? »* "
            "ou *« Comment obtenir un logement CROUS en arrivant de l'étranger ? »*"
        ),
    }]
if "nb_questions" not in st.session_state:
    st.session_state.nb_questions = 0

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Nouvelle question --------------------------------------------------------
if user_query := st.chat_input("Ex : Quelles sont les conditions pour bénéficier de Visale ?"):
    # Memoire : les derniers echanges, sans le message d'accueil
    historique = st.session_state.messages[1:][-2 * HISTORY_TURNS:]

    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        if st.session_state.nb_questions >= MAX_QUESTIONS:
            response = ("Vous avez atteint la limite de questions pour cette démo. "
                        "Merci de votre visite !")
            st.write(response)
        else:
            st.session_state.nb_questions += 1

            # Etape 0 : rendre la question autonome grace a l'historique
            question_recherche = reformuler(client, user_query, historique)

            # Etape R : recherche des extraits pertinents
            docs = vector_store.similarity_search(question_recherche, k=K)
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)

            # Etape G : generation de la reponse par Groq
            try:
                response = st.write_stream(stream_answer(client, user_query, context, historique))
            except Exception:
                response = ("Le service est momentanément indisponible. "
                            "Merci de réessayer dans quelques instants.")
                st.write(response)

            sources = list(dict.fromkeys(doc.metadata.get("source", "Inconnue") for doc in docs))
            with st.expander("📚 Sources officielles consultées"):
                st.caption(f"Recherche effectuée pour : « {question_recherche} »")
                for source in sources:
                    if source.startswith("http"):
                        st.markdown(f"- [{source}]({source})")
                    else:
                        st.markdown(f"- {source}")

    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Bas de page --------------------------------------------------------------
st.caption("🚧 **Bientôt** : un module agent est en cours de développement. "
           "Il pourra réaliser certaines démarches directement pour vous.")
