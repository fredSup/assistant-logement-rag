"""
Indexation des sources dans ChromaDB.
A relancer a chaque fois que tu ajoutes une source ou que tu modifies le glossaire.

Deux types de sources :
  1. des pages web officielles (URLS ci-dessous) ;
  2. des documents locaux rediges a la main (dossier data/), comme le glossaire.
"""
import os
import shutil

from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_DIR = "./chroma_db"
DATA_DIR = "./data"

# Modele d'embedding MULTILINGUE. streamlit_app.py doit utiliser exactement le meme.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Sources web, classees par theme.
# On privilegie Service-Public : pages officielles, lisibles par un programme, et datees.
URLS = [
    # --- Aides au logement (CAF : APL, ALS, ALF) ---
    "https://www.service-public.gouv.fr/particuliers/vosdroits/F35848",
    # --- Garantie Visale ---
    "https://www.service-public.gouv.fr/particuliers/vosdroits/F33453",
    "https://www.visale.fr/visale-pour-les-locataires/eligibilite/",
    "https://www.actionlogement.fr/demande-de-garantie-visale",
    # --- CROUS : logement en residence universitaire ---
    "https://www.service-public.gouv.fr/particuliers/vosdroits/F31912",
]

# Controle qualite : une page trop courte ou contenant ces mots n'est pas un vrai
# contenu (site en maintenance, blocage anti-robot...). On l'ecarte au lieu de l'indexer.
TAILLE_MIN = 1500
MOTS_SUSPECTS = ["maintenance", "access denied", "captcha", "enable javascript",
                 "activez javascript", "page introuvable", "erreur 404"]


def nettoyer(texte: str) -> str:
    """Retire le bruit des pages web (menus, boutons, liens isoles).
    Filtre simple : on garde les lignes d'au moins 4 mots."""
    lignes = [l.strip() for l in texte.splitlines()]
    return "\n".join(l for l in lignes if len(l.split()) >= 4)


def page_valide(doc) -> bool:
    texte = doc.page_content.lower()
    if len(texte) < TAILLE_MIN:
        print(f"   [ECARTEE] trop courte ({len(texte)} caracteres) : {doc.metadata['source']}")
        return False
    for mot in MOTS_SUSPECTS:
        if mot in texte:
            print(f"   [ECARTEE] contient '{mot}' : {doc.metadata['source']}")
            return False
    return True


def charger_pages_web():
    print("1. Chargement des pages web...")
    docs = []
    for url in URLS:
        # Une page a la fois : si un site est en panne, les autres sont quand meme chargees.
        try:
            loader = WebBaseLoader(url, header_template={
                "User-Agent": "Mozilla/5.0 (assistant-logement-rag)"})
            for doc in loader.load():
                doc.page_content = nettoyer(doc.page_content)
                if page_valide(doc):
                    print(f"   [OK] {len(doc.page_content)} caracteres : {url}")
                    docs.append(doc)
        except Exception as e:
            print(f"   [ERREUR] {url} : {e}")
    return docs


def charger_documents_locaux():
    print("2. Chargement des documents locaux (data/)...")
    docs = []
    if not os.path.isdir(DATA_DIR):
        print("   Aucun dossier data/ : etape ignoree.")
        return docs
    for nom in sorted(os.listdir(DATA_DIR)):
        if nom.endswith((".md", ".txt")):
            chemin = os.path.join(DATA_DIR, nom)
            for doc in TextLoader(chemin, encoding="utf-8").load():
                doc.metadata["source"] = f"Glossaire de l'assistant ({nom})"
                print(f"   [OK] {len(doc.page_content)} caracteres : {nom}")
                docs.append(doc)
    return docs


def main():
    # On repart d'une base vide pour ne jamais melanger anciennes et nouvelles donnees.
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    docs = charger_pages_web() + charger_documents_locaux()
    if not docs:
        print("Aucun document valide : indexation annulee.")
        return

    print("3. Decoupage en morceaux...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"   -> {len(chunks)} morceaux a partir de {len(docs)} documents")

    print("4. Vectorisation et stockage dans ChromaDB...")
    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_DIR)

    print("Indexation terminee.")


if __name__ == "__main__":
    main()
