# 🏠 Logi — l'assistant logement des étudiants

**Logi** est un assistant conversationnel qui répond aux questions de tous les étudiants sur le logement en France, avec une attention particulière pour ceux qui découvrent le système français : **aides au logement de la CAF (APL, ALS, ALF), garantie Visale et logement CROUS**. Il s'appuie **uniquement sur des sources officielles** et cite les pages consultées.

> 🟡 **Version 1 — bêta.** Première version publique, ouverte aux tests et aux retours.
>
> 🚧 **En cours de développement :** un module agent capable de réaliser certaines démarches directement pour l'utilisateur.
>
> **[▶ Tester la démo](#)** *(lien disponible après déploiement)*

---

## Le problème

Un étudiant qui cherche un logement doit jongler entre plusieurs sites officiels (CAF, Visale, CROUS, Service-Public), chacun avec son vocabulaire et ses règles. Les informations sont dispersées et parfois périmées selon les sources. La difficulté est encore plus grande pour les étudiants étrangers, qui découvrent à la fois le système français et des démarches propres à leur situation (titre de séjour, absence de garant en France…).

**L'objectif :** un seul point d'entrée où poser ses questions en langage naturel, avec des réponses fiables et sourcées.

## Fonctionnement

Logi repose sur une architecture **RAG** (*Retrieval-Augmented Generation*) : au lieu de laisser un modèle de langage répondre de mémoire, on va d'abord chercher les passages pertinents dans les sources officielles, puis on demande au modèle de rédiger sa réponse **à partir de ces passages uniquement**.

```
                         INDEXATION (ingest.py)
Pages officielles ──► Contrôle qualité ──► Nettoyage ──► Découpage ──► Embeddings ──► ChromaDB
Glossaire (data/) ─────────────────────────────────────────┘           (multilingue)

                         QUESTION (streamlit_app.py)
Question + historique ──► Reformulation ──► Recherche des 5 ──► LLM (Groq) ──► Réponse + sources
                          (question autonome)  extraits les plus proches
```

## Choix techniques

| Brique | Choix | Pourquoi |
|---|---|---|
| Sources | Fiches Service-Public, visale.fr, Action Logement | Officielles, lisibles par un programme et **datées** (« vérifié le… ») |
| Glossaire | Document rédigé et maintenu à la main | Les sites officiels ne définissent pas les notions de base (CAF, CROUS, VLS-TS…), pourtant essentielles pour un étudiant étranger |
| Contrôle qualité | Rejet des pages trop courtes ou suspectes | Un site en maintenance ou qui bloque les robots renvoie une page sans erreur : sans contrôle, elle serait indexée |
| Découpage | `RecursiveCharacterTextSplitter` (800 caractères, recouvrement 150) | Extraits assez courts pour être précis, avec du contexte |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` via FastEmbed | **Multilingue** : un modèle anglais dégradait nettement la recherche en français ; léger, sans PyTorch |
| Base vectorielle | ChromaDB | Simple, locale, embarquée dans l'application |
| Reformulation | `gpt-oss-20b` via Groq | Transforme une question de suivi (« résume la définition ») en question autonome (« résume la définition de la CAF ») avant la recherche |
| Génération | `gpt-oss-120b` via Groq | Réponses en quelques secondes, contre 1 à 2 minutes avec un modèle local sur CPU |
| Interface | Streamlit | Réponse en streaming, mémoire de conversation, sources consultées |

### Garde-fous

- Le modèle répond **uniquement à partir des extraits fournis**, et dit qu'il ne sait pas en renvoyant vers le site officiel sinon.
- Les **sources consultées** et la question réellement recherchée sont affichées sous chaque réponse.
- Les questions de suivi proposées se limitent à ce que les sources permettent de traiter.
- Limite de questions par visiteur et gestion des erreurs de l'API.
- Les clés API ne sont jamais versionnées (secrets Streamlit).

## Sources indexées

| Thème | Sources |
|---|---|
| Aides au logement (CAF) | [Service-Public — Aide au logement étudiant (APL, ALS, ALF)](https://www.service-public.gouv.fr/particuliers/vosdroits/F35848) |
| Garantie Visale | [Service-Public — Garantie Visale](https://www.service-public.gouv.fr/particuliers/vosdroits/F33453) · [Visale — Éligibilité](https://www.visale.fr/visale-pour-les-locataires/eligibilite/) · [Action Logement — Suivre sa demande](https://www.actionlogement.fr/demande-de-garantie-visale) |
| Logement CROUS | [Service-Public — Demander un logement étudiant au Crous](https://www.service-public.gouv.fr/particuliers/vosdroits/F31912) |
| Notions de base | Glossaire de l'assistant ([`data/glossaire.md`](data/glossaire.md)) |

## Lancer le projet en local

```bash
git clone https://github.com/fredSup/assistant-logement-rag.git
cd assistant-logement-rag
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Créer le fichier `.streamlit/secrets.toml` avec une clé [Groq](https://console.groq.com) :

```toml
GROQ_API_KEY = "votre_cle"
```

Puis :

```bash
python ingest.py                  # construit la base vectorielle
streamlit run streamlit_app.py    # lance l'interface
```

## Limites connues

- Le corpus couvre trois thèmes : les questions hors périmètre (titre de séjour, colocation, assurance…) ne sont pas encore traitées en détail.
- Pas encore d'évaluation chiffrée de la qualité des réponses : la bêta sert justement à collecter de vraies questions d'utilisateurs.
- Les consignes données au modèle sont suivies dans la grande majorité des cas, mais ne constituent pas une garantie absolue.

## Feuille de route

- [x] **v1 bêta** : RAG sur sources officielles, glossaire, mémoire de conversation, reformulation des questions
- [ ] Intégrer les retours des premiers testeurs
- [ ] **Module agent** : réaliser certaines démarches directement pour l'utilisateur
- [ ] Élargir le corpus (Loca-Pass, titre de séjour étudiant, colocation, assurance habitation)
- [ ] Jeu d'évaluation construit à partir des questions réelles, avec mesure de la qualité des réponses
- [ ] Recherche hybride (vecteurs + mots-clés) et reranking
- [ ] Date de vérification affichée pour chaque source

## Auteur

**Frédéric Moukengue** — Data Engineer & IA
[LinkedIn](https://www.linkedin.com/in/frederic-moukengue) · [GitHub](https://github.com/fredSup)

---

*Logi fournit une information générale et ne remplace pas les organismes officiels. Vérifiez toujours les conditions à jour avant toute démarche.*
