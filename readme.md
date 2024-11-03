# Résumé du Script

Le script permet de crawler un site web, d'extraire le texte des pages HTML et PDF, puis de générer des embeddings à l'aide de l'API OpenAI.

## Importations
- Gestion des requêtes HTTP
- Parsing HTML
- Extraction de texte PDF
- Manipulation de données
- Multithreading
- Interaction avec l'API OpenAI

## Configuration
- **Logging** : Enregistrement des événements et erreurs dans `crawler_log.txt`.
- **Patterns d'URL** : Validation des URLs à crawler via expressions régulières.
- **Domaine Cible** : Définition du domaine racine (`www.ouellet.com`) et de l'URL de départ.
- **User-Agent** : Simulation d'agents utilisateurs avec `fake_useragent`.
- **Paramètres de Requête** : 
  - Tentatives maximales (`MAX_RETRIES`)
  - Délai entre tentatives (`RETRY_DELAY`)
  - Nombre maximal de threads (`MAX_WORKERS`)
  - Pause aléatoire entre requêtes (`SLEEP_TIME`)
- **Tailles des Chunks** : Configuration des segments de texte en tokens (`CHUNK_SIZES`).

## Crawling et Extraction de Texte
- **Parser d'Hyperliens** : Extraction des liens internes dans les pages HTML.
- **Nettoyage des URLs et Noms de Fichiers** : Normalisation et sanitisation.
- **Extraction de Texte** :
  - Depuis HTML avec `BeautifulSoup`
  - Depuis PDF avec `pdfplumber`
- **Normalisation des URLs** : Assure la consistance des URLs traitées.
- **Traitement des URLs** : Gestion des requêtes HTTP, identification du type de contenu, et extraction du texte.

## Fonctions d'Embedding
- **Configuration de l'API OpenAI** : Initialisation avec la clé API et gestion des tokens.
- **Division du Texte en Chunks** : Segmentation respectant les limites de tokens.
- **Génération des Embeddings** : Création de vecteurs via l'API OpenAI.
- **Traitement et Sauvegarde** : Parcours des fichiers, génération des embeddings, et sauvegarde en formats CSV, JSON, NPY.

## Exécution Principale
- **Crawling** : Extraction du texte des pages web et PDF.
- **Génération des Embeddings** : Traitement des fichiers texte et création des embeddings.
- **Sauvegarde des Résultats** : Stockage structuré des embeddings générés.
