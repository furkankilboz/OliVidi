

import os
import glob as globmod
import urllib.request

import ollama
import chromadb

BASE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
OLLAMA_MODEL = "llama3"
COLLECTION_NAME = "olive_docs"

SYSTEM_PROMPT = (
    "You are OliVidi, an expert olive tree disease consultant. "
    "Based on the scientific knowledge provided, give a detailed consultation report.\n\n"
    "Format your response as:\n"
    "1. **Diagnosis Summary**\n"
    "2. **Severity Assessment**\n"
    "3. **Recommended Treatment**\n"
    "4. **Prevention Measures**\n"
    "5. **Additional Notes**"
)

REPORT_TEMPLATE = """### 1. Diagnosis Summary
{diagnosis}

### 2. Severity Assessment
{severity}

### 3. {treatment_heading}
{treatment}

### 4. Prevention Measures
{prevention}

### 5. Additional Notes
{notes}

---
*Report generated offline from local knowledge base (Ollama not connected).*"""


def _ollama_available():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3).close()
        return True
    except Exception:
        return False



def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            return collection
    except Exception:
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)
    docs = []
    for path in sorted(globmod.glob(os.path.join(DOCS_DIR, "**", "*.txt"), recursive=True)):
        with open(path, "r", encoding="utf-8") as f:
            docs.append(f.read())

    if not docs:
        raise RuntimeError("No .txt files found in docs/")

    chunks, ids = [], []
    for i, doc in enumerate(docs):
        words = doc.split()
        for j in range(0, len(words), 150):
            chunk = " ".join(words[j:j + 200])
            chunks.append(chunk)
            ids.append(f"doc{i}_chunk{j}")

    embeddings = []
    for chunk in chunks:
        resp = ollama.embed(model=OLLAMA_MODEL, input=chunk)
        embeddings.append(resp["embeddings"][0])

    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection


def _ollama_consultation(label, confidence):
    collection = _get_collection()

    if label == "healthy":
        query = (
            f"The olive leaf is HEALTHY ({confidence:.1%} confidence). "
            "Give a health status report and maintenance recommendations."
        )
    else:
        query = (
            f"The olive leaf is DISEASED — olive peacock eye spot ({confidence:.1%} confidence). "
            "Give diagnosis, severity, treatment with fungicides/dosages, and prevention."
        )

    query_emb = ollama.embed(model=OLLAMA_MODEL, input=query)["embeddings"][0]
    results = collection.query(query_embeddings=[query_emb], n_results=3)
    context = "\n\n".join(results["documents"][0])

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Scientific Knowledge:\n{context}\n\nQuestion: {query}"},
        ],
        options={"temperature": 0.3},
    )
    return response["message"]["content"]


# Offline fallback

def _offline_consultation(label, confidence):
    if label == "healthy":
        return REPORT_TEMPLATE.format(
            diagnosis=(
                f"The olive leaf has been classified as **HEALTHY** with "
                f"**{confidence:.1%}** confidence. No visible signs of disease detected."
            ),
            severity="No disease detected — the tree appears to be in good health.",
            treatment_heading="Maintenance Recommendations",
            treatment=(
                "- Apply **preventive copper-based fungicide** (Bordeaux mixture) in autumn "
                "(October-November) and late winter (February-March).\n"
                "- **Prune regularly** to improve air circulation and light penetration.\n"
                "- Remove fallen leaves to reduce fungal inoculum.\n"
                "- Use **drip irrigation** instead of overhead watering.\n"
                "- Apply balanced fertilization — avoid excessive nitrogen."
            ),
            prevention=(
                "- Monitor leaves during cool, wet periods (autumn/spring).\n"
                "- Consider tolerant cultivars (Leccino, Frantoio, Koroneiki) for new groves.\n"
                "- Maintain proper tree spacing for air movement."
            ),
            notes=(
                "Healthy olive leaves are dark green on top, silvery-gray beneath, "
                "and remain on the tree for 2-3 years. Continue routine inspections."
            ),
        )

    return REPORT_TEMPLATE.format(
        diagnosis=(
            f"The olive leaf has been classified as **DISEASED** (Olive Peacock Eye Spot "
            f"— *Spilocaea oleaginea*) with **{confidence:.1%}** confidence. "
            f"This is the most common foliar disease of olive trees worldwide."
        ),
        severity=(
            "Peacock eye spot can cause **20-30% defoliation** in severe cases, "
            "leading to significant yield losses and reduced oil content. "
            "Early treatment is critical."
        ),
        treatment_heading="Recommended Treatment",
        treatment=(
            "**Chemical Control:**\n"
            "- Apply **copper-based fungicides** (Bordeaux mixture, copper hydroxide, "
            "copper oxychloride).\n"
            "- **Timing:** First in October-November, second in February-March.\n"
            "- **Dosage:** 300-500 g metallic copper/ha (follow label).\n"
            "- Severe cases: systemic fungicides (**difenoconazole** or **tebuconazole**).\n"
            "- Rotate fungicide classes to prevent resistance.\n\n"
            "**Cultural Practices:**\n"
            "- **Prune** to improve air circulation and light penetration.\n"
            "- **Remove and destroy** fallen infected leaves.\n"
            "- Switch to **drip irrigation** (avoid overhead watering).\n"
            "- Maintain proper tree spacing.\n"
            "- Avoid excessive nitrogen fertilization."
        ),
        prevention=(
            "- Monitor during cool wet weather (5-20 C, >80% humidity).\n"
            "- Use weather-based disease models to time applications.\n"
            "- Tolerant cultivars: **Leccino, Frantoio, Koroneiki**.\n"
            "- Susceptible cultivars: Coratina, Ascolana, Picholine."
        ),
        notes=(
            "- Biological control (*Bacillus subtilis*, *Trichoderma* spp.) is promising "
            "but limited commercially.\n"
            "- Organic-approved copper formulations are available.\n"
            "- Keep records of disease severity across seasons."
        ),
    )


def get_consultation(disease_label, confidence):
    if _ollama_available():
        return _ollama_consultation(disease_label, confidence)
    return _offline_consultation(disease_label, confidence)


if __name__ == "__main__":
    mode = "RAG (Ollama)" if _ollama_available() else "Offline"
    print(f"OliveConsultantAgent — {mode} mode")
    print(get_consultation("diseased", 0.92))
