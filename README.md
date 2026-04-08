# AI-Powered-Startup-Legal-Compliance-Assistant-using-Retrieval-Augmented-Generation-RAG-
# ⚖️ AI-Powered Startup Legal Compliance Assistant (RAG System)

## 📌 Project Overview

Startups and small businesses must comply with multiple legal and regulatory requirements such as company registration laws, taxation policies, labour regulations, and government startup initiatives. However, these regulations are documented in large legal documents that are difficult for entrepreneurs to interpret.

This project builds an **AI-Powered Legal Compliance Assistant** using the **Retrieval-Augmented Generation (RAG)** architecture. The system retrieves relevant information from official legal documents and generates accurate, context-aware responses using a Large Language Model.

The assistant allows startup founders and business owners to quickly understand legal requirements without manually searching through hundreds of pages of legal text.

---

# 🚀 Key Features

* Retrieval-Augmented Generation (RAG) architecture
* Semantic search using vector embeddings
* Chroma vector database with local persistence
* Context-aware answers using LLM
* Source citation for transparency
* Streamlit interactive chatbot interface
* Fallback mechanism when documents are not found

---

# 🧠 System Architecture

```
Legal Documents (PDF)
        │
        ▼
PyMuPDFLoader
        │
        ▼
Recursive Text Splitter
        │
        ▼
HuggingFace Embeddings
        │
        ▼
Chroma Vector Database
        │
        ▼
Retriever
        │
        ▼
Large Language Model (Groq - LLaMA 3.3 70B)
        │
        ▼
Generated Answer
```

---

# 📂 Project Structure

```
AI-Legal-Compliance-Assistant
│
├── rag_data_documents
│   ├── companies_act_2013.pdf
│   ├── cgst_act_2017.pdf
│   ├── cgst_rules_2017.pdf
│   ├── code_on_wages_2019.pdf
│   └── startup_india_action_plan.pdf
│
├── chroma_db
│   └── ... persisted Chroma collection files ...
│
├── rag_load_to_emb.py
├── app.py
├── requirements.txt
│
├── README.md
└── problem statement.md
```

---

# 📚 Dataset / Knowledge Base

The system uses official legal and policy documents related to Indian startup compliance.

### Documents Used

| Document                  | Description                                |
| ------------------------- | ------------------------------------------ |
| Companies Act 2013        | Corporate governance and company formation |
| CGST Act 2017             | Goods and Services Tax legislation         |
| CGST Rules 2017           | GST compliance procedures                  |
| Code on Wages 2019        | Labour and wage regulations                |
| Startup India Action Plan | Government initiatives for startups        |

---

# ⚙️ Technologies Used

### Programming

* Python

### Frameworks & Libraries

* LangChain
* HuggingFace Embeddings
* ChromaDB
* Streamlit
* PyMuPDF

### LLM

* Groq API
* LLaMA 3.3 70B

### Concepts

* Retrieval Augmented Generation (RAG)
* Semantic Search
* Vector Databases
* Prompt Engineering

---

# 🛠️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/legal-compliance-rag.git
cd legal-compliance-rag
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Setup Environment Variables

Create a `.env` file.

```env
GROQ_API_KEY=your_api_key_here
```

---

# 📊 Step 1: Build Vector Database

Run the ingestion script to process legal documents.

```bash
python rag_load_to_emb.py
```

This script:

* Loads PDF legal documents
* Splits text into chunks
* Generates embeddings with the `BAAI/bge-base-en-v1.5` model
* Stores vectors in a persisted local Chroma database

---

# 💬 Step 2: Run Chat Application

```bash
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

# ❓ Example Queries

The assistant can answer questions like:

* What documents are required to register a private limited company?
* What is the GST registration threshold?
* What penalties apply for late GST filing?
* What are minimum wage requirements under the Code on Wages?
* What benefits are provided under the Startup India initiative?

---

# 🔎 Retrieval + Fallback Logic

The system implements a **two-stage answering process**.

### Case 1: Relevant documents found

* Retrieve top similar chunks
* Generate answer using retrieved context

### Case 2: No relevant documents

* Use LLM general knowledge
* Provide fallback answer

This ensures robustness and better user experience.

---

# 👥 Target Users

* Startup Founders
* Small Business Owners
* Compliance Officers
* Legal Researchers
* Entrepreneurship Students

---

# 🎯 Expected Impact

This system simplifies legal research for startups by:

* Reducing time spent reading legal documents
* Providing contextual answers
* Improving accessibility of legal information
* Supporting regulatory compliance

---

# 🔮 Future Improvements

* Add more legal documents
* Implement hybrid search (BM25 + Vector)
* Add citation highlighting
* Deploy on cloud (AWS / GCP)
* Build API service
* Add multilingual support

---

# 👨‍💻 Author

**Venkata Dinesh Kumar Reddy**

Aspiring Data Scientist | AI & ML Enthusiast

Skills:

* Python
* SQL
* Machine Learning
* NLP
* Retrieval-Augmented Generation (RAG)

---

# 📜 License

This project is developed for educational and research purposes.
