# 🔍 AI Research Assistant

An intelligent **research automation tool** powered by **Google Gemini AI**, **DuckDuckGo Search**, and **BeautifulSoup**.  
It performs **automated research** on any topic by searching the web, extracting content, summarizing findings, analyzing key insights, and generating a **comprehensive research report** in both **Markdown** and **JSON** formats.


## 🚀 Features

✅ **AI-Powered Summarization & Analysis** using Google Gemini  
✅ **Automated Web Search** via DuckDuckGo Search API  
✅ **Content Extraction** with BeautifulSoup  
✅ **Relevance Scoring** for ranking sources  
✅ **Named Entity Extraction** (dates, URLs, emails)  
✅ **Comprehensive Research Report** generation (Markdown + JSON)  
✅ **Interactive CLI Interface** for easy use  
✅ **Offline fallback mode** (if no Gemini API key is set)


## 🧠 How It Works

1. **User provides a research topic.**  
2. The assistant generates multiple **search queries** related to the topic.  
3. It performs **web and news searches** using DuckDuckGo.  
4. Extracts and cleans content from top URLs.  
5. Summarizes and analyzes content using **Google Gemini AI**.  
6. Combines insights into a **final report** with:
   - Executive Summary  
   - Key Findings  
   - Detailed Analysis  
   - Recommendations  
   - Source List  



## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Raj4884/Task.git
cd Task
````

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

### 3️⃣ Activate the Environment

#### On Windows (PowerShell)

```bash
venv\Scripts\activate
```

#### On macOS/Linux

```bash
source venv/bin/activate
```

---

## 📦 Requirements

All necessary dependencies are listed below.
You can install them using the command after the list.

### 📋 requirements.txt

```txt
requests
beautifulsoup4
duckduckgo-search
google-generativeai
python-dotenv
```

### 🧩 Install dependencies

```bash
pip install -r requirements.txt
```

*(Alternatively, just copy the above list into a `requirements.txt` file if needed.)*

---

## 🔑 Configure Google Gemini API

Get your Gemini API key from
👉 [Google AI Studio (Makersuite)](https://makersuite.google.com/app/apikey)

Then set it as an environment variable:

### On Windows:

```bash
setx GEMINI_API_KEY "your_api_key_here"
```

### On macOS/Linux:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Alternatively, create a `.env` file in the project folder:

```
GEMINI_API_KEY=your_api_key_here
```

---

## 🧭 Usage

### Run in Interactive CLI Mode

```bash
python research_assistant.py
```

**Example session:**

```
🔍 AI Research Assistant powered by Google Gemini
📝 Enter research topic: Artificial Intelligence in Healthcare
Include recent news? (y/n): y
```

The tool will:

* Search for sources
* Extract and analyze content
* Generate `research_report_YYYYMMDD_HHMMSS.md` and `.json`

---

## 📂 Output Example

### Markdown Report

```markdown
# Research Report: Artificial Intelligence in Healthcare

**Generated:** 2025-11-09T13:25:00  
**AI Model:** Google Gemini Pro  

## Executive Summary
AI is transforming healthcare through data-driven diagnostics and treatment personalization...

## Key Findings
1. AI improves early disease detection.
2. Ethical concerns remain regarding patient data privacy.

## Recommendations
1. Implement strict data governance.
2. Focus on explainable AI frameworks.
```
