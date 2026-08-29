# 🔬 ScholarlyForge AI: Scientific Paper Outline Generator

## 🎯 Overview

ScholarlyForge AI is an advanced framework generation tool engineered specifically for students and early-career researchers. By synthesizing abstract concepts and experimental source code into structured scientific paper outlines, this system automates the creation of LaTeX templates and BibTeX bibliographies. This automation allows researchers to dedicate their cognitive resources to core methodologies and empirical validation.

## ✨ Key Features

- 📂 **Multimodal File Ingestion**: Upload Python/JS scripts, PDFs, or DOCX files. The system parses your code logic alongside textual notes.
- 🧠 **Code-to-Methodology**: Synthesizes raw source code into structured "Methodology" and "Experimental Setup" sections.
- 📜 **Dynamic LaTeX Templating**: Generates journal-specific `.tex` files dynamically, adapting document classes and bibliography styles based on user selection.
- 🌐 **Literature Context**: Integrates with Semantic Scholar & arXiv to retrieve related works and identify research gaps.
- 📦 **Version History**: Automatically archives generated LaTeX projects locally in `.zip` format for easy retrieval.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### ⚙️ System Prerequisites

- **Python**: Version 3.10 or 3.11.
- **API Key**: A valid Google Gemini API Key.
- **LaTeX Distribution**: Optional, but recommended (e.g., TeX Live, MiKTeX) to compile the output locally, though you can also use Overleaf.

For systems utilizing the Chocolatey package manager, install the necessary multimedia and typesetting dependencies via the following command:

```bash
choco install ffmpeg miktex
```

### 🛠️ Installation Instructions

Execute the following sequence to isolate dependencies and configure the operational environment.

**1. Virtual Environment Initialization**

Establish a secure, isolated Python environment using the terminal:

```bash
python -m venv env
```

**2. Environment Activation**

Activate the virtual environment corresponding to your operating system:

*For Windows:*
```cmd
env\Scripts\activate
```

*For macOS and Linux:*
```bash
source env/bin/activate
```

**3. Dependency Installation**

Ensure the `requirements.txt` file is located within the active directory. Execute the following command to install all mandatory libraries:

```bash
pip install -r requirements.txt
```

To force a comprehensive update of all currently installed packages, execute:

```bash
pip install -U -r requirements.txt
```

### 🔑 Application Configuration

The system utilizes the Google Gemini API for natural language reasoning and analytical processing.

1. Create a file named `.env` in the root directory of the application.
2. Define the API key variable precisely as follows:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### ▶️ Execution Protocol

Launch the Streamlit web interface by executing the following command in the terminal:

```bash
streamlit run app.py
```

## 📄 License and Attribution

This software is distributed under the MIT License. It remains free and open-source to support the academic community. Any utilization, modification, or distribution of this codebase requires explicit attribution to the original creator, [tandoancs](https://github.com/tandoancs).