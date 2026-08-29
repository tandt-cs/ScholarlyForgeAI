import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import datetime
import re
import os
# Xóa import ssl, requests, urllib3 nếu không còn dùng ở đâu khác
import csv
from dotenv import load_dotenv
from typing import List, Optional

# File processing libraries
import PyPDF2
import docx

# API and AI libraries
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from semanticscholar import SemanticScholar
import arxiv

# Data processing and Templating libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jinja2

# ==========================================
# 1. PYDANTIC SCHEMAS DEFINITION
# ==========================================
class NoveltyAnalysis(BaseModel):
    research_gaps: List[str] = Field(
        description="Blind spots or limitations of the current literature"
    )
    proposed_contribution: str = Field(
        description="Core contribution of the proposed idea to bridge the research gap"
    )

class PaperSubSection(BaseModel):
    title: str = Field(description="Subsection title")
    content_summary: str = Field(description="Detailed summary of the subsection content")

class PaperSection(BaseModel):
    title: str = Field(description="Section title (e.g., Introduction, Methodology)")
    content_summary: str = Field(
        description="Detailed summary of the content to be written, inserting citation keys if required"
    )
    subsections: Optional[List[PaperSubSection]] = Field(
        default=None, description="Subsections if applicable"
    )

class ExperimentalSetup(BaseModel):
    proposed_datasets: List[str] = Field(description="Proposed datasets for utilization")
    evaluation_metrics: List[str] = Field(description="Metrics for evaluating effectiveness")
    hardware_software_requirements: str = Field(
        description="Hardware and software library requirements"
    )

class PaperOutline(BaseModel):
    title: str = Field(description="Proposed paper title")
    model_acronym: str = Field(
        description="Concise and impactful acronym representing the proposed model/method (e.g., iForestASD, PPO, NovaGNN)"
    )
    abstract: str = Field(description="Paper abstract")
    novelty: NoveltyAnalysis
    sections: List[PaperSection]
    experiment: ExperimentalSetup

# ==========================================
# 2. DATA PROCESSING, FILE READING & FILTERING
# ==========================================
def extract_text_from_files(uploaded_files) -> str:
    combined_text = ""
    for file in uploaded_files:
        try:
            if file.name.endswith(".pdf"):
                combined_text += f"\n[PDF DOCUMENT - {file.name}]:\n"
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        combined_text += text + "\n"
            elif file.name.endswith(".docx"):
                combined_text += f"\n[WORD DOCUMENT - {file.name}]:\n"
                doc = docx.Document(file)
                for para in doc.paragraphs:
                    combined_text += para.text + "\n"
            elif file.name.endswith(".txt"):
                combined_text += f"\n[TEXT DOCUMENT - {file.name}]:\n"
                combined_text += file.getvalue().decode("utf-8") + "\n"
            elif file.name.endswith((".py", ".js")):
                ext = file.name.split(".")[-1]
                lang = "python" if ext == "py" else "javascript"
                combined_text += (
                    f"\n[EXPERIMENTAL SOURCE CODE {lang.upper()} - {file.name}]:\n"
                )
                code_content = file.getvalue().decode("utf-8")
                combined_text += f"```{lang}\n{code_content}\n```\n"
        except Exception as e:
            st.warning(f"Failed to read file {file.name}: {e}")
    return combined_text

def extract_references_from_files(ref_files) -> str:
    ref_text = ""
    for file in ref_files:
        try:
            ref_text += f"\n[REFERENCE SOURCE - {file.name}]:\n"
            ref_text += file.getvalue().decode("utf-8") + "\n"
        except Exception as e:
            st.warning(f"Failed to read reference file {file.name}: {e}")
    return ref_text

@st.cache_data(show_spinner=False)
def fetch_academic_papers(query: str) -> pd.DataFrame:
    papers_data = []
    sch = SemanticScholar()
    try:
        s2_results = sch.search_paper(
            query,
            limit=10,
            fields=[
                "title",
                "year",
                "abstract",
                "authors",
                "citationCount",
                "influentialCitationCount",
                "externalIds",
            ],
        )
        for p in s2_results:
            if p.abstract and p.year:
                if (datetime.datetime.now().year - p.year <= 5) or (
                    p.influentialCitationCount and p.influentialCitationCount > 50
                ):
                    authors = (
                        ", ".join([a.name for a in p.authors])
                        if p.authors
                        else "Unknown"
                    )
                    doi = (
                        p.externalIds.get("DOI", f"s2_{p.paperId}")
                        if p.externalIds
                        else f"s2_{p.paperId}"
                    )
                    papers_data.append(
                        {
                            "id": doi,
                            "title": p.title,
                            "year": p.year,
                            "abstract": p.abstract,
                            "authors": authors,
                            "source": "Semantic Scholar",
                        }
                    )
    except Exception as e:
        st.info(f"Network blocked Semantic Scholar request: {e}")

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query, max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate
        )
        for r in client.results(search):
            papers_data.append(
                {
                    "id": r.get_short_id(),
                    "title": r.title,
                    "year": r.published.year,
                    "abstract": r.summary,
                    "authors": ", ".join([a.name for a in r.authors]),
                    "source": "arXiv",
                }
            )
    except Exception as e:
        st.info(f"Network blocked arXiv request: {e}")

    return pd.DataFrame(papers_data)

def filter_papers_by_cosine_similarity(
    prompt: str, df: pd.DataFrame, threshold: float = 0.05
) -> pd.DataFrame:
    if df.empty:
        return df
    docs = [prompt] + df["abstract"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    df["similarity"] = cosine_sim
    filtered_df = df[df["similarity"] >= threshold].sort_values(
        by="similarity", ascending=False
    )
    return filtered_df

# ==========================================
# 3. LATEX AND BIBTEX GENERATION
# ==========================================
def generate_bibtex(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    bibtex_str = ""
    for _, row in df.iterrows():
        first_author = (
            row["authors"].split(",")[0].split()[-1].replace("-", "")
            if row["authors"]
            else "Unknown"
        )
        title_word = re.sub(r"[^a-zA-Z0-9]", "", row["title"].split()[0])
        cit_key = f"{first_author}{row['year']}{title_word}"
        bibtex_str += f"@article{{{cit_key},\n  title={{{row['title']}}},\n  author={{{row['authors']}}},\n  year={{{row['year']}}},\n  abstract={{{row['abstract']}}}\n}}\n\n"
    return bibtex_str

def render_latex(outline: PaperOutline, journal_name: str) -> str:
    env = jinja2.Environment(
        block_start_string="\\BLOCK{",
        block_end_string="}",
        variable_start_string="\\VAR{",
        variable_end_string="}",
        comment_start_string="\\#{",
        comment_end_string="}",
        trim_blocks=True,
        autoescape=False,
    )

    # [FIX]: Dynamically resolve LaTeX document class and bibliography style based on target journal.
    # Prevents hardcoding issue where all papers were uniformly generated as IEEE format.
    doc_class = "\\documentclass[conference]{IEEEtran}"
    bib_style = "\\bibliographystyle{IEEEtran}"
    journal_lower = journal_name.lower()
    
    if "acm" in journal_lower:
        doc_class = "\\documentclass[sigconf]{acmart}"
        bib_style = "\\bibliographystyle{ACM-Reference-Format}"
    elif "elsevier" in journal_lower or "elsarticle" in journal_lower:
        doc_class = "\\documentclass[preprint,12pt]{elsarticle}"
        bib_style = "\\bibliographystyle{elsarticle-num}"
    elif "springer" in journal_lower:
        doc_class = "\\documentclass[sn-mathphys]{sn-jnl}"
        bib_style = "\\bibliographystyle{sn-mathphys}"
    elif "mdpi" in journal_lower:
        doc_class = "\\documentclass[journal,article,submit,moreauthors,pdftex]{Definitions/mdpi}"
        bib_style = "\\bibliographystyle{mdpi}"

    latex_template = """
        \\VAR{ document_class }
        \\usepackage{cite}
        \\usepackage{amsmath,amssymb,amsfonts}
        \\usepackage{graphicx}

        \\begin{document}

        \\title{\\VAR{ title }}

        \\author{\\IEEEauthorblockN{Author 1}
        \\IEEEauthorblockA{\\textit{Institute/University}\\\\
        Email: author@example.com}
        }

        \\maketitle

        \\begin{abstract}
        \\VAR{ abstract }
        \\end{abstract}

        \\BLOCK{ for sec in sections }
        \\section{\\VAR{ sec.title }}
        \\VAR{ sec.content_summary }

        \\BLOCK{ if sec.subsections }
        \\BLOCK{ for subsec in sec.subsections }
        \\subsection{\\VAR{ subsec.title }}
        \\VAR{ subsec.content_summary }
        \\BLOCK{ endfor }
        \\BLOCK{ endif }
        \\BLOCK{ endfor }

        \\section{Experimental Setup}
        \\textbf{Datasets:} \\VAR{ ', '.join(experiment.proposed_datasets) } \\\\
        \\textbf{Metrics:} \\VAR{ ', '.join(experiment.evaluation_metrics) } \\\\
        \\textbf{Hardware/Software:} \\VAR{ experiment.hardware_software_requirements }

        \\VAR{ bibliography_style }
        \\bibliography{references}

        \\end{document}
    """
    template = env.from_string(latex_template)

    return template.render(
        title=outline.title,
        abstract=outline.abstract,
        sections=outline.sections,
        experiment=outline.experiment,
        document_class=doc_class,
        bibliography_style=bib_style
    )
# ==========================================
# 4. HISTORY STORAGE SYSTEM (FILE I/O & VERSIONING)
# ==========================================
def save_and_create_zip_history(
    outline: PaperOutline, journal: str, tex_content: str, final_bib_content: str
):
    """Saves the physical ZIP file (containing main.tex and references.bib), records to CSV, and manages versioning."""
    # 1. Create directory using yyyy-mm-dd format
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join("histories", today_str)
    os.makedirs(folder_path, exist_ok=True)

    # 2. Process safe filenames
    safe_acronym = re.sub(r"[^a-zA-Z0-9_-]", "", outline.model_acronym)
    if not safe_acronym:
        safe_acronym = "Draft_Model"

    # Simplify and clean Journal name (e.g., "Springer Nature" -> "Springer-Nature")
    safe_journal = re.sub(r"[^a-zA-Z0-9]", "-", journal).strip("-")
    safe_journal = re.sub(r"-+", "-", safe_journal)

    # 3. Versioning Logic (Auto-increment version v1, v2, v3...)
    version = 1
    while True:
        zip_filename = f"{safe_journal}_{safe_acronym}_v{version}.zip"
        zip_filepath = os.path.join(folder_path, zip_filename)
        if not os.path.exists(zip_filepath):
            break
        version += 1

    # 4. Create physical ZIP file in the directory
    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("main.tex", tex_content)
        zip_file.writestr("references.bib", final_bib_content)

    # 5. Save metadata to histories/history.csv
    csv_path = os.path.join("histories", "history.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["Date", "Time", "Model_Acronym", "Journal", "Title", "Zip_Path"]
            )
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        writer.writerow(
            [today_str, time_str, safe_acronym, journal, outline.title, zip_filepath]
        )

    # 6. Read ZIP file back from disk as bytes for Streamlit Download button
    with open(zip_filepath, "rb") as f:
        zip_bytes = f.read()

    return zip_filename, zip_bytes

def display_history_sidebar():
    """Displays History menu utilizing Material Design styling."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕒 Generation History")

    csv_path = os.path.join("histories", "history.csv")
    if not os.path.exists(csv_path):
        st.sidebar.info("No saved versions available.")
        return

    try:
        df_hist = pd.read_csv(csv_path)
        if df_hist.empty:
            st.sidebar.info("No saved versions available.")
            return

        # Sort newest to top
        df_hist = df_hist.sort_values(by=["Date", "Time"], ascending=[False, False])

        # Utilize Streamlit native container to simulate Material Cards
        for _, row in df_hist.head(10).iterrows():
            with st.sidebar.container(border=True):
                st.markdown(f"**✨ {row['Model_Acronym']}**")
                st.caption(f"_{row['Title'][:60]}..._")
                st.markdown(
                    f"<span style='font-size: 0.8em; color: gray;'>📅 {row['Date']} {row['Time']} | 📖 {row['Journal']}</span>",
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.sidebar.error(f"Error reading history: {str(e)}")

# ==========================================
# 5. STREAMLIT INTERFACE & MAIN EXECUTION FLOW
# ==========================================
st.set_page_config(page_title="ScholarlyForge AI", layout="wide")
st.title("🔬 ScholarlyForge AI: Scientific Paper Outline Generator")

# Load environment variables
load_dotenv()

# API Key Configuration
env_api_key = os.getenv("GEMINI_API_KEY")
if env_api_key:
    api_key = env_api_key
    st.sidebar.success("✅ Connected to Gemini API Key")
else:
    api_key = st.sidebar.text_input("🔑 Enter Google Gemini API Key:", type="password")

# Target Journal Configuration
journal_options = [
    "IEEEtran (IEEE)",
    "acmart (ACM)",
    "elsarticle (Elsevier)",
    "Springer Nature",
    "MDPI",
    "PLOS ONE",
    "Science",
    "Other (Custom input)...",
]
selected_journal = st.sidebar.selectbox(
    "📖 Select journal/conference format:", journal_options
)

if selected_journal == "Other (Custom input)...":
    journal = st.sidebar.text_input("✍️ Enter target journal or conference name:")
else:
    journal = selected_journal

# Display History Menu
display_history_sidebar()

# Main Area 1: Idea & Experiment
st.subheader("1. Provide Research Idea & Experimental Data (Required)")
prompt_text = st.text_area(
    "Enter your research idea (Optional if files are uploaded):",
    height=100,
    placeholder="Example: Proposing a GNN model to optimize resource allocation. PyTorch code developed...",
)
uploaded_files = st.file_uploader(
    "Attach Experimental files (PDF, DOCX, TXT, PY, JS):",
    type=["pdf", "docx", "txt", "py", "js"],
    accept_multiple_files=True,
)

# Main Area 2: References (Optional)
st.subheader("2. Provide Reference Materials (Optional)")
st.info(
    "💡 Upload a .bib or .txt file containing references if you want the AI to adhere to your specific bibliography."
)
ref_files = st.file_uploader(
    "Attach reference bibliography (Supported: .bib, .txt)",
    type=["bib", "txt"],
    accept_multiple_files=True,
)

if st.button("🚀 Generate Paper & Download LaTeX", type="primary"):
    if not api_key:
        st.error("Please configure the Gemini API Key first.")
        st.stop()

    final_prompt = prompt_text
    file_text = ""
    if uploaded_files:
        if len(uploaded_files) > 3:
            st.warning("⚠️ Processing only the first 3 experimental files.")
            uploaded_files = uploaded_files[:3]

        with st.spinner("📄 Reading experimental data..."):
            file_text = extract_text_from_files(uploaded_files)
            final_prompt = f"{prompt_text}\n\n{file_text}"

    if not final_prompt.strip():
        st.warning(
            "Please enter an idea or upload at least 1 document/code file in section 1."
        )
        st.stop()

    if not journal:
        st.warning("Please specify a target journal.")
        st.stop()

    user_refs_text = ""
    user_bib_raw = ""
    if ref_files:
        with st.spinner("📚 Reading attached reference materials..."):
            user_refs_text = extract_references_from_files(ref_files)
            user_bib_raw = "\n".join(
                [
                    f.getvalue().decode("utf-8")
                    for f in ref_files
                    if f.name.endswith(".bib")
                ]
            )

    try:
        client = genai.Client(api_key=api_key)

        context_papers = ""
        df_filtered = pd.DataFrame()

        with st.status(
            "🔍 Retrieving network literature (Semantic Scholar/arXiv)...", expanded=True
        ) as status:
            search_query = (
                prompt_text[:150] if prompt_text else "Machine Learning AI System"
            )
            df_papers = fetch_academic_papers(search_query)

            if df_papers.empty:
                st.info("Network data skipped. Utilizing attached documents (if any).")
            else:
                df_filtered = filter_papers_by_cosine_similarity(
                    final_prompt, df_papers, threshold=0.05
                )
                if not df_filtered.empty:
                    st.write(
                        f"Retained {len(df_filtered)} highly relevant documents."
                    )
                    context_papers = "\n".join(
                        [
                            f"- {row['title']} ({row['year']}): {row['abstract']}"
                            for _, row in df_filtered.iterrows()
                        ]
                    )

            status.update(
                label="✅ Data collection phase completed",
                state="complete",
                expanded=False,
            )

        if user_refs_text:
            context_papers += (
                f"\n\n[USER PROVIDED REFERENCES]:\n{user_refs_text}"
            )
        if not context_papers.strip():
            context_papers = "[NOTE: No reference materials provided. Rely solely on SOURCE CODE and internal knowledge for design.]"

        with st.status(
            "🧠 Initializing Gemini reasoning chain...", expanded=True
        ) as status:
            st.write(f"Designing structure for: {journal}...")

            sys_instruct = "You are a senior researcher and peer-review expert."
            llm_prompt = f"""
            Target Journal: {journal}
            
            Idea & SOURCE CODE (CORE DATA): 
            "{final_prompt}"
            
            Current References:
            {context_papers}
            
            Requirements:
            1. Design a scientific paper outline.
            2. Automatically conceptualize and assign an impactful ACRONYM for the proposed model/method (e.g., iForestASD) into model_acronym.
            3. Thoroughly analyze the ATTACHED SOURCE CODE (if any) to write a realistic Methodology and Experimental Setup.
            4. Identify research gaps from the Reference Materials section.
            """

            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=llm_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruct,
                    response_mime_type="application/json",
                    response_schema=PaperOutline,
                    temperature=0.2,
                ),
            )

            paper_outline: PaperOutline = response.parsed
            status.update(
                label="✅ AI analysis completed", state="complete", expanded=False
            )

        st.subheader(
            f"📄 Proposed Structure: {paper_outline.model_acronym} (Targeted for {journal})"
        )
        st.markdown(f"**Title:** {paper_outline.title}")
        st.info(f"**Abstract:**\n{paper_outline.abstract}")

        col1, col2 = st.columns(2)
        with col1:
            st.success("**Novelty & Gaps:**")
            for gap in paper_outline.novelty.research_gaps:
                st.markdown(f"- ⚠️ {gap}")
            st.markdown(f"**Contribution:** {paper_outline.novelty.proposed_contribution}")

        with col2:
            st.warning("**Experimental Setup:**")
            st.markdown(
                f"- **Data:** {', '.join(paper_outline.experiment.proposed_datasets)}"
            )
            st.markdown(
                f"- **Metrics:** {', '.join(paper_outline.experiment.evaluation_metrics)}"
            )
            st.markdown(
                f"- **System:** {paper_outline.experiment.hardware_software_requirements}"
            )

        with st.spinner("📦 Packaging and storing history..."):
            tex_code = render_latex(paper_outline, journal)
            bib_code = generate_bibtex(df_filtered)
            final_bib_code = bib_code + "\n\n" + user_bib_raw

            zip_filename, zip_bytes = save_and_create_zip_history(
                paper_outline, journal, tex_code, final_bib_code
            )

            st.success("🎉 Complete! The latest version has been safely stored.")

            st.download_button(
                label="⬇️ DOWNLOAD LATEX PROJECT (ZIP)",
                data=zip_bytes,
                file_name=zip_filename,
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")