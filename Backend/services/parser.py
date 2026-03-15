import pdfplumber
import docx
import re
from typing import Dict, List
import os


SKILL_KEYWORDS = {
    "machine_learning": [
        "machine learning", "ml", "model training", "supervised learning",
        "unsupervised learning", "overfitting", "regularization"
    ],
    "deep_learning": [
        "deep learning", "neural network", "cnn", "rnn", "transformer"
    ],
    "data_preprocessing": [
        "data cleaning", "missing data", "imputation", "feature engineering"
    ],
    "mlops": [
        "deployment", "docker", "mlops", "model monitoring"
    ],
    "programming": [
        "python", "java", "c++", "sql"
    ],
    "software_engineering": [
        "data structures", "algorithms", "system design", "api"
    ]
}


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text



def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    text = " ".join([para.text for para in doc.paragraphs])
    return text





def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text 
   

def extract_skills(cleaned_text: str) -> Dict[str, List[str]]:
    extracted_skills = {}

    for skill, keywords in SKILL_KEYWORDS.items():
        matched = []
        for keyword in keywords:
            if keyword in cleaned_text:
                matched.append(keyword)
        if matched:
            extracted_skills[skill] = list(set(matched))

    return extracted_skills


def parse_resume(file_path: str) -> Dict:
    if file_path.endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        raw_text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format")

    cleaned_text = clean_text(raw_text)
    skills = extract_skills(cleaned_text)

    return {
        "raw_text_length": len(raw_text),
        "cleaned_text_length": len(cleaned_text),
        "skills_extracted": skills
    }


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    resume_path = os.path.join(BASE_DIR, "raghav_cv.pdf")
    # change path
    parsed_data = parse_resume(resume_path)
    print(parsed_data)
