import os
import re
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from parser import parse_resume   



def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


SKILL_KEYWORDS = {

    "machine_learning": [
        "machine learning", "ml", "machine learning models",
        "supervised learning", "unsupervised learning",
        "classification", "regression", "clustering",
        "overfitting", "underfitting", "regularization",
        "model evaluation", "cross validation"
    ],

 
    "deep_learning": [
        "deep learning", "neural network", "neural networks",
        "cnn", "convolutional neural network",
        "rnn", "recurrent neural network",
        "lstm", "gru", "transformer",
        "attention mechanism",
        "tensorflow", "keras", "pytorch"
    ],

  
    "data_preprocessing": [
        "data preprocessing", "data cleaning",
        "missing data", "imputation",
        "feature engineering", "feature selection",
        "normalization", "standardization",
        "scaling", "encoding"
    ],


    "nlp": [
        "natural language processing", "nlp",
        "text preprocessing", "tokenization",
        "stemming", "lemmatization",
        "tf idf", "word embeddings",
        "bert", "transformers", "text classification"
    ],

   
    "computer_vision": [
        "computer vision", "image processing",
        "opencv", "object detection",
        "image classification", "image segmentation"
    ],

   
    "mlops": [
        "mlops", "deployment", "model deployment",
        "docker", "containerization",
        "api", "rest api", "fastapi", "flask",
        "model monitoring", "model serving",
        "ci cd", "pipeline"
    ],

   
    "programming": [
        "python", "java", "c++", "sql",
        "pandas", "numpy", "scikit learn",
        "matplotlib", "seaborn"
    ],

    
    "cloud": [
        "aws", "azure", "gcp",
        "cloud deployment", "cloud services",
        "s3", "ec2", "lambda"
    ],

    
    "software_engineering": [
        "data structures", "algorithms",
        "system design", "design patterns",
        "object oriented programming",
        "oop", "version control", "git"
    ]
}



def extract_skills_from_text(text: str) -> List[str]:
    skills = []
    for keywords in SKILL_KEYWORDS.values():
        for kw in keywords:
            if kw in text:
                skills.append(kw)
    return list(set(skills))



def compute_skill_overlap(resume_skills: List[str], jd_skills: List[str]) -> float:
    if not jd_skills:
        return 0.0
    overlap = set(resume_skills) & set(jd_skills)
    return len(overlap) / len(set(jd_skills))



model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_semantic_similarity(resume_text: str, jd_text: str) -> float:
    resume_embedding = model.encode(resume_text)
    jd_embedding = model.encode(jd_text)
    similarity = cosine_similarity(
        [resume_embedding], [jd_embedding]
    )[0][0]
    return similarity



def compute_final_score(
    resume_text: str,
    jd_text: str,
    resume_skills: List[str],
    jd_skills: List[str]
) -> Dict:

    skill_score = compute_skill_overlap(resume_skills, jd_skills)
    semantic_score = compute_semantic_similarity(resume_text, jd_text)

    final_score = (
        0.5 * skill_score +
        0.5 * semantic_score
    )

    return {
        "skill_overlap_score": round(skill_score * 100, 2),
        "semantic_similarity_score": round(semantic_score * 100, 2),
        "final_match_score": round(final_score * 100, 2)
    }



if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    resume_path = os.path.join(BASE_DIR, "raghav_cv.pdf")
    jd_path = os.path.join(BASE_DIR, "sample_jd.txt")

    parsed_resume = parse_resume(resume_path)

    resume_skills = []
    for values in parsed_resume["skills_extracted"].values():
        resume_skills.extend(values)

    resume_text = " ".join(resume_skills)

    jd_raw = open(jd_path, "r", encoding="utf-8").read()
    jd_clean = clean_text(jd_raw)
    jd_skills = extract_skills_from_text(jd_clean)

    scores = compute_final_score(
        resume_text,
        jd_clean,
        resume_skills,
        jd_skills
    )

    print(f"Skill Overlap Score: {scores['skill_overlap_score']}%")
    print(f"Semantic Similarity: {scores['semantic_similarity_score']}%")
    print(f"FINAL MATCH SCORE: {scores['final_match_score']}%")
