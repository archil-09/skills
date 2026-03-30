import nltk
nltk.download('punkt')
import streamlit as st
import pickle

import PyPDF2
import re
import docx

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

clf = pickle.load(open('clf.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
lc = pickle.load(open('lc.pkl', 'rb'))

def clean_resume(text):
    clean_text = re.sub(r'https\S+\s',' ',text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+',' ',clean_text)
    clean_text = re.sub(r'@\S+',' ',clean_text)
    clean_text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    tokens = word_tokenize(clean_text.lower())

# load stopwords
    stop_words = set(stopwords.words('english'))

# remove stopwords
    filtered = [word for word in tokens if word not in stop_words]

    return ' '.join(filtered)
def predict(resume):
    cleaned_resume = clean_resume(resume)
    input_features = vectorizer.transform([cleaned_resume])
    input_features = input_features.toarray()
    prediction_id = clf.predict(input_features)[0]
    predicted_category_name = lc.inverse_transform([prediction_id])
    return predicted_category_name[0]
def skill_gap(resume):
    user_skills = extract_skills(resume)
    category = predict(resume)

    required = set(required_skills.get(category, []))
    user = set(user_skills)

    gap = required - user
    return gap
required_skills = {
    "Data Science": [
        "python", "r", "sql", "machine learning", "deep learning",
        "statistics", "numpy", "pandas", "data visualization", "tableau",
        "tensorflow", "nlp"
    ],

    "HR": [
        "recruitment", "communication", "employee relations",
        "payroll", "hr policies", "talent management"
    ],

    "Advocate": [
        "legal research", "litigation", "drafting", "contract law",
        "legal writing", "negotiation"
    ],

    "Arts": [
        "creativity", "design", "illustration", "painting",
        "visual communication"
    ],

    "Web Designing": [
        "html", "css", "javascript", "ui/ux", "figma",
        "responsive design"
    ],

    "Mechanical Engineer": [
        "autocad", "solidworks", "thermodynamics",
        "manufacturing", "cad", "ansys"
    ],

    "Sales": [
        "communication", "negotiation", "lead generation",
        "crm", "customer service"
    ],

    "Health and fitness": [
        "nutrition", "fitness training", "anatomy",
        "exercise science", "diet planning"
    ],

    "Civil Engineer": [
        "autocad", "construction", "structural analysis",
        "site management", "surveying"
    ],

    "Java Developer": [
        "java", "spring", "hibernate", "oop",
        "mysql", "rest api"
    ],

    "Business Analyst": [
        "excel", "sql", "data analysis",
        "requirement gathering", "power bi"
    ],

    "SAP Developer": [
        "sap", "abap", "erp", "sap hana"
    ],

    "Automation Testing": [
        "selenium", "testng", "automation testing",
        "java", "cypress"
    ],

    "Electrical Engineering": [
        "circuit design", "power systems",
        "matlab", "electronics"
    ],

    "Operations Manager": [
        "operations", "logistics", "project management",
        "supply chain"
    ],

    "Python Developer": [
        "python", "django", "flask",
        "api", "sql"
    ],

    "DevOps Engineer": [
        "docker", "kubernetes", "aws",
        "ci/cd", "linux"
    ],

    "Network Security Engineer": [
        "cybersecurity", "networking", "firewalls",
        "ethical hacking"
    ],

    "PMO": [
        "project management", "planning",
        "risk management", "reporting"
    ],

    "Database": [
        "sql", "mongodb", "database design",
        "postgresql"
    ],

    "Hadoop": [
        "hadoop", "hive", "spark",
        "big data"
    ],

    "ETL Developer": [
        "etl", "data warehousing",
        "sql", "informatica"
    ],

    "DotNet Developer": [
        ".net", "c#", "asp.net",
        "sql server"
    ],

    "Blockchain": [
        "blockchain", "ethereum",
        "solidity", "smart contracts"
    ],

    "Testing": [
        "manual testing", "test cases",
        "bug tracking", "qa"
    ]
}
def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in SKILLS:
        if skill in text:
            found_skills.append(skill)

    return list(set(found_skills))
SKILLS = [
    "python", "pandas", "numpy", "scipy", "scikit learn",
    "sql", "java", "javascript", "jquery",
    "machine learning", "regression", "svm", "naive bayes",
    "knn", "random forest", "decision trees",
    "nlp", "natural language processing",
    "deep learning", "word embedding",
    "tableau", "matplotlib", "ggplot",
    "mysql", "sqlserver", "cassandra", "hbase",
    "elasticsearch", "kafka", "docker", "flask",
    "git", "html", "css", "angular"
]
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(file):
    # Try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("Resume skill gaps identification")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.")
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        # Extract text from the uploaded file
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")

            # Display extracted text (optional)
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Make prediction
            st.subheader("Predicted Category")
            category = predict(resume_text)
            gap = skill_gap(resume_text)

            st.write(f"The predicted category of the uploaded resume is: **{category}**")
            st.write(f"The skill gap in the uploaded resume is: {gap}")
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()