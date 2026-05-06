import joblib

model = joblib.load("models/resume_pipeline.pkl")

def test_single(text):
    prediction = model.predict([text])[0]

    print("\nINPUT TEXT:")
    print(text[:200])

    print("\nPREDICTED ROLE:")
    print(prediction)


def test_multiple():
    samples = [
        "John Snow johnsnow@example.com | (555) 987-6543 | linkedin.com/in/johnsnow | github.com/johnnow Summary Results-driven Data Scientist and Software Engineer who architected and developed highly scalable machine learning solutions. Designed and deployed advanced predictive models that accelerated business intelligence capabilities and increased overall revenue. Consistently recognized for the ability to translate technical requirements into actionable business outcomes. Committed to continuous learning and staying updated with the latest advancements in artificial intelligence and cloud computing. Proven track record of delivering end-to-end data systems from initial data ingestion to final API deployment. Skills Languages: Python, Java, Javascript, SQL Frameworks: FastAPI, Django, React, Flask, NextJS Databases: PostgreSQL, MongoDB, MySQL, Redis, ElasticSearch Cloud & DevOps: AWS, Docker, Kubernetes, CI/CD, Terraform Machine Learning: Machine Learning, Deep Learning, TensorFlow, Pandas, NumPy, Scikit-Learn, NLP, PyTorch Experience Senior Data Scientist | InnovateTech Solutions | Jan 2022 - Present •	Engineered and deployed a comprehensive deep learning recommendation engine that increased customer engagement by 45%. •	Architected robust automated data pipelines processing data for over 5000 users daily using Python and SQL. •	Managed an agile group of 6 engineers to successfully deliver 4 projects ahead of schedule. •	Optimized complex PostgreSQL database queries, which directly reduced reporting latency by 60%. •	Spearheaded the migration of legacy monolith applications to scalable AWS microservices, saving $150,000 in annual infrastructure costs. •	Analyzed user interaction logs across 12 systems to diagnose and resolve critical performance bottlenecks. •	Mentored 5 members of the junior team through weekly code reviews and technical training sessions. •	Orchestrated the integration of CI/CD workflows utilizing Docker and Kubernetes to ensure seamless deployments. •	Streamlined internal documentation practices which accelerated the onboarding process for new hires. Data Engineer | DataWorks Inc | Jun 2020 - Dec 2021 •	Developed scalable RESTful APIs utilizing FastAPI and Django to serve predictions to over 50 clients. •	Implemented natural language processing models to classify customer support tickets with high accuracy. •	Automated daily reporting workflows using scheduled Python scripts, eliminating hours of manual data entry. •	Evaluated emerging machine learning frameworks to determine the optimal tech stack for future initiatives. •	Refactored legacy Java codebases to improve maintainability and strictly adhere to modern software design patterns. Projects Intelligent Resume Classifier System •	Built an end-to-end text classification application utilizing advanced machine learning techniques and Scikit-Learn. •	Designed an interactive frontend user interface using React to display analytics and prediction confidence scores. •	Transformed large volumes of unstructured document data into clean, structured datasets for model training. •	Launched the final application on AWS cloud infrastructure, ensuring high availability and fault tolerance. Customer Segmentation Dashboard •	Identified distinct customer personas using clustering algorithms and detailed demographic data analysis. •	Defined key performance indicators to track the success of targeted marketing campaigns across regional markets. •	Achieved a significant reduction in prediction error rates by 3x through tuning hyperparameters extensively. Education Bachelor of Technology in Computer Science Vardhaman College of Engineering | Aug 2016 - May 2020 Coursework: Data Structures, Algorithms, Database Management Systems, Computer Networks, Artificial Intelligence. Led the university coding club and organized competitive programming events for over 200 participants. Graduated with honors, maintaining a top percentile ranking throughout the academic tenure. Certifications AWS Certified Solutions Architect Associate Deep Learning Specialization Certification",
        "Frontend developer skilled in React, HTML, CSS, JavaScript",
        "Java backend engineer with Spring Boot and microservices experience",
        "Data analyst with SQL, Excel, and visualization experience"
    ]

    for text in samples:
        test_single(text)
        print("\n" + "="*50)


if __name__ == "__main__":
    test_multiple()