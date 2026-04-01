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
        "Experienced Python developer with machine learning and data analysis skills",
        "Frontend developer skilled in React, HTML, CSS, JavaScript",
        "Java backend engineer with Spring Boot and microservices experience",
        "Data analyst with SQL, Excel, and visualization experience"
    ]

    for text in samples:
        test_single(text)
        print("\n" + "="*50)


if __name__ == "__main__":
    test_multiple()