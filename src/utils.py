def format_symptoms_for_report(input_data: dict) -> str:
    symptoms = []
    for symptom, value in input_data.items():
        if symptom == "age":
            symptoms.append(f"Age: {value}")
        else:
            answer = "Yes" if value == 1 else "No"
            symptom_name = symptom.replace("_", " ").title()
            symptoms.append(f"{symptom_name}: {answer}")
    return "\n".join(symptoms)