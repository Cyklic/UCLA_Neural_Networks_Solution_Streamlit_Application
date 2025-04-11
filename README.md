# UCLA Admission Chances Predictior Application
This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://leonard-umoru-ucla-neural-networks-solution.streamlit.app/)

This application predicts whether a student is eligible for admission at UCLA loan based on inputs derived from the current admission dataset, built to help students in shortlisting universities based on their profiles. The model aims to help students assess admission eligibility by leveraging machine learning predictions.

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter details such as GRE Score, TOEFL Score, CGPA, University Rating, and other relevant factors.
- Real-time prediction of admission eligibility based on the trained model.
- Accessible via Streamlit Community Cloud.

## Dataset
The world is developing rapidly, and continuously looking for the best knowledge and experience among people. This motivates people all around the world to stand out in their jobs and look for higher degrees that can help them in improving their skills and knowledge. As a result, the number of students applying for Master's programs has increased substantially.

The current admission dataset was created for the prediction of admissions into the University of California, Los Angeles (UCLA). It was built to help students in shortlisting universities based on their profiles. The predicted output gives them a fair idea about their chances of getting accepted.

Specifics:

Machine Learning task: Classification model
Target variable: Admit_Chance
Input variables: Refer to data dictionary below
Success Criteria: Accuracy of 90% and above
Data Dictionary:
The dataset contains several parameters which are considered important during the application for Masters Programs. The parameters included are :

GRE_Score: (out of 340)
TOEFL_Score: (out of 120)
University_Rating: It indicates the Bachelor University ranking (out of 5)
SOP: Statement of Purpose Strength (out of 5)
LOR: Letter of Recommendation Strength (out of 5)
CGPA: Student's Undergraduate GPA(out of 10)
Research: Whether the student has Research Experience (either 0 or 1)
Admit_Chance: (ranging from 0 to 1)

## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).

## Model
The predictive model applies preprocessing steps like encoding categorical variables and scaling numerical features. The classification model used is Multi layer perceptron.

## Future Enhancements
* Adding support for multiple datasets.
* Incorporating explainability tools like SHAP to provide insights into predictions.
* Adding visualizations to better represent user input and model predictions.

## Installation (for local deployment)
If you want to run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit_eligibility_application.git
   cd credit_eligibility_application

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\\Scripts\\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit application:
   ```bash
   streamlit run app.py

#### Thank you for using the UCLA Admission Chances Predictor Application! Feel free to share your feedback.
