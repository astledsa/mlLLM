import os
import pickle
import numpy as np
import pandas as pd
from io import StringIO
from openai import OpenAI
from lime import lime_tabular
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from models import (LinearRegressionModel, LogisticRegressionModel, RandomForestRegressorModel, 
                   RandomForestClassifierModel, GradientBoostingRegressorModel, GradientBoostingClassifierModel, 
                   SVRModel, SVCModel, KNNRegressorModel, KNNClassifierModel)



client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],)

def feature_selector(query, sample_data, data_dict=None):
    if data_dict==None:
        data = "The data dictionary is not provided. Refer to the sample data."
    else:
        with open(f'{data_dict}') as f:
            data = f.read()
    
    try:
        df = pd.read_csv(StringIO(sample_data))
    except:
        raise Exception("Error reading this format:", sample_data)
    
    first = df.head(10)
    
    prompt = f"""
            You are an expert Data Scientist. Your task is to go through the following information about the document and perform 
            the following tasks.

            Query: "{query}"
            Data Dictionary: "{data}"
            Data preview: 
            {first.to_string(index=False)}
            
            Using the above information: 
            Select three preferred models based on the problem statement and use the exact words mentioned here: LinearRegression, 
            LogisticRegression, RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, 
            SupportVectorMachineRegressor, SupportVectorMachineClassifier, KNeighborsRegressor, KNeighborsClassifier
            Select only the features that have a direct and measurable influence on the target variable or are essential to solving 
            the problem described in the query. Avoid selecting features that are primarily descriptive (e.g., identifiers, 
            categories with no predictive value) or time-related unless they provide clear insight into the problem. Focus on 
            numerical or transactional features with a high likelihood of affecting the outcome.

            **Return ONLY the result in this exact format:**
            Reccomended Model: <Model1>, <Model2>, <Model3>
            Dependent variable: <dependent_variable>
            Independent variables: <variable1>, <variable2>, <variable3>, ...

            Ensure the selection contains only those features that contribute to improving model performance and generalization, 
            while avoiding irrelevant or redundant attributes. Do not include any extra text or explanations. Your response must 
            strictly adhere to this format.
            """
    
    response = client.chat.completions.create(
        messages=[{
            "role": "system", 
            "content": prompt, 
            }],
        
        model="gpt-4o",
        temperature=0.15,
    )

    response_text = response.choices[0].message.content.strip()
    
    lines = response_text.split('\n')
    recommended_models = [model.strip() for model in lines[0].split(': ')[1].split(',')]
    target = lines[1].split(': ')[1].strip()
    independent = [var.strip() for var in lines[2].split(': ')[1].split(',')]
    
    return  independent, target, df, recommended_models

    

def feature_engineering(df, X_cols, y_col):
    if len(df) > 100000:
        df = df.head(100000).copy()
    else:
        df = df.copy()
    
    X = df[X_cols].copy()
    y = df[y_col].copy()
    
    if y.isnull().any():
        mask = y.notnull()
        X = X[mask]
        y = y[mask]
    
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    
    if X.isnull().any().any():
        if len(numeric_columns) > 0:
            numeric_imputer = SimpleImputer(strategy='mean')
            X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])
        
        if len(categorical_columns) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])
    
    for col in numeric_columns:
        if np.isinf(X[col]).any():
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = SimpleImputer(strategy='mean').fit_transform(X[[col]])
    
    for col in numeric_columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X[col] = np.clip(X[col], lower_bound, upper_bound)
    
    if len(numeric_columns) > 0:
        scaler = StandardScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    else:
        scaler = None
    
    label_encoders = {}
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def run_preferred_models(preferred_models, X_train, y_train, X_test, y_test):
    model_mapping = {
        'LinearRegression': LinearRegressionModel,
        'LogisticRegression': LogisticRegressionModel,
        'RandomForestRegressor': RandomForestRegressorModel,
        'RandomForestClassifier': RandomForestClassifierModel,
        'GradientBoostingRegressor': GradientBoostingRegressorModel,
        'GradientBoostingClassifier': GradientBoostingClassifierModel,
        'SupportVectorMachineRegressor': SVRModel,
        'SupportVectorMachineClassifier': SVCModel,
        'KNeighborsRegressor': KNNRegressorModel,
        'KNeighborsClassifier': KNNClassifierModel
    }
    
    results = {}
    explanations = {}

    for model_name in preferred_models:
        model_name = model_name.strip()
        model_class = model_mapping.get(model_name)
        if model_class is None:
            continue
        
        is_classifier = 'Classifier' in model_name or model_name == 'LogisticRegression'
        
        if not np.issubdtype(y_train.dtype, np.number):
            if is_classifier:
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train)
                y_test_encoded = le.transform(y_test)
            else:  
                try:
                    y_train_encoded = y_train.astype(float)
                    y_test_encoded = y_test.astype(float)
                except ValueError:
                    print(f"Cannot convert target variable to numeric for {model_name}. Skipping this model.")
                    continue
        else:
            y_train_encoded = y_train
            y_test_encoded = y_test

        model_instance = model_class(X_train, y_train_encoded, X_test, y_test_encoded)
        
        try:
            model_instance.train()
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}. Skipping this model.")
            continue
        
        test_results = model_instance.test()
        results[model_name] = test_results
        # pickle.dump(model_instance, open("model_instance.pkl", 'wb'))

        if is_classifier:
            mode = 'classification'
            predict_fn = model_instance.predict_proba
        else:
            mode = 'regression'
            predict_fn = model_instance.predict
       
        interpretor = lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns,
            mode=mode,
            training_labels=y_train_encoded
        )
        
        exp = interpretor.explain_instance(
            data_row=X_test.iloc[7], 
            predict_fn=predict_fn
        )

        explanations[model_name] = exp
    
    return results, explanations

def lime_output(explanations):
    # for model_name, exp in explanations.items():
        # print(f"Explanation for {model_name}:")
        # exp.show_in_notebook(show_table=True)

    explanation_narratives = ""

    for model_name, exp in explanations.items():
        explanation_narrative = f"\nExplanation for {model_name}:\n"
        explanation_list = exp.as_list()
        explanation_narrative += f"For the {model_name} model, the following features had the most significant impact on the prediction:\n"
        for feature, weight in explanation_list:
            impact_type = "positive" if weight > 0 else "negative"
            explanation_narrative += f"- Feature '{feature}' had a {impact_type} impact with a weight of {abs(weight):.2f}. "
        explanation_narratives += explanation_narrative + "\n"
    return explanation_narratives


class Pipleline:
    def __init__(self, query, sample_data, data_dict=None):
        self.query = query 
        self.sample_data = sample_data
        self.data_dict = data_dict
    
    def forward(self):
        independent, target, df, recommended_models = feature_selector(self.query, self.sample_data, data_dict=self.data_dict)
        X_train, X_test, y_train, y_test = feature_engineering(df, independent, target)
        test_results, explanations =  run_preferred_models(recommended_models, X_train, y_train, X_test, y_test)
        explanations_narratives = lime_output(explanations)
        prompt = f"""As a data scientist, you've analyzed a dataset to address the following problem:

{self.query}

Your analysis, using interpretable machine learning techniques, has revealed these key insights:

{explanations_narratives}

Please provide an explanation of your findings in a clear, insightful manner that a non-technical stakeholder could understand. Your response should:

1. Briefly recap the problem and why it's important to solve.

2. Describe the most significant patterns or trends you found in the data. Focus on 2-3 key features that stood out in your analysis.

3. For each key feature:
   - Explain what it represents in simple terms
   - Describe how it relates to the problem
   - Offer potential reasons for its importance, based on your domain knowledge

4. Discuss any surprising or counterintuitive findings, if any.

5. Suggest 2-3 concrete actions or decisions that could be made based on these insights.

6. Briefly mention any limitations of the analysis or areas for further investigation.

Guidelines:
- Use clear, jargon-free language
- Provide context and reasoning for your insights
- Balance detail with clarity - aim for a concise yet informative explanation
- Feel free to use analogies or real-world examples to illustrate points
- Avoid technical details about the model or analysis process unless directly relevant
- Your tone should be professional but conversational, as if explaining to a colleague

Remember, your goal is to convey the value and implications of your data analysis in a way that enables informed decision-making."""
        response = client.chat.completions.create(
            messages=[{
                "role": "system",  
                "content": prompt,  
            }],
            
            model="gpt-4o",
            temperature=0.3,
        )
    
        response_text = response.choices[0].message.content.strip()
        return response_text, test_results
    
