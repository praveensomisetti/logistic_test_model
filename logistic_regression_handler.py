import json
import joblib
import pandas as pd

# Load model
model = joblib.load('logistic_regression_model.pkl')

def lambda_handler(event, context):
    try:
        # Check if event['body'] exists
        if 'body' in event:
            data = json.loads(event['body'])
        else:
            data = event  # Handle direct invocation with test data

        # Extract only the required features
        features = ['Pclass', 'Sex', 'Age', 'Fare']
        input_data = {feature: data[feature] for feature in features if feature in data}
        
        if len(input_data) != len(features):
            raise ValueError(f"Input data is missing one or more required features: {features}")

        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict(df)[0]

        # Return response
        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': int(prediction)}),
            'headers': {
                'Content-Type': 'application/json'
            }
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
