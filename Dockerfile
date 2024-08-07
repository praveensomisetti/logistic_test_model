# Use the official AWS Lambda Python 3.8 base image
FROM public.ecr.aws/lambda/python:3.8

# Copy model and handler code
COPY logistic_regression_model.pkl ${LAMBDA_TASK_ROOT}
COPY logistic_regression_handler.py ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN pip install pandas scikit-learn joblib

# Set the CMD to your handler
CMD ["logistic_regression_handler.lambda_handler"]
