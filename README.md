# Vehicle Insurance Cross-Sell Prediction - End-to-End MLOps Project
 
An end-to-end MLOps project that predicts whether a customer will be interested in **vehicle insurance** (`Response` = Yes / No). The repository covers the full workflow:
 
- Data storage in MongoDB Atlas
- Modular training pipeline (ingestion, validation, transformation, training, evaluation, pushing)
- Model registry on AWS S3
- FastAPI prediction service
- Dockerization
- CI/CD with GitHub Actions, AWS ECR and EC2
> **Goal**: Build a production-style ML pipeline that classifies customers as likely / unlikely to buy vehicle insurance, using modern DevOps practices.
 
---
 
## Project Architecture
 
```
                        +-------------------+
                        | Template Creation |
                        +--------+----------+
                                 ↓
                        +--------v----------+
                        | Environment Setup |
                        +--------+----------+
                                 ↓
                        +--------v------------+
                        | MongoDB Integration |
                        +--------+------------+
                                 ↓
        +------------------------v-------------------------------+
        | Data Pipelines (Ingestion, Validation, Transformation) |
        +------------------------+-------------------------------+
                                 ↓
            +--------------------v------------------------+
            | Model Training + Evaluation + Pushing to S3 |
            +--------------------+------------------------+
                                 ↓
            +--------------------v------------------------+
            | FastAPI + Docker + ECR + EC2 Deployment     |
            +---------------------------------------------+
```
 
---
 
## Features
 
- Local package creation using `setup.py` and `pyproject.toml`
- MongoDB Atlas as the data source
- Modular `src/` architecture with config and artifact entities for every stage
- Data ingestion, validation and transformation (with SMOTEENN for class imbalance)
- Model training (RandomForest), evaluation against the production model, and pushing to S3
- AWS S3 used as a simple model registry
- Prediction web app served with FastAPI (HTML form + `/train` route)
- Docker image built and pushed to AWS ECR, deployed on EC2 through GitHub Actions
- App served on port `5000`
---
 
## Tech Stack
 
| Domain             | Tools / Services Used                                  |
| ------------------ | ------------------------------------------------------ |
| Programming        | Python 3.10                                            |
| Package Management | pip, conda                                             |
| Data Storage       | MongoDB Atlas                                          |
| Data Handling      | pandas, numpy, PyYAML                                  |
| Model Training     | scikit-learn (RandomForestClassifier), imbalanced-learn (SMOTEENN) |
| Cloud Services     | AWS S3, IAM, EC2, ECR (accessed via boto3)             |
| Web App            | FastAPI, Uvicorn, Jinja2 (HTML form)                   |
| Containerization   | Docker                                                 |
| CI/CD              | GitHub Actions, self-hosted runner on EC2              |
 
---
 
## Directory Structure
 
```
.
├── .github/workflows/
│   └── aws.yaml                # CI/CD workflow
├── config/
│   ├── schema.yaml             # Dataset schema used for validation/transformation
│   └── model.yaml
├── src/
│   ├── components/             # Ingestion, validation, transformation, trainer, evaluation, pusher
│   ├── configuration/          # MongoDB & AWS connections
│   ├── constants/              # Project-wide constants
│   ├── data_access/            # MongoDB -> DataFrame logic
│   ├── entity/                 # Config entities, artifact entities, estimators
│   ├── cloud_storage/          # AWS S3 integration
│   ├── pipline/                # Training and prediction pipelines
│   ├── logger/                 # Logging setup
│   ├── exception/              # Custom exception
│   └── utils/                  # Helper functions
├── notebook/                   # EDA and MongoDB demo notebooks
├── templates/ & static/        # HTML form and CSS for the web app
├── app.py                      # FastAPI application
├── demo.py                     # Quick local test script
├── template.py                 # Creates the project folder structure
├── Dockerfile                  # Docker setup
├── projectflow.txt             # Step-by-step build notes
├── requirements.txt
├── pyproject.toml & setup.py   # Local package management
└── README.md
```
 
---
 
## Setup Guide
 
### 1. Create project template & environment
 
```bash
python template.py
conda create -n vehicle python=3.10 -y
conda activate vehicle
pip install -r requirements.txt
```
 
### 2. Setup local packages
 
```bash
pip list  # Confirm that the local `src` package is installed
```
 
### 3. MongoDB Atlas integration
 
- Create a project and a cluster (M0 tier)
- Create a DB user and whitelist `0.0.0.0/0`
- Copy the Python connection string (replace `<password>`)
- Push the dataset (`notebook/data.csv`) to MongoDB using `notebook/mongoDB_demo.ipynb`
### 4. Environment variables
 
```bash
# Mac/Linux
export MONGODB_URL="mongodb+srv://<user>:<pass>@cluster.mongodb.net/..."
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
 
# Windows PowerShell
$env:MONGODB_URL = "mongodb+srv://<user>:<pass>@cluster.mongodb.net/..."
$env:AWS_ACCESS_KEY_ID = "..."
$env:AWS_SECRET_ACCESS_KEY = "..."
```
 
### 5. Run locally
 
```bash
python app.py        # app starts at http://localhost:5000
```
 
### 6. Run with Docker
 
```bash
docker build -t vehicleproj .
docker run -p 5000:5000 \
  -e AWS_ACCESS_KEY_ID="..." \
  -e AWS_SECRET_ACCESS_KEY="..." \
  -e MONGODB_URL="..." \
  vehicleproj
```
 
---
 
## ML Pipeline Components
 
The training pipeline (`src/pipline/training_pipeline.py`) runs these stages in order. Each stage takes a **config** entity as input and returns an **artifact** entity that is passed to the next stage.
 
```
MongoDB -> Ingestion -> Validation -> Transformation -> Trainer -> Evaluation -> Pusher (S3)
```
 
### Data Ingestion
 
- Fetches the `Proj1-Data` collection from MongoDB
- Converts it to a pandas DataFrame and saves it as a CSV (feature store)
- Splits it into train and test sets (75% / 25%)
### Data Validation
 
- Reads the expected schema from `config/schema.yaml`
- Checks that the number of columns matches the schema
- Checks that all required numerical and categorical columns are present
- Writes the result to a validation report; the pipeline stops if validation fails
### Data Transformation
 
- Maps `Gender` to binary (Female = 0, Male = 1)
- Creates dummy variables for `Vehicle_Age` and `Vehicle_Damage`
- Scales `Age` and `Vintage` with StandardScaler, and `Annual_Premium` with MinMaxScaler
- Handles class imbalance using SMOTEENN
- Saves the preprocessing object and the transformed arrays
### Model Training
 
- Trains a `RandomForestClassifier` (200 trees, `max_depth=10`, `criterion="entropy"`)
- Reports accuracy, F1, precision and recall
- Saves preprocessing + model together as a single `MyModel` object (`model.pkl`)
 
### Model Evaluation
 
- Loads the current production model from S3 (if one exists)
- Compares the **F1-score** of the new model and the production model on the test data
- If no model exists in S3 yet, the new model is accepted directly
### Model Pusher
 
- Uploads the accepted model to the S3 bucket
### Prediction Pipeline
 
- The FastAPI form data is converted into a DataFrame
- The model is loaded from S3 and used to predict
- The result is shown as `Response-Yes` or `Response-No`
---
 
## AWS Cloud Integration
 
### S3 Bucket
 
- Region: `ap-southeast-2`
- Bucket name: set in `src/constants/__init__.py` (`MODEL_BUCKET_NAME`). Use your own unique bucket name.
- Model object key: `model.pkl`
### IAM
 
- Create an IAM user with the required permissions and generate access keys (used as `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`)
---
 
## CI/CD Pipeline with GitHub Actions & Docker
 
The workflow is defined in `.github/workflows/aws.yaml` and has two jobs:
 
1. **Continuous-Integration** (`ubuntu-latest`): checkout code, configure AWS credentials, log in to ECR, build the Docker image and push it to ECR.
2. **Continuous-Deployment** (`self-hosted` runner on EC2, runs after CI): log in to ECR and run the image with `docker run -d -p 5000:5000`, passing the AWS and MongoDB secrets as environment variables.
Setup steps:
 
1. Create an ECR repository and an EC2 Ubuntu instance
2. Install Docker on EC2
3. Register the EC2 machine as a GitHub **self-hosted runner** (Settings > Actions > Runners) and start it with `./run.sh`
4. Add the secrets listed below to the GitHub repository
5. Open port `5000` in the EC2 Security Group (inbound rule)
6. Push to `main` to trigger the pipeline
> Note: make sure the lines in `aws.yaml` are **uncommented**, otherwise the workflow will not run.
 
---
 
## Access the Application
 
Once deployed:
 
```text
http://<EC2-PUBLIC-IP>:5000/
```
 
| Route          | Method | Purpose                                  |
| -------------- | ------ | ---------------------------------------- |
| `/`            | GET    | Shows the prediction form                |
| `/`            | POST   | Returns the prediction for submitted data|
| `/train`       | GET    | Triggers the full training pipeline      |
 
---
 
## Secrets Used in GitHub Actions
 
| Name                    | Description                           |
| ----------------------- | ------------------------------------- |
| `AWS_ACCESS_KEY_ID`     | AWS Access Key                        |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Key                        |
| `AWS_DEFAULT_REGION`    | Default AWS Region (`ap-southeast-2`) |
| `ECR_REPO`              | ECR repository name                   |
| `MONGODB_URL`           | MongoDB Atlas connection string       |
 
---
 
## Known Limitations
 
- SMOTEENN is applied to the test set as well as the training set, which can make test metrics look different from real-world performance
- The train-test split does not use a fixed `random_state`, so results change between runs
- `MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE` is defined but the evaluation currently only checks whether the new F1-score is higher
---
 
## Future Enhancements
 
- Add model drift detection
- Add monitoring with Prometheus/Grafana
- Integrate GitHub issues via bot
- Improve the frontend UI
- Switch to Terraform for infrastructure provisioning 
---
