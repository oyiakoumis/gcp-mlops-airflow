# MLOps pipeline with Apache Airflow (GCP)

This repository is a personal project I made to deploy an MLOps pipeline on **Cloud Composer** (**Apache Airflow**):

- The /dags folder contains the Airflow dag that trains a random-forest algorithm every 2 weeks, when train data is shared on **Cloud Storage** (via `GoogleCloudStorageObjectSensor`). The script then deploys the newly trained model on **AI Platform**.
  
- The /function-source folder contains the source code to deploy the ML model over a **Cloud Functions** endpoint. The API is secured via an API key.