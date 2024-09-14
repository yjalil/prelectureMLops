# Step 1 : Uploaded data to bigquery
# Step 2 : Created my local dir for the project
# Step 3 : Create the env
# Step 4 : Connect to bigquery
# Step 5 : Setup GCP
```
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get install apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-sdk
sudo apt-get install google-cloud-sdk-app-engine-python
```

gcloud init

gcloud auth application-default login

gcloud auth login

## Create Service account
