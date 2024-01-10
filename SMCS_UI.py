from flask import Flask, render_template, request
import pandas as pd
import openai
import numpy as np
from openai import AzureOpenAI
from azure.storage.filedatalake import DataLakeServiceClient
import pyarrow.parquet as pq
import io
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


app = Flask(__name__, static_folder='templates', template_folder='templates')

# Read embedding file from ADLS

key_vault_url = "https://marketingbidevkeyvault.vault.azure.net/"
secret_name = "smcsadls"

    # Use DefaultAzureCredential to authenticate
credential = DefaultAzureCredential()
    
    # Create a SecretClient using the key vault URL and credential
secret_client = SecretClient(vault_url=key_vault_url, credential=credential)

    # Get the secret value
secret = secret_client.get_secret(secret_name)

# Print the secret value
def get_json_value(api_key, value):
    import json
    api_key_json = json.loads(api_key)
    key = api_key_json.get(value)
    return key


account_key = get_json_value(secret.value, "account_key")
account_name = 'fmoadlseastusppe'
file_system_name = 'mash-stage-silver'
directory_name = "Gen_AI/ICM_EMBEDDING/"
file_name = "Embedding.parquet"


file_path = directory_name + file_name

source_service_client = DataLakeServiceClient(account_url=f"https://{account_name}.dfs.core.windows.net", credential=account_key)
source_file_system_client = source_service_client.get_file_system_client(file_system=file_system_name)
source_file_client = source_file_system_client.get_file_client(file_path=file_path)

        # Read data from source
data = source_file_client.download_file().readall()

# Use pyarrow to read the Parquet file from the downloaded content
table = pq.read_table(io.BytesIO(data))

# Convert the table to a pandas DataFrame if needed
df = table.to_pandas()
print(df.head())

#############################################################################################################
# Get api_key in the environment variable

key_vault_url = "https://marketingbidevkeyvault.vault.azure.net/"
secret_name = "smcsopenai"

    # Use DefaultAzureCredential to authenticate
credential = DefaultAzureCredential()
    
    # Create a SecretClient using the key vault URL and credential
secret_client = SecretClient(vault_url=key_vault_url, credential=credential)

    # Get the secret value
secret = secret_client.get_secret(secret_name)

api_key = get_json_value(secret.value, "api_key")

import os
os.environ["OPENAI_API_KEY"] = api_key
#######################################################################
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(text):
    client = AzureOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],  
    api_version="2023-12-01-preview",
    azure_endpoint = "https://smcsopenai.openai.azure.com/")
    text = text.replace("\n", " ")
    embedding = client.embeddings.create(input=[text],model="SMCSEMBEDDING")
    return embedding.data[0].embedding


def search_response(df, query, api_key, n=3, pprint=True):
    query_embedding = get_embedding(query)
    df["similarity"] = df.Embedding.apply(lambda x: cosine_similarity(x, query_embedding))
    df.sort_values("similarity", ascending=False, inplace=True)
    df = df[df.similarity >= 0.7]
    df1 = df.head(n)

    if len(df1) > 0:
        result = []
        for i in range(len(df1)):
            response ={'Mitigation': [], 'Comments':[]}
            row = df1.iloc[i]  # Access the ith row
            mitigation_value = row['Mitigation']
            #description_value = row['Description']
            text_value = row['Text']
            #summary_value = row['Summary']
            
            response['Mitigation'] = mitigation_value
            #response['Description'] = description_value
            response['Comments'] = text_value
            #response['RootCause'] = summary_value
            result.append(response)
        response = result
    else:
        response = ["Please rephrase your query..."]

    return response

def prediction(query, df):
    #api_key = 
    results = search_response(df, query, api_key)
    if isinstance(results[0], dict):
        final_result = ""
        for i, response in enumerate(results):
            print("________________________________________________________________________________________________________________________________________________________")
            final_result += "________________________________________________________________________________________________________________________________________________________\n"
            print(f"Response: {i + 1} \n")
            final_result += f"Response: {i + 1} \n"
            for key, value in response.items():
                print(key, ": ", value, "\n")
                final_result += "<h5>" + key + "</h5>" + value + "\n"
        return final_result
    else:
        return results[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        user_query = request.form['user_query']
        result = prediction(user_query, df)
        return render_template("results.html", result=result, query=user_query)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
