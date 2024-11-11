## Prerequisites

- An Azure account with the following services set up:
  - Azure OpenAI
  - Azure Cognitive Search
  - Azure Text Analytics

## Setup
1. **Create a `API_KEY.json` file to store your API keys and endpoints.** Place this file in the same directory as `app.py`. The `API_KEY.json` file should have the following structure:

```json
{
     "AZURE_OAI_ENDPOINT": "<Your Azure OpenAI Endpoint>",
     "AZURE_OAI_KEY": "<Your Azure OpenAI Key>",
     "AZURE_OAI_DEPLOYMENT": "<Your Azure OpenAI Deployment Name>",
     "AZURE_SEARCH_ENDPOINT": "<Your Azure Cognitive Search Endpoint>",
     "AZURE_SEARCH_KEY": "<Your Azure Cognitive Search Key>",
     "AZURE_SEARCH_INDEX": "<Your Azure Cognitive Search Index Name>",
     "AZURE_LANGUAGE_KEY": "<Your Azure Text Analytics Key>",
     "AZURE_LANGUAGE_ENDPOINT": "<Your Azure Text Analytics Endpoint>"
}
```
2. **Install the required Python packages.** Before running the application, set up your Python environment and install the dependencies by running:
```bash
pip install -r requirements.txt
```
## Running 
1. Navigate to the directory containing `app.py` and `API_KEY.json`.
2. Run the application using the following command:
```bash
python app.py
```

## Changes

- **System Prompt Modification:** The `system` prompt has been modified to address an issue where the model did not generate responses when no data was retrieved from the database. The updated prompt ensures that the model generates a helpful response even if the search does not return relevant data.
```prompt
Retrieve information by first querying the RAG database. If the query yields no relevant results, automatically generate a response using the AI model.
```
- **Code Adjustments:** Modifications have been made in the `chat_function` to handle scenarios where the assistant's reply indicates that the requested information was not found in the retrieved data. In such cases, a fallback call is made to the Azure OpenAI model without the data sources parameter, allowing the model to generate a response based solely on its training data.