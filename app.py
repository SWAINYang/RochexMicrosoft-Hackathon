import os
import json
import gradio as gr

from openai import AzureOpenAI
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

def authenticate_client(language_key, language_endpoint):
    ta_credential = AzureKeyCredential(language_key)
    text_analytics_client = TextAnalyticsClient(
        endpoint=language_endpoint,
        credential=ta_credential)
    return text_analytics_client

with open('API_KEY.json', 'r') as f:
    config = json.load(f)

azure_oai_endpoint = config["AZURE_OAI_ENDPOINT"]
azure_oai_key = config["AZURE_OAI_KEY"]
azure_oai_deployment = config["AZURE_OAI_DEPLOYMENT"]
azure_search_endpoint = config["AZURE_SEARCH_ENDPOINT"]
azure_search_key = config["AZURE_SEARCH_KEY"]
azure_search_index = config["AZURE_SEARCH_INDEX"]
azure_language_key = config["AZURE_LANGUAGE_KEY"]
azure_language_endpoint = config["AZURE_LANGUAGE_ENDPOINT"]

client = AzureOpenAI(
    base_url=f"{azure_oai_endpoint}/openai/deployments/{azure_oai_deployment}",
    api_key=azure_oai_key,
    api_version="2024-05-01-preview")

client_ta = authenticate_client(azure_language_key, azure_language_endpoint)

def chat_function(message, history=None, remove_pii=False, temperature=0.7, top_p=0.95):
    if history is None:
        history = []
    try:
        messages = [{"role": "system", "content": "Retrieve information by first querying the RAG database. If the query yields no relevant results, automatically generate a response using the AI model."}]

        if history:
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})

        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model=azure_oai_deployment,
            messages=messages,
            max_tokens=800,
            temperature=temperature,  
            top_p=top_p,              
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            extra_body={
                "data_sources": [
                    {
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": azure_search_endpoint,
                            "index_name": azure_search_index,
                            "authentication": {
                                "type": "api_key",
                                "key": azure_search_key
                            }
                        }
                    }
                ]
            }
        )

        assistant_reply = response.choices[0].message.content

        if "the requested information is not found in the retrieved data" in assistant_reply.lower():
            response_fallback = client.chat.completions.create(
                model=azure_oai_deployment,
                messages=messages,
                max_tokens=800,
                temperature=temperature,  
                top_p=top_p               
            )
            assistant_reply = response_fallback.choices[0].message.content

        if remove_pii:
            response_PII = client_ta.recognize_pii_entities([assistant_reply], language="en")
            result = [doc for doc in response_PII if not doc.is_error]
            for doc in result:
                assistant_reply = doc.redacted_text

        history.append([message, assistant_reply])
        return history

    except Exception as ex:
        print(ex)
        history.append(["System", "An error occurred: " + str(ex)])
        return history

css = """
/* General container and font styling */
.gradio-container {
    font-family: Arial, sans-serif;
    color: #333; /* Dark text color for better readability */
    background-color: #f9f9f9; /* Light grey background color for the container */
}

/* Styling the chat history */
.gradio-chatbot {
    height: 300px;
    color: #333; /* Dark text color for better readability */
    background-color: #ffffff; /* White background for the chatbot */
    border: 1px solid #e0e0e0; /* Light grey border for subtle definition */
}

/* Styling the text input box */
.gr-textbox input {
    width: 100%; /* Full width */
    padding: 10px 15px;
    background-color: #ffffff; /* White background for the input */
    border: 1px solid #e0e0e0; /* Light grey border */
    color: #333; /* Dark text color */
}

/* Styling the checkbox */
.gr-checkbox input[type='checkbox'] {
    transform: scale(1.2); /* Slightly larger checkbox for easier interaction */
}
.gr-checkbox label {
    color: #333; /* Dark text color */
    font-size: 25px; /* Sufficient font size for readability */
}

/* Styling buttons */
.gr-button button {
    width: 100%;
    background-color: #e0e0e0; /* Light grey background for buttons */
    color: #333; /* Dark text color */
    border: 1px solid #ccc; /* Light grey border */
    padding: 10px 0;
    margin-top: 10px;
    border-radius: 5px;
    font-size: 30px;
}
"""
with gr.Blocks(css=css) as demo:
    gr.HTML("""
    <div style='display: flex; align-items: center;'>
        <img src='https://www.pngitem.com/pimgs/m/410-4101985_la-roche-pharma-ag-roche-pharma-logo-hd.png' style='width: 50px; margin-right: 10px;' alt='Brand Logo'/>
        <span style='font-size: 36px;'>Chat Interface</span>
    </div>
    """)
    chat_history = gr.Chatbot(label="Chat History")
    user_input = gr.Textbox(label="Enter your message")
    remove_pii_checkbox = gr.Checkbox(label="Enable PII Removal", value=False)

    with gr.Row():
        temperature_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.05,
            label="Temperature",
            info="Controls the randomness of the model's output. Higher values (e.g., 0.8) make the output more random and creative. Lower values (e.g., 0.2) make it more focused and deterministic."
        )
        top_p_slider = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (Nucleus Sampling)",
            info="Limits the model's token selection to the top probability `p` sum. Lower values (e.g., 0.5) make the output more focused, while higher values (e.g., 0.95) allow for more diversity."
        )

    submit_btn = gr.Button("Send")

    submit_btn.click(
        fn=chat_function,
        inputs=[user_input, chat_history, remove_pii_checkbox, temperature_slider, top_p_slider],
        outputs=chat_history
    )

if __name__ == "__main__":
    demo.launch()