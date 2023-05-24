from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    GPTVectorStoreIndex,
    LLMPredictor,
    PromptHelper,
    )
from langchain import OpenAI
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = ''

def construct_index(directory_path):
    # Configuration parameters
    MAX_INPUT_SIZE = 4096
    NUM_OUTPUTS = 512
    MAX_CHUNK_OVERLAP = 20
    CHUNK_SIZE_LIMIT = 600


    # Initialize and configure components
    prompt_helper = PromptHelper(
        MAX_INPUT_SIZE, NUM_OUTPUTS, MAX_CHUNK_OVERLAP, chunk_size_limit=CHUNK_SIZE_LIMIT
    )
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0, model_name="text-davinci-003", max_tokens=NUM_OUTPUTS
        )
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    # Load documents and create index
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTVectorStoreIndex.from_documents(
        documents=documents, service_context=service_context
    )

    #index.save_to_disk('index.json')

    return index

def process_query(input_text):
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(input_text)
        return (
            str(response.response) + "\n\n" + str(response.get_formatted_sources())
        )


    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        return "An error occurred. Please try again."

index = construct_index("docs")

# Create Gradio interface
iface = gr.Interface(
    fn=process_query,
    inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
    outputs="text",
    title="Custom-Trained AI",
)

iface.launch(server_name="0.0.0.0", server_port=7060, share=False)