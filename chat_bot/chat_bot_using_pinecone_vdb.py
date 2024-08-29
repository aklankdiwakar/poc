import pdfplumber
import streamlit as st
import time
import torch
from transformers import LongformerTokenizer, LongformerModel
from pinecone import ServerlessSpec
# from pinecone.grpc import PineconeGRPC
from pinecone import Pinecone as PineconeGRPC
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
# tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
# model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
time.sleep(15)

# lm_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
# # lm_model = AutoModelForCausalLM.from_pretrained('facebook/bart-large-cnn')
# lm_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')
# lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# lm_tokenizer.pad_token = lm_tokenizer.eos_token./
# lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
# lm_tokenizer.pad_token_id = lm_tokenizer.eos_token_id
#
# print("Vocabulary size:", len(lm_tokenizer))
# print("Embedding size:", lm_model.config.vocab_size)

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the tokenizer and model
# lm_tokenizer = T5Tokenizer.from_pretrained('t5-large')
# lm_model = T5ForConditionalGeneration.from_pretrained('t5-large')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# lm_model.to(device)
# lm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token='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')
# lm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token='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')
summarizer = pipeline('text-generation', model='gpt2')

pdf_path = '/Users/akdiwaka/Downloads/PA_Checklist_Automation_guide.pdf'
output_path = 'output.txt'

# Initialize Pinecone
pc = PineconeGRPC(api_key='****')

######################################################################
import openai
# from langchain_community.vectorstores import Pinecone
# get openai api key from platform.openai.com
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'
#
openai.api_key = '**'
# model_name = 'text-embedding-ada-002'
# query = 'Data Collection at Source'
# res = openai.Embedding.create(
#     input=[query],
#     engine=model_name
# )
#
# text_field = "text"
#
# # switch back to normal index for langchain
# index = pc.Index(index_name)
#
# xq = res['data'][0]['embedding']
#
# # get relevant contexts (including the questions)
# res = index.query(vector=xq, top_k=5, include_metadata=True)

# query = "Data Collection at Source"

# vectorstore.similarity_search(
#     query,  # our search query
#     k=3  # return 3 most relevant docs
# )
##########################################################################################

# breakpoint()
# Connect to the index
index = pc.Index('pdf-index')


def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings.flatten().tolist()


def query_vector_db(_query, top_k=2):
    query_embedding = embed_text(_query)
    print(query_embedding)
    _results = index.query(vector=query_embedding, include_metadata=True, top_k=top_k)
    return _results


def summarize_text(text, max_length=512):
    summary = summarizer(text, max_length=max_length, pad_token_id=50256, truncation=True,
                         num_return_sequences=1)
    return summary[0]['generated_text']



# def generate_response_with_llm(_query, _summaries):
#     # Join summaries and create the input text
#     context = " ".join(_summaries)
#     input_text = f"Context: {context}\n\nQuestion: {_query}\n\nAnswer:"
#
#     # Tokenize the input text without padding (GPT-2 does not use padding tokens)
#     inputs = lm_tokenizer(input_text, return_tensors='pt')
#     print("Input IDs:", inputs['input_ids'])
#     print("Max ID:", inputs['input_ids'].max().item())
#     token_ids = inputs['input_ids'].squeeze().tolist()
#     max_id = max(token_ids)
#     vocab_size = lm_tokenizer.vocab_size
#     if max_id >= vocab_size:
#         raise ValueError(f"Token ID {max_id} exceeds vocab size {vocab_size}.")
#
#     # Move model to the appropriate device (CPU or GPU)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     lm_model.to(device)
#
#     # Move input tensors to the same device as the model
#     input_ids = inputs['input_ids'].to(device)  # This should be a 2D tensor
#     attention_mask = inputs.get('attention_mask')
#     if attention_mask is not None:
#         attention_mask = attention_mask.to(device)  # Ensure this tensor is also moved to the device
#     breakpoint()
#
#     # Generate the output
#     outputs = lm_model.generate(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         max_new_tokens=256,  # Set the number of new tokens to generate
#         num_return_sequences=1,
#         pad_token_id=lm_tokenizer.pad_token_id
#         # Set pad_token_id to eos_token_id for open-end generation
#     )
#
#     # Decode the generated text
#     response = lm_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response


# def generate_response_with_llm(_query, _summaries):
#     # context = " ".join(_summaries)
#     # formatted_summaries = "\n".join([f"- {summary.strip()}" for summary in _summaries])
#     # input_text = f"Context: {context}\n\nQuestion: {_query}\n\nAnswer:"
#
#     formatted_summaries = ""
#     for summary in _summaries:
#         # Split the summary into smaller lines if necessary
#         if len(summary) > 100:  # Adjust this limit as needed
#             parts = [summary[i:i+100] for i in range(0, len(summary), 100)]
#             formatted_summaries += "\n".join([f"- {part.strip()}" for part in parts]) + "\n"
#         else:
#             formatted_summaries += f"- {summary.strip()}\n"
#
#     input_text = (
#         f"Context:\n{formatted_summaries}\n\n"
#         f"Question: {_query}\n\n"
#         f"Answer the question by referring to the above steps and explain clearly."
#     )
#     print(input_text)
#     inputs = lm_tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True,
#                           padding='longest').to(device)
#     # inputs = lm_tokenizer.encode(input_text, return_tensors='pt', max_length=512,
#     #                              truncation=True)
#     # outputs = lm_model.generate(inputs, max_length=512, num_return_sequences=100)
#     # Generate multiple response sequences using beam search
#     outputs = lm_model.generate(
#         input_ids=inputs['input_ids'],
#         attention_mask=inputs['attention_mask'],
#         max_length=512,
#         num_beams=10,
#         num_return_sequences=10,
#         no_repeat_ngram_size=2,  # Helps to avoid repeating phrases
#         early_stopping=True
#     )
#     # breakpoint()
#
#     responses = [lm_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
#     return responses


# def generate_response_with_llm(_query, _summaries):
#     response = ""
#     for summary in _summaries:
#         context = "Existing OCI CDB DB Cold Backup Steps (Iteration 1 only)"
#         input_text = f"Context: {context}\n\nSteps: {summary}\n\nAnswer (in bullet points):"
#
#         inputs = lm_tokenizer.encode(input_text, return_tensors='pt', max_length=512,
#                                      truncation=True)
#         outputs = lm_model.generate(inputs, max_length=512, num_return_sequences=1)
#         response += lm_tokenizer.decode(outputs[0], skip_special_tokens=True) + "\n"
#     return response.strip()


# # Streamlit Interface
# st.title("PDF Processing with Pinecone and Transformers")
# # Example query
# # query = "Get me command line to run pa check automation?"
# query = st.text_input("Enter your query:")
# if query:
#     with st.status("Getting your data...", expanded=True) as status1:
#         st.write("Searching for data...")
#         results = query_vector_db(query)
#         top_k_results = [result['metadata']['text'] for result in results['matches']]
#
#         summaries = [summarize_text(result) for result in top_k_results]
#         response_with_llm = generate_response_with_llm(query, summaries)
#         st.write(f"Match: {response_with_llm}")
#         # for result in results['matches']:
#         #     st.write(f"Match: {result['metadata']['text']} (Score: {result['score']})")
#         status1.update(label="Download complete!", state="complete", expanded=False)
#     st.button("Rerun")


# query = 'Backup Module Configuration at Target'
# query = 'Configure Backup Module'
query = 'How to take the existing OCI CDB DB cold backup at OSS'

# query = 'Data Collection at Source'
results = query_vector_db(query)
# breakpoint()
# top_k_results = [(result['metadata']['text'], result['id'], int(result['metadata']['chunk']))
#                  for result in results['matches']]


def get_id_list(_result):
    id_list = list()
    vid, chunk = result['id'], int(_result['metadata']['chunk'])
    _id = vid.rsplit('_', 1)[0]
    for each_id in list(map(lambda x: '_'.join([_id, str(x)]), range(0, chunk))):
        _results = index.query(id=each_id, include_metadata=True, top_k=1)
        id_list.append(_results['matches'][0]['metadata']['text'])
    return id_list


for result in results['matches']:
    top_k_results = get_id_list(result)

# print(top_k_results)
# breakpoint()
summaries = [summarize_text(result)+'******\n' for result in top_k_results]
# response_with_llm = generate_response_with_llm(query, top_k_results)
print(summaries)
