# import pdfplumber
import time
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from transformers import AutoTokenizer, AutoModel
# from transformers import LongformerTokenizer, LongformerModel
from confluence import get_conf_data
# Load pre-trained model and tokenizer
# tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
# model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')

pdf_path = '/Users/akdiwaka/Downloads/PA_Checklist_Automation_guide.pdf'
output_path = 'output.txt'
index_name = 'pdf-index'


# def extract_text_from_pdf(_pdf_path):
#     structured_data = []
#     with pdfplumber.open(_pdf_path) as pdf:
#         for page_num, page in enumerate(pdf.pages):
#             text = page.extract_text(layout=True)
#             structured_data.append({
#                 'page': page_num,
#                 'text': text
#             })
#     return structured_data

# structured_text = extract_text_from_pdf(pdf_path)
structured_text = get_conf_data()
# breakpoint()
# Initialize Pinecone
pc = Pinecone(api_key='*****')

# Create an index if it doesn't already exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
# Connect to the index
index = pc.Index(index_name)


# Store embeddings with metadata
def store_embeddings(_embeddings, _segments, _page_num):
    # breakpoint()
    # chunk = 0
    for i, embedding in enumerate(_embeddings):

        vector = {
            'id': f'page_{_page_num}_segment_{i}',
            'values': embedding,
            'metadata': {'text': _segments[i], 'chunk': len(_embeddings)}
        }
        index.upsert(vectors=[vector])


def save_text_to_file(text, _output_path):
    with open(_output_path, 'w') as file:
        file.write(text)


def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings.flatten().tolist()

# breakpoint()
# structured_text = extract_text_from_pdf(pdf_path)
for data in structured_text:
    # Split text into chunks
    segments = [data['text'][i:i + 512] for i in range(0, len(data['text']), 512)]
    embeddings = [embed_text(segment) for segment in segments]
    store_embeddings(embeddings, segments, data['page'])


# save_text_to_file(structured_text, output_path)
# print(f"Extracted text saved to {output_path}")