import chainlit as cl
#import torch
import chromadb
#from transformers import BitsAndBytesConfig
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Pairsing_docx_file import docx2md, docx2txt

# Load model retrieval Bge-m3
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) 

# Load model generation Qwen 2.5 - 0.5B
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Create collection for storing vector data
client = chromadb.Client()
collection = client.create_collection("collection")
# Create text_splitter to split text get from doccument
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)

# Path to document file
file_path = "data\data.docx" 

def get_vector_db(file_path):
    '''
    This function create vectorDB storing data from document 
    '''
    text = docx2md(file_path)
    #text = docx2txt(file_path)
    chunks = text_splitter.split_text(text)
    docs_embedding = []

    for docs in chunks:
        docs_embedding.append(model.encode(str(docs))['dense_vecs']) # Dense Embedding data from document

    for i, doc in enumerate(chunks):
      collection.add(
        documents=[str(doc)],  
        embeddings=[docs_embedding[i]], 
        metadatas=[{"source": f"doc_{i}"}],
        ids=[f"doc_{i}"],  
    )
    return collection

collection = get_vector_db(file_path) # Generate Collection

 
def generate_answer(prompt):
    '''
    This function is usage of Qwen 2.5-0.5B
    '''
    #Quantinized model with BitsAndBytes
    '''
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    '''
    # Config model for low gpu
    model_gen = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                low_cpu_mem_usage = True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model_gen.device)

    generated_ids = model_gen.generate(
        **model_inputs,
        max_new_tokens= 512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def generate_answer_from_retrieval(query, vector_db):
    '''
    This function generate answer from data retrieval from vector DataBase by following step:
    - Embedding query(question)
    - Retrieval from vectorDB texts similarity with query by using compute distance from 2 vector
    - Calculate the distance:
        + If the distance is too big. Lets the model anwser form pretrained data
        + If the distance normal. Create prompt with data retrieved and query
    - Send prompt to model and return processed answer
    '''

    # Retrieval data from collection
    embedding_query = model.encode(query)['dense_vecs']
    results = vector_db.query(
        query_embeddings=embedding_query,
        n_results = 3
    )

    if float(results['distances'][0][0]) < 1.5:
        # Create prompt
        prompt ='You are a chatbot QA, your job is answer questions based on the information provided. You only can answer the question with information get from information below. Here is the information: \n'
        for doc in results['documents']:
            prompt += f"{doc[0]}\n{doc[1]}\n"  
        prompt +=  f"Answer the following question only based on information above: \"{query}\"\n\n"

        # Get answer from model (Qwen2.5)
        answer = generate_answer(prompt) 
        return answer
    else:
        generation_ans = generate_answer(query)
        answer = "I do not have information from the given data for this question. The answer may not be correct... \n" + generation_ans
        return answer


# Chainlit UI
@cl.on_message
async def handle_message(message: cl.Message):
    response = generate_answer_from_retrieval((message.content), collection)
    await cl.Message(content=response).send()