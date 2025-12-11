import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image
from PIL import Image as PILImage
from intro_multimodal_rag_utils import create_get_metadata_df
from intro_multimodal_rag_utils import (
    display_images,
    get_gemini_response,
    get_text_embedding_from_text_embedding_model,
    print_text_to_image_citation,
    print_text_to_text_citation,
)
import time
from vector_db import VectorDB, build_vector_db_from_dataframes

# Load environment variables from parent directory's .env file
env_path =  '.env'
load_dotenv(dotenv_path=env_path)
# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")

print(f"Using Project ID: {PROJECT_ID}")
print(f"Using Location: {LOCATION}")
print(f"Credentials: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
text_model = GenerativeModel("gemini-2.5-flash")
multimodal_model = text_model
multimodal_model_flash = text_model

st = time.time()

# 初始化 VectorDB
vector_db = VectorDB(persist_directory="./chroma_db")

# 設定資料路徑和 collection 名稱
path = "data2/"
text_collection_name = "text_embeddings"
image_collection_name = "image_embeddings"
image_text_collection_name = "image_text_embeddings"  # 新增：用文字 embedding 搜尋圖片

# 檢查是否需要建立 embeddings
collections_exist = (
    vector_db.collection_exists(text_collection_name) and 
    vector_db.collection_exists(image_collection_name) and
    vector_db.collection_exists(image_text_collection_name)
)

if collections_exist:
    print("\n✓ Vector database already exists. Skipping PDF processing...")
    print(f"  - Text embeddings: {vector_db.get_collection_count(text_collection_name)}")
    print(f"  - Image embeddings (multimodal): {vector_db.get_collection_count(image_collection_name)}")
    print(f"  - Image embeddings (text): {vector_db.get_collection_count(image_text_collection_name)}")
    
    # 不需要重新處理 PDF，直接設定為 None
    text_metadata_df = None
    image_metadata_df = None
else:
    print("\n📄 Processing PDFs and building embeddings...")
    text_metadata_df, image_metadata_df = create_get_metadata_df(path, model=multimodal_model)
    
    print("\n\n --- Completed processing. ---")
    print(text_metadata_df.head())
    print(image_metadata_df.head())
    
    # 建立向量資料庫
    print("\n💾 Building vector database...")
    
    # 1. Text embeddings
    if not vector_db.collection_exists(text_collection_name):
        texts = text_metadata_df['chunk_text'].tolist()
        embeddings = text_metadata_df['text_embedding_chunk'].tolist()
        metadatas = []
        ids = []
        for idx, row in text_metadata_df.iterrows():
            metadata = {
                'file_name': row['file_name'],
                'page_num': int(row['page_num']),
                'chunk_number': int(row['chunk_number']),
                'type': 'text'
            }
            metadatas.append(metadata)
            ids.append(f"text_{row['file_name']}_{row['page_num']}_{row['chunk_number']}")
        
        vector_db.add_text_embeddings(
            collection_name=text_collection_name,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    # 2. Image multimodal embeddings (1408維)
    if not vector_db.collection_exists(image_collection_name):
        descriptions = image_metadata_df['img_desc'].tolist()
        embeddings = image_metadata_df['mm_embedding_from_img_only'].tolist()
        metadatas = []
        ids = []
        for idx, row in image_metadata_df.iterrows():
            metadata = {
                'file_name': row['file_name'],
                'page_num': int(row['page_num']),
                'img_num': int(row['img_num']),
                'img_path': row['img_path'],
                'img_desc': row['img_desc'],
                'type': 'image'
            }
            metadatas.append(metadata)
            ids.append(f"img_{row['file_name']}_{row['page_num']}_{row['img_num']}")
        
        vector_db.add_image_embeddings(
            collection_name=image_collection_name,
            image_descriptions=descriptions,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    # 3. Image text embeddings (768/3072維，與 text embedding 相同維度)
    if not vector_db.collection_exists(image_text_collection_name):
        descriptions = image_metadata_df['img_desc'].tolist()
        embeddings = image_metadata_df['text_embedding_from_image_description'].tolist()
        metadatas = []
        ids = []
        for idx, row in image_metadata_df.iterrows():
            metadata = {
                'file_name': row['file_name'],
                'page_num': int(row['page_num']),
                'img_num': int(row['img_num']),
                'img_path': row['img_path'],
                'img_desc': row['img_desc'],
                'type': 'image_text'
            }
            metadatas.append(metadata)
            ids.append(f"imgtxt_{row['file_name']}_{row['page_num']}_{row['img_num']}")
        
        collection = vector_db.get_or_create_collection(image_text_collection_name)
        collection.add(
            documents=descriptions,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✓ Added {len(descriptions)} image text embeddings to {image_text_collection_name}")
    
    print("\n✓ Vector database built successfully!")


query = """知識庫可以獨自設定每個知識庫能否下載嗎？該如何操作？"""

print(f"\n🔍 Searching for: {query}")

# 2. 從向量資料庫搜尋相關的 text chunks 和 images
# 取得 query embedding
query_embedding = get_text_embedding_from_text_embedding_model(query)

# 搜尋相似的文字
print("\n📝 Searching text embeddings...")
text_results = vector_db.search_similar(
    collection_name=text_collection_name,
    query_embedding=query_embedding,
    top_k=5
)

# 整理文字結果
matching_results_chunks_data = {}
for i, (doc, metadata, distance) in enumerate(zip(
    text_results['documents'][0],
    text_results['metadatas'][0],
    text_results['distances'][0]
)):
    matching_results_chunks_data[i] = {
        'chunk_text': doc,
        'file_name': metadata['file_name'],
        'page_num': metadata['page_num'],
        'chunk_number': metadata['chunk_number'],
        'cosine_score': round(1 - distance, 2)  # 轉換為相似度分數
    }

print(f"✓ Found {len(matching_results_chunks_data)} relevant text chunks")

# 搜尋相似的圖片（使用 text embedding 搜尋 image_text_embeddings）
print("\n🖼️  Searching image embeddings...")
image_results = vector_db.search_similar(
    collection_name=image_text_collection_name,  # 改用 text embedding collection
    query_embedding=query_embedding,
    top_k=5
)

# 整理圖片結果
matching_results_image_fromdescription_data = {}
for i, (doc, metadata, distance) in enumerate(zip(
    image_results['documents'][0],
    image_results['metadatas'][0],
    image_results['distances'][0]
)):
    matching_results_image_fromdescription_data[i] = {
        'img_path': metadata['img_path'],
        'img_desc': metadata['img_desc'],
        'image_description': doc,
        'file_name': metadata['file_name'],
        'page_num': metadata['page_num'],
        'img_num': metadata['img_num'],
        'cosine_score': round(1 - distance, 2),
        'image_object': Image.load_from_file(metadata['img_path'])  # 載入圖片
    }

print(f"✓ Found {len(matching_results_image_fromdescription_data)} relevant images")

# 3.
# combine all the selected relevant text chunks
context_text = []
for key, value in matching_results_chunks_data.items():
    context_text.append(value["chunk_text"])
final_context_text = "\n".join(context_text)

# combine all the relevant images and their description generated by Gemini
context_images = []
for key, value in matching_results_image_fromdescription_data.items():
    context_images.extend(
        ["Image: ", value["image_object"], "Caption: ", value["image_description"]]
    )


# 4. create the prompt for Gemini model
prompt = f""" Instructions: use the images and the text provided as Context: to answer Question:
Make sure to think thoroughly before answering the question and put the necessary steps to arrive at the answer in bullet points for easy explainability.
If unsure, respond, "Not enough context to answer".

Context:
 - Text Context:
 {final_context_text}
 - Image Context:
 {context_images}
 
請使用下列問題的語言做回答：
{query}

Answer:
"""

# Generate Gemini response with streaming output
ret = get_gemini_response(
    multimodal_model,
    model_input=[prompt],
    stream=True,
    generation_config=GenerationConfig(temperature=1),
)
print("\n\n---------------Gemini Response------------------\n")
print(ret)

print("---------------Matched Images------------------\n")
print(matching_results_image_fromdescription_data[0]["img_path"],
        matching_results_image_fromdescription_data[1]["img_path"],)
print("spend time:", time.time() - st)
# display_images(
#     [
#         matching_results_image_fromdescription_data[0]["img_path"],
#         matching_results_image_fromdescription_data[1]["img_path"],
#         matching_results_image_fromdescription_data[2]["img_path"],
#         matching_results_image_fromdescription_data[3]["img_path"],
#     ],
#     resize_ratio=0.5,
# )
