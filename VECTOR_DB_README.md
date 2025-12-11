# ChromaDB 向量資料庫整合說明

## 功能特性

✅ **持久化儲存**：embeddings 儲存在本地 `./chroma_db` 目錄
✅ **自動檢測**：如果資料庫已存在，跳過 PDF 處理直接搜尋
✅ **快速啟動**：第二次執行只需幾秒鐘
✅ **本地儲存**：圖片路徑儲存在 metadata 中，實際圖片檔案保持在原位置

## 安裝依賴

```bash
pip install chromadb
```

## 使用方式

### 首次執行（建立資料庫）

```bash
python multimodal_rag.py
```

程式會：
1. 處理 PDF 文件
2. 生成 embeddings
3. 儲存到 ChromaDB
4. 執行查詢

### 後續執行（使用現有資料庫）

```bash
python multimodal_rag.py
```

程式會：
1. 檢測到現有資料庫
2. **跳過 PDF 處理**（節省大量時間）
3. 直接從資料庫搜尋
4. 執行查詢

## 資料結構

### Text Collection
```python
{
    "id": "text_document.pdf_1_1",
    "embedding": [0.1, 0.2, ...],  # 768維
    "document": "chunk text content",
    "metadata": {
        "file_name": "document.pdf",
        "page_num": 1,
        "chunk_number": 1,
        "type": "text"
    }
}
```

### Image Collection
```python
{
    "id": "img_document.pdf_1_1",
    "embedding": [0.1, 0.2, ...],  # 1408維
    "document": "image description",
    "metadata": {
        "file_name": "document.pdf",
        "page_num": 1,
        "img_num": 1,
        "img_path": "./images/doc_image_1_1.jpeg",
        "img_desc": "A table showing...",
        "type": "image"
    }
}
```

## 管理資料庫

### 查看現有 collections

```python
from vector_db import VectorDB

vector_db = VectorDB()
print(vector_db.list_collections())
print(f"Text count: {vector_db.get_collection_count('text_embeddings')}")
print(f"Image count: {vector_db.get_collection_count('image_embeddings')}")
```

### 強制重建資料庫

如果 PDF 內容有更新，需要重建：

```python
from vector_db import VectorDB, build_vector_db_from_dataframes

vector_db = VectorDB()

# 刪除舊的 collections
vector_db.delete_collection("text_embeddings")
vector_db.delete_collection("image_embeddings")

# 重新處理 PDF 並建立資料庫
text_df, image_df = create_get_metadata_df(path, model=multimodal_model)
build_vector_db_from_dataframes(
    text_df=text_df,
    image_df=image_df,
    vector_db=vector_db,
    force_rebuild=True
)
```

或者直接刪除資料庫目錄重新開始：

```bash
rm -rf ./chroma_db
python multimodal_rag.py
```

## 效能對比

### 首次執行（建立資料庫）
- PDF 處理：5-10 分鐘（取決於文件數量）
- Embedding 生成：需要呼叫 Vertex AI API
- 資料庫建立：幾秒鐘

### 後續執行（使用現有資料庫）
- PDF 處理：**跳過** ⚡
- Embedding 生成：**跳過** ⚡
- 資料庫搜尋：< 1 秒 ⚡
- **總時間：幾秒鐘完成查詢！**

## 目錄結構

```
multimodal-rag/
├── chroma_db/              # ChromaDB 持久化目錄（自動建立）
│   ├── chroma.sqlite3
│   └── ...
├── images/                 # 從 PDF 提取的圖片
│   └── *.jpeg
├── data2/                  # PDF 原始檔案
│   └── *.pdf
├── vector_db.py           # 向量資料庫工具
├── intro_multimodal_rag_utils.py
└── multimodal_rag.py      # 主程式
```

## 常見問題

### Q: 如何更新單一文件？
A: 目前需要刪除整個 collection 重建。未來可以實作增量更新。

### Q: 資料庫檔案有多大？
A: 取決於文件數量。大約：
- 每個文字 chunk：~3KB
- 每個圖片 embedding：~6KB
- 100 頁 PDF：約 5-10MB

### Q: 可以備份資料庫嗎？
A: 可以！直接複製 `./chroma_db` 目錄即可。

### Q: 支援多個專案嗎？
A: 可以為不同專案使用不同的 `persist_directory`：
```python
vector_db_project1 = VectorDB(persist_directory="./chroma_db_project1")
vector_db_project2 = VectorDB(persist_directory="./chroma_db_project2")
```

## 進階功能

### 按文件名過濾搜尋

```python
# 只搜尋特定文件
results = vector_db.search_similar(
    collection_name="text_embeddings",
    query_embedding=query_embedding,
    top_k=5,
    filter_dict={"file_name": "specific_document.pdf"}
)
```

### 調整搜尋結果數量

修改 `multimodal_rag.py` 中的 `top_k` 參數：

```python
text_results = vector_db.search_similar(
    collection_name=text_collection_name,
    query_embedding=query_embedding,
    top_k=10  # 返回前 10 個結果
)
```
