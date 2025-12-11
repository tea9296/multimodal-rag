"""
Vector Database utility using ChromaDB for persistent storage
"""
import os
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import numpy as np


class VectorDB:
    """管理 ChromaDB 向量資料庫的類別"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        初始化 ChromaDB client
        
        Args:
            persist_directory: 資料庫持久化目錄
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # 初始化 ChromaDB client (persistent)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        print(f"ChromaDB initialized at: {persist_directory}")
    
    def get_or_create_collection(
        self, 
        collection_name: str,
        embedding_dimension: int = 768
    ) -> chromadb.Collection:
        """
        取得或建立 collection
        
        Args:
            collection_name: collection 名稱
            embedding_dimension: embedding 維度
            
        Returns:
            ChromaDB collection
        """
        try:
            # 嘗試取得現有 collection
            collection = self.client.get_collection(name=collection_name)
            print(f"✓ Loaded existing collection: {collection_name} (count: {collection.count()})")
        except:
            # 建立新 collection
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"dimension": embedding_dimension}
            )
            print(f"✓ Created new collection: {collection_name}")
        
        return collection
    
    def add_text_embeddings(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        新增文字 embeddings 到 collection
        
        Args:
            collection_name: collection 名稱
            texts: 文字內容列表
            embeddings: embedding 向量列表
            metadatas: metadata 列表
            ids: 自訂 ID 列表（可選）
        """
        collection = self.get_or_create_collection(collection_name)
        
        # 如果沒有提供 IDs，自動生成
        if ids is None:
            existing_count = collection.count()
            ids = [f"text_{existing_count + i}" for i in range(len(texts))]
        
        # 批次新增
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Added {len(texts)} text embeddings to {collection_name}")
    
    def add_image_embeddings(
        self,
        collection_name: str,
        image_descriptions: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        新增圖片 embeddings 到 collection
        
        Args:
            collection_name: collection 名稱
            image_descriptions: 圖片描述列表
            embeddings: embedding 向量列表
            metadatas: metadata 列表（包含 img_path）
            ids: 自訂 ID 列表（可選）
        """
        collection = self.get_or_create_collection(collection_name, embedding_dimension=3072)
        
        # 如果沒有提供 IDs，自動生成
        if ids is None:
            existing_count = collection.count()
            ids = [f"img_{existing_count + i}" for i in range(len(image_descriptions))]
        
        # 批次新增
        collection.add(
            documents=image_descriptions,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Added {len(image_descriptions)} image embeddings to {collection_name}")
    
    def search_similar(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        搜尋相似的 embeddings
        
        Args:
            collection_name: collection 名稱
            query_embedding: 查詢的 embedding 向量
            top_k: 返回前 k 個結果
            filter_dict: 過濾條件（可選）
            
        Returns:
            搜尋結果字典
        """
        collection = self.get_or_create_collection(collection_name)
        
        # 搜尋
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        return results
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        檢查 collection 是否存在
        
        Args:
            collection_name: collection 名稱
            
        Returns:
            是否存在
        """
        try:
            self.client.get_collection(name=collection_name)
            return True
        except:
            return False
    
    def get_collection_count(self, collection_name: str) -> int:
        """
        取得 collection 中的項目數量
        
        Args:
            collection_name: collection 名稱
            
        Returns:
            項目數量
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            return collection.count()
        except:
            return 0
    
    def delete_collection(self, collection_name: str) -> None:
        """
        刪除 collection
        
        Args:
            collection_name: collection 名稱
        """
        try:
            self.client.delete_collection(name=collection_name)
            print(f"✓ Deleted collection: {collection_name}")
        except Exception as e:
            print(f"✗ Failed to delete collection {collection_name}: {e}")
    
    def list_collections(self) -> List[str]:
        """
        列出所有 collections
        
        Returns:
            collection 名稱列表
        """
        collections = self.client.list_collections()
        return [col.name for col in collections]


def build_vector_db_from_dataframes(
    text_df,
    image_df,
    vector_db: VectorDB,
    text_collection_name: str = "text_embeddings",
    image_collection_name: str = "image_embeddings",
    force_rebuild: bool = False
) -> Tuple[bool, bool]:
    """
    從 DataFrame 建立向量資料庫
    
    Args:
        text_df: 文字 metadata DataFrame
        image_df: 圖片 metadata DataFrame
        vector_db: VectorDB 實例
        text_collection_name: 文字 collection 名稱
        image_collection_name: 圖片 collection 名稱
        force_rebuild: 是否強制重建
        
    Returns:
        (text_built, image_built) 是否建立了新的 collections
    """
    text_built = False
    image_built = False
    
    # 檢查並建立 text embeddings
    if force_rebuild or not vector_db.collection_exists(text_collection_name):
        print(f"\n📝 Building text embeddings collection...")
        
        # 準備資料
        texts = text_df['chunk_text'].tolist()
        embeddings = text_df['text_embedding_chunk'].tolist()
        
        metadatas = []
        ids = []
        for idx, row in text_df.iterrows():
            metadata = {
                'file_name': row['file_name'],
                'page_num': int(row['page_num']),
                'chunk_number': int(row['chunk_number']),
                'type': 'text'
            }
            metadatas.append(metadata)
            ids.append(f"text_{row['file_name']}_{row['page_num']}_{row['chunk_number']}")
        
        # 如果強制重建，先刪除舊的
        if force_rebuild and vector_db.collection_exists(text_collection_name):
            vector_db.delete_collection(text_collection_name)
        
        # 新增到資料庫
        vector_db.add_text_embeddings(
            collection_name=text_collection_name,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        text_built = True
    else:
        print(f"✓ Text collection already exists (count: {vector_db.get_collection_count(text_collection_name)})")
    
    # 檢查並建立 image embeddings
    if force_rebuild or not vector_db.collection_exists(image_collection_name):
        print(f"\n🖼️  Building image embeddings collection...")
        
        # 準備資料
        descriptions = image_df['img_desc'].tolist()
        embeddings = image_df['mm_embedding_from_img_only'].tolist()
        
        metadatas = []
        ids = []
        for idx, row in image_df.iterrows():
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
        
        # 如果強制重建，先刪除舊的
        if force_rebuild and vector_db.collection_exists(image_collection_name):
            vector_db.delete_collection(image_collection_name)
        
        # 新增到資料庫
        vector_db.add_image_embeddings(
            collection_name=image_collection_name,
            image_descriptions=descriptions,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        image_built = True
    else:
        print(f"✓ Image collection already exists (count: {vector_db.get_collection_count(image_collection_name)})")
    
    return text_built, image_built
