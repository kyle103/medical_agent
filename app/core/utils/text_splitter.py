from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

class TextSplitter:
    """
    文本分割工具类，用于将长文本分割为适合向量存储的短文本片段
    """
    
    @staticmethod
    def split_text(
        text: str, 
        chunk_size: int = 500, 
        chunk_overlap: int = 100,
        separators: List[str] = None
    ) -> List[str]:
        """
        分割文本为多个片段
        
        Args:
            text: 要分割的文本
            chunk_size: 每个片段的最大长度
            chunk_overlap: 片段之间的重叠长度
            separators: 分割符列表，默认为中文标点符号
        
        Returns:
            分割后的文本片段列表
        """
        if not text:
            return []
        
        # 默认分割符，优先使用语义分割
        if separators is None:
            separators = [
                "\n\n",  # 段落
                "\n",    # 换行
                "。",    # 句号
                "！",    # 感叹号
                "？",    # 问号
                "，",    # 逗号
                "、"     # 顿号
            ]
        
        # 使用RecursiveCharacterTextSplitter进行分割
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
        
        return splitter.split_text(text)
    
    @staticmethod
    def split_text_with_metadata(
        text: str, 
        metadata: Dict[str, Any],
        chunk_size: int = 500, 
        chunk_overlap: int = 100,
        separators: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        分割文本并为每个片段添加元数据
        
        Args:
            text: 要分割的文本
            metadata: 原始文本的元数据
            chunk_size: 每个片段的最大长度
            chunk_overlap: 片段之间的重叠长度
            separators: 分割符列表，默认为中文标点符号
        
        Returns:
            包含文本片段和元数据的字典列表
        """
        chunks = TextSplitter.split_text(text, chunk_size, chunk_overlap, separators)
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            result.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        return result