"""
NCF CÓ THỂ HOẠT ĐỘNG Ở 3 MODES:

Mode 1: STANDALONE NCF
Input:  user_id, product_id
Output: score [0, 1]
→ Dùng internal embeddings học từ interactions

Mode 2: NCF + USER EMBEDDING MODEL (HYBRID)
Input:  user_id + user_embedding_vector, product_id  
Output: score [0, 1]
→ Kết hợp NCF embeddings + pre-trained user vectors

Mode 3: NCF AS COMPATIBILITY PREDICTOR
Input:  user_embedding_vector, product_embedding_vector
Output: compatibility_score [0, 1]
→ NCF trở thành scoring function cho embeddings
"""



NCF nhận gì:
python# Mode 2 (RECOMMENDED)
Input = {
    'user_id': int,
    'user_embedding': np.array (128-dim),  # From user model
    'product_id': int,
    'product_embedding': np.array (768-dim)  # From MGTE/Milvus
}
NCF trả về gì:
pythonOutput = {
    'compatibility_score': float,  # [0, 1]
    'explanation': {
        'user_id_contribution': float,
        'user_features_contribution': float,
        'product_id_contribution': float,
        'product_features_contribution': float
    }
}