### 🧠 Mục tiêu

Kết hợp **BGEM3 embeddings (content)** và **Collaborative Filtering embeddings (CF)** để tạo hệ thống **recommendation vừa hiểu ngữ nghĩa (semantic)**, vừa dựa trên **hành vi người dùng (behavioral)**, đồng thời tận dụng **Milvus** để tìm kiếm nhanh theo vector.

---

### ⚙️ Huấn luyện mô hình CF

Sử dụng mô hình **Matrix Factorization** đơn giản trong TensorFlow:

```python
user_vector = Embedding(num_users, dim)
item_vector = Embedding(num_items, dim)
score = sigmoid(dot(user_vector, item_vector))
```

Kết quả của quá trình huấn luyện là hai embedding:

* `user_cf_embeddings`: vector hành vi người dùng
* `item_cf_embeddings`: vector đặc trưng sản phẩm theo khía cạnh hành vi

---

### 🧩 Cấu trúc Milvus

Hệ thống có ba collection chính:

1. **content_products** – chứa BGEM3 embeddings để phục vụ tìm kiếm theo ngữ nghĩa (vd: tìm theo text “Apple”).
   Trường chính: `id`, `name`, `dense_vector (768d)`

2. **hybrid_products** – chứa fused embeddings (kết hợp giữa BGEM3 và CF) để phục vụ recommendation.
   Trường chính: `id`, `metadata`, `dense_vector (768d fused)`

3. **hybrid_customers** – lưu vector biểu diễn hành vi người dùng (kết hợp CF và nội dung gần đây).
   Trường chính: `id`, `dense_vector (768d fused)`

---

### 🧮 Cách tạo fused embedding

Kết hợp BGEM3 và CF embedding rồi đưa về cùng kích thước (768d):

```python
concat = torch.cat([item_bgem3, item_cf], dim=-1)
item_fused = Linear(concat, out_dim=768)
```

Công thức tương tự áp dụng cho user embedding:
`user_fused = Linear(cat([user_recent_bgem3_avg, user_cf]), out_dim=768)`

---

### 💾 Lưu vào Milvus

* Collection `content_products`: dùng cho **semantic search**
* Collection `hybrid_products`: dùng cho **recommendation theo user**
* Collection `hybrid_customers`: chứa vector user để làm query

---

### 🔍 Query logic

* Khi user tìm kiếm text như “Apple”:
  → Encode text bằng BGEM3
  → Query trong `content_products`
  → Trả về kết quả theo ngữ nghĩa nội dung

* Khi hệ thống cần gợi ý sản phẩm cho user:
  → Lấy `user_fused` embedding (CF + content)
  → Query trong `hybrid_products`
  → Trả về top sản phẩm phù hợp nhất với hành vi và sở thích user

---

Kết quả là một hệ thống recommendation lai (hybrid) — **vừa hiểu ngữ nghĩa**, **vừa hiểu hành vi**, **search nhanh**, **dễ mở rộng** khi có dữ liệu hoặc người dùng mới.
