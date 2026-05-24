"""
Giao dien Gradio - He thong AI Phat hien Dao van
1 o nhap van ban -> So sanh voi dataset VISP -> Hien thi ket qua
Chay: python app.py
"""
import gradio as gr
import os
from plagiarism_detector import PlagiarismDetector

print("[*] Dang khoi tao he thong...")
detector = PlagiarismDetector(use_embedding=True, dataset_path='visp_train.xlsx', max_docs=3000)

SAMPLE1 = "Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính tập trung vào việc tạo ra các hệ thống có khả năng thực hiện các nhiệm vụ đòi hỏi trí thông minh của con người."
SAMPLE2 = "Để chủ động phòng bệnh sán dây và ấu trùng sán lợn, người dân cần ăn uống các thức ăn đã được nấu chín kỹ, chế biến hợp vệ sinh."
SAMPLE3 = "Việc ăn uống các thức ăn đã được nấu chín kỹ và chế biến hợp vệ sinh giúp phòng tránh bệnh sán dây và ấu trùng sán lợn."


def check_plagiarism(text, method):
    if not text or not text.strip():
        return "⚠️ Vui lòng nhập văn bản cần kiểm tra!", "", ""

    method_map = {
        "TF-IDF (So sánh từ khóa)": "tfidf",
        "Embedding (So sánh ngữ nghĩa)": "embedding",
        "Hybrid (Kết hợp cả hai)": "hybrid"
    }
    method_key = method_map.get(method, "hybrid")

    # Tim kiem trong dataset
    search_results = detector.search_dataset(text, top_k=10, threshold=0.1)

    if not search_results:
        main = """## 🟢 Kết quả: KHÔNG PHÁT HIỆN ĐẠO VĂN
Không tìm thấy văn bản tương đồng trong cơ sở dữ liệu.
"""
        return main, "Không có đoạn văn trùng lặp.", "0.0%"

    # Tinh chi tiet voi ket qua tot nhat
    best = search_results[0]
    result = detector.detect(text, best['text'], method=method_key)
    score = result['similarity_score']
    pct = score * 100
    level = result['level']

    if level == 'high':
        badge = "🔴 ĐẠO VĂN CAO"
    elif level == 'medium':
        badge = "🟡 NGHI NGỜ TRÙNG LẶP"
    else:
        badge = "🟢 KHÔNG ĐẠO VĂN"

    main = f"""## Kết quả: {badge}
### 📊 Độ tương đồng cao nhất: **{pct:.1f}%**
| Phương pháp | Điểm |
|---|---|
| TF-IDF | {result['tfidf_score']*100:.1f}% |
| Embedding | {result['embedding_score']*100:.1f}% |
| **Tổng hợp** | **{pct:.1f}%** |

⏱️ Thời gian xử lý: **{result['processing_time']:.3f}s**

### 📋 Phân loại mức độ:
| Mức độ | Khoảng | Trạng thái |
|---|---|---|
| 🟢 Thấp | 0 - 30% | {"✅ **Hiện tại**" if level=="low" else ""} |
| 🟡 Trung bình | 30 - 60% | {"⚠️ **Hiện tại**" if level=="medium" else ""} |
| 🔴 Cao | > 60% | {"🔴 **Hiện tại**" if level=="high" else ""} |
"""

    # Hien thi cac van ban trung lap tu dataset
    matched_text = "## 🔍 Các văn bản tương đồng trong cơ sở dữ liệu\n\n"
    for i, r in enumerate(search_results[:5], 1):
        s = r['score'] * 100
        if s < 5:
            continue
        if s >= 60:
            icon = "🔴"
        elif s >= 30:
            icon = "🟡"
        else:
            icon = "🟢"
        matched_text += f"### {icon} Kết quả {i} — Tương đồng: **{s:.1f}%**\n"
        matched_text += f"- **Nguồn**: {r['source']} | **Chủ đề**: {r['topic']}\n"
        matched_text += f"- **Nội dung**: *\"{r['text'][:300]}\"*\n\n"

    # Hien thi cac cau trung lap cu the
    if result['matched_sentences']:
        matched_text += "---\n### 📌 Chi tiết các câu trùng lặp:\n\n"
        for j, m in enumerate(result['matched_sentences'][:5], 1):
            matched_text += f"**Cặp {j}** (Tương đồng: {m['score']*100:.1f}%)\n"
            matched_text += f"- 📝 Văn bản nhập: *\"{m['sent1'][:200]}\"*\n"
            matched_text += f"- 📂 Trong CSDL: *\"{m['sent2'][:200]}\"*\n\n"

    return main, matched_text, f"{pct:.1f}%"


def check_from_file(file_obj, method):
    if file_obj is None:
        return "⚠️ Vui lòng tải lên file!", "", ""
    text = read_file(file_obj)
    if not text:
        return "❌ Không thể đọc file! Hỗ trợ: .txt, .docx, .pdf", "", ""
    return check_plagiarism(text, method)


def read_file(file_obj):
    if file_obj is None: return ""
    fp = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    ext = os.path.splitext(fp)[1].lower()
    try:
        if ext == '.txt':
            return open(fp, 'r', encoding='utf-8').read()
        elif ext == '.docx':
            from docx import Document
            return '\n'.join([p.text for p in Document(fp).paragraphs])
        elif ext == '.pdf':
            from PyPDF2 import PdfReader
            return '\n'.join([pg.extract_text() or '' for pg in PdfReader(fp).pages])
        else:
            return open(fp, 'r', encoding='utf-8').read()
    except:
        return ""


def run_evaluation(n, thresh):
    try:
        from evaluator import load_visp_dataset, create_evaluation_dataset, evaluate_model
        n = int(n); half = n // 2
        df = load_visp_dataset('visp_test.xlsx', max_samples=n*5)
        data = create_evaluation_dataset(df, n_positive=half, n_negative=half)
        r = evaluate_model(detector, data, method='tfidf', threshold=thresh, verbose=False)
        return f"""## 📊 Kết quả đánh giá — TF-IDF
| Chỉ số | Giá trị |
|---|---|
| **Độ chính xác (Accuracy)** | {r['accuracy']*100:.1f}% |
| **Precision** | {r['precision']*100:.1f}% |
| **Recall** | {r['recall']*100:.1f}% |
| **F1-Score** | {r['f1_score']*100:.1f}% |
| Thời gian | {r['total_time']:.2f}s |
| Số mẫu | {len(data)} |
| Ngưỡng | {thresh} |

### Ma trận nhầm lẫn (Confusion Matrix)
```
{r['confusion_matrix']}
```"""
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"


# === GIAO DIEN ===
with gr.Blocks() as app:
    gr.HTML("""<div style='text-align:center;padding:20px'>
        <h1>🔍 Hệ Thống AI Phát Hiện Đạo Văn</h1>
        <p style='color:#666'>Sử dụng TF-IDF & Sentence Embedding | Dataset: VISP (Vietnamese Paraphrase)</p>
        <p style='color:#888;font-size:0.85em'>Môn: Lập Trình Trí Tuệ Nhân Tạo</p>
    </div>""")

    with gr.Tabs():
        with gr.TabItem("📝 Kiểm tra đạo văn"):
            text_input = gr.Textbox(
                label="📄 Nhập văn bản cần kiểm tra",
                placeholder="Dán hoặc nhập văn bản cần kiểm tra đạo văn tại đây...",
                lines=8, value=SAMPLE3
            )
            with gr.Row():
                method = gr.Dropdown(
                    ["TF-IDF (So sánh từ khóa)", "Embedding (So sánh ngữ nghĩa)", "Hybrid (Kết hợp cả hai)"],
                    value="Hybrid (Kết hợp cả hai)", label="🧠 Phương pháp phân tích"
                )
                score_box = gr.Textbox(label="📊 Điểm tương đồng", interactive=False)
            btn = gr.Button("🔍 Kiểm Tra Đạo Văn", variant="primary", size="lg")
            gr.Markdown("**Ví dụ mẫu:**")
            with gr.Row():
                ex1 = gr.Button("📌 Mẫu 1: Có trong CSDL", size="sm")
                ex2 = gr.Button("📌 Mẫu 2: Diễn đạt lại", size="sm")
                ex3 = gr.Button("📌 Mẫu 3: Nội dung mới", size="sm")
            res = gr.Markdown(label="Kết quả")
            match_res = gr.Markdown(label="Văn bản trùng lặp trong CSDL")

            btn.click(fn=check_plagiarism, inputs=[text_input, method], outputs=[res, match_res, score_box])
            ex1.click(fn=lambda: SAMPLE3, outputs=[text_input])
            ex2.click(fn=lambda: SAMPLE2, outputs=[text_input])
            ex3.click(fn=lambda: SAMPLE1, outputs=[text_input])

        with gr.TabItem("📁 Tải file lên"):
            f1 = gr.File(label="📄 Chọn file (.txt, .docx, .pdf)", file_types=[".txt",".docx",".pdf"])
            method2 = gr.Dropdown(
                ["TF-IDF (So sánh từ khóa)", "Embedding (So sánh ngữ nghĩa)", "Hybrid (Kết hợp cả hai)"],
                value="Hybrid (Kết hợp cả hai)", label="🧠 Phương pháp"
            )
            btn2 = gr.Button("🔍 Kiểm tra đạo văn từ file", variant="primary", size="lg")
            fr = gr.Markdown(); fm = gr.Markdown(); fs = gr.Textbox(visible=False)
            btn2.click(fn=check_from_file, inputs=[f1, method2], outputs=[fr, fm, fs])

        with gr.TabItem("📊 Đánh giá Model"):
            gr.Markdown("""## 📊 Đánh giá hiệu năng hệ thống
Chạy đánh giá trên dataset VISP để xem các chỉ số:
- **Accuracy** — Độ chính xác tổng thể
- **Precision** — Tỷ lệ phát hiện đúng đạo văn
- **Recall** — Khả năng phát hiện tất cả trường hợp đạo văn
- **F1-Score** — Cân bằng giữa Precision & Recall
""")
            with gr.Row():
                ns = gr.Slider(50, 500, value=200, step=50, label="Số mẫu đánh giá")
                et = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Ngưỡng phân loại")
            eb = gr.Button("🚀 Chạy đánh giá", variant="primary")
            eo = gr.Markdown()
            eb.click(fn=run_evaluation, inputs=[ns,et], outputs=[eo])

        with gr.TabItem("ℹ️ Hướng dẫn"):
            gr.Markdown("""## 📖 Hướng dẫn sử dụng

### 🎯 Cách hoạt động
1. Nhập văn bản cần kiểm tra vào ô nhập liệu
2. Chọn phương pháp phân tích
3. Nhấn **"Kiểm Tra Đạo Văn"**
4. Hệ thống sẽ tìm kiếm trong **cơ sở dữ liệu VISP** (~3000 văn bản)
5. Hiển thị các văn bản tương đồng và mức độ đạo văn

### 🧠 Các phương pháp phân tích
| Phương pháp | Mô tả | Ưu điểm |
|---|---|---|
| **TF-IDF** | So sánh dựa trên tần suất từ khóa | Nhanh, tốt với sao chép nguyên văn |
| **Embedding** | So sánh ngữ nghĩa bằng Sentence Transformer | Phát hiện được diễn đạt lại |
| **Hybrid** | Kết hợp (40% TF-IDF + 60% Embedding) | Chính xác nhất |

### 📊 Mức độ đạo văn
| Mức | Khoảng | Ý nghĩa |
|---|---|---|
| 🟢 Thấp | 0 - 30% | Không đạo văn |
| 🟡 Trung bình | 30 - 60% | Nghi ngờ trùng lặp |
| 🔴 Cao | > 60% | Đạo văn cao |

### 📂 Dataset VISP
- Nguồn: ViNewsQA, ViNLI, ViQuAD, ALQAC
- Chủ đề: Sức khỏe, Chính trị, Văn hóa, Khoa học, Thể thao...

### 🛠️ Công nghệ
Python, scikit-learn, sentence-transformers, NLTK, Gradio, Pandas, NumPy
""")

    gr.HTML("""<div style='text-align:center;padding:20px;color:#888;border-top:1px solid #eee;margin-top:20px'>
        <p>🎓 Mini Project — Môn Lập Trình Trí Tuệ Nhân Tạo</p>
        <p>Công nghệ: Python | TF-IDF | Sentence Transformer | Cosine Similarity | Gradio</p>
    </div>""")


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
