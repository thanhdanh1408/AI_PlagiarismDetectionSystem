"""
Giao dien Gradio - He thong AI Phat hien Dao van
Chay: python app.py
"""
import gradio as gr
import os
from plagiarism_detector import PlagiarismDetector

print("[*] Dang khoi tao he thong...")
# Khởi tạo model (Lưu ý: Nếu máy RAM yếu, bạn có thể giảm max_docs xuống 1000 hoặc 2000)
detector = PlagiarismDetector(use_embedding=True, dataset_path='visp_train.xlsx', max_docs=3000)

SAMPLE1 = "Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính tập trung vào việc tạo ra các hệ thống có khả năng thực hiện các nhiệm vụ đòi hỏi trí thông minh của con người."
SAMPLE2 = "Để chủ động phòng bệnh sán dây và ấu trùng sán lợn, người dân cần ăn uống các thức ăn đã được nấu chín kỹ, chế biến hợp vệ sinh."
SAMPLE3 = "Việc ăn uống các thức ăn đã được nấu chín kỹ và chế biến hợp vệ sinh giúp phòng tránh bệnh sán dây và ấu trùng sán lợn."

def highlight_text(original_text, matched_sentences):
    """Thuật toán dò tìm và bôi màu các câu trùng lặp (Đã tối ưu lỗi đè khoảng trắng)"""
    if not original_text or not matched_sentences:
        return [(original_text, "🟢 Hợp lệ")]
    
    matches = []
    for m in matched_sentences:
        s1 = m['sent1']
        score = m['score']
        
        # Gán nhãn dựa trên điểm số
        if score >= 0.6: 
            label = "🔴 Đạo văn"
        elif score >= 0.3: 
            label = "🟡 Nghi ngờ"
        else: 
            continue
        
        # Tìm vị trí câu trong đoạn văn gốc
        start_idx = original_text.find(s1)
        if start_idx != -1:
            matches.append((start_idx, start_idx + len(s1), label))
            
    if not matches:
        return [(original_text, "🟢 Hợp lệ")]
        
    # Sắp xếp theo vị trí xuất hiện từ trên xuống dưới
    matches.sort(key=lambda x: x[0])
    
    # Gom cụm các đoạn bôi màu bị đè lên nhau
    resolved = []
    last_end = 0
    for start, end, label in matches:
        if not resolved:
            resolved.append((start, end, label))
            last_end = end
            continue
            
        last_start, last_e, last_label = resolved[-1]
        if start >= last_e:
            resolved.append((start, end, label))
            last_end = end
        elif end > last_e:
            # Ưu tiên nhãn "Đạo văn" nếu có sự chồng chéo
            new_label = "🔴 Đạo văn" if "Đạo văn" in [label, last_label] else label
            resolved[-1] = (last_start, end, new_label)
            last_end = end
            
    # Phân mảnh text để Gradio HighlightedText có thể render
    result = []
    curr = 0
    for start, end, label in resolved:
        if start > curr:
            result.append((original_text[curr:start], None))
        result.append((original_text[start:end], label))
        curr = end
        
    if curr < len(original_text):
        result.append((original_text[curr:], None))
        
    return result

def check_plagiarism(text, method):
    """Hàm xử lý chính (Trả về 4 tham số cho UI)"""
    if not text or not text.strip():
        return "⚠️ Vui lòng nhập văn bản cần kiểm tra!", "", "", [("Vui lòng nhập văn bản", None)]

    method_map = {
        "TF-IDF (So sánh từ khóa)": "tfidf",
        "Embedding (So sánh ngữ nghĩa)": "embedding",
        "Hybrid (Kết hợp cả hai)": "hybrid"
    }
    method_key = method_map.get(method, "hybrid")

    search_results = detector.search_dataset(text, top_k=10, threshold=0.1)

    if not search_results:
        main = """## 🟢 Kết quả: KHÔNG PHÁT HIỆN ĐẠO VĂN
Không tìm thấy văn bản tương đồng trong cơ sở dữ liệu."""
        return main, "Không có đoạn văn trùng lặp.", "0.0%", [(text, "🟢 Hợp lệ")]

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
### 📊 Độ tương đồng tổng quát: **{pct:.1f}%**
| Phương pháp | Điểm |
|---|---|
| TF-IDF | {result['tfidf_score']*100:.1f}% |
| Embedding | {result['embedding_score']*100:.1f}% |
| **Tổng hợp** | **{pct:.1f}%** |

⏱️ Thời gian xử lý: **{result['processing_time']:.3f}s**
"""

    matched_text = "## 🔍 Nguồn dữ liệu trùng lặp trong CSDL\n\n"
    for i, r in enumerate(search_results[:5], 1):
        s = r['score'] * 100
        if s < 5: continue
        icon = "🔴" if s >= 60 else "🟡" if s >= 30 else "🟢"
        matched_text += f"### {icon} Kết quả {i} — Tương đồng: **{s:.1f}%**\n"
        matched_text += f"- **Nguồn**: {r['source']} | **Chủ đề**: {r['topic']}\n"
        matched_text += f"- **Nội dung gốc**: *\"{r['text'][:300]}...\"*\n\n"

    # Xử lý bôi màu
    highlight_data = highlight_text(text, result['matched_sentences'])

    return main, matched_text, f"{pct:.1f}%", highlight_data

def check_from_file(file_obj, method):
    if file_obj is None:
        return "⚠️ Vui lòng tải lên file!", "", "", [("Chưa có file", None)]
    text = read_file(file_obj)
    if not text:
        return "❌ Không thể đọc file! Hỗ trợ: .txt, .docx, .pdf", "", "", [("Lỗi đọc file", None)]
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
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return ""

def run_evaluation(n, thresh):
    try:
        from evaluator import load_visp_dataset, create_evaluation_dataset, evaluate_model
        n = int(n); half = n // 2
        df = load_visp_dataset('visp_test.xlsx', max_samples=n*5)
        data = create_evaluation_dataset(df, n_positive=half, n_negative=half)
        r = evaluate_model(detector, data, method='tfidf', threshold=thresh, verbose=False)
        return f"""## 📊 Kết quả đánh giá hệ thống
| Chỉ số | Giá trị |
|---|---|
| **Độ chính xác (Accuracy)** | {r['accuracy']*100:.1f}% |
| **Precision** | {r['precision']*100:.1f}% |
| **Recall** | {r['recall']*100:.1f}% |
| **F1-Score** | {r['f1_score']*100:.1f}% |
| Thời gian chạy | {r['total_time']:.2f}s |
| Số mẫu test | {len(data)} |
| Ngưỡng (Threshold) | {thresh} |

### Ma trận nhầm lẫn (Confusion Matrix)
```text
{r['confusion_matrix']}
```"""
    except Exception as e:
        return f"❌ Lỗi khi chạy đánh giá: {str(e)}"

# ==========================================
# GIAO DIỆN CHÍNH (GUI)
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.HTML("""<div style='text-align:center;padding:15px'>
        <h1 style='color:#2c3e50;'>🔍 Hệ Thống AI Phát Hiện Đạo Văn</h1>
        <p style='color:#7f8c8d; font-size: 1.1em;'>Sử dụng kỹ thuật NLP Hybrid (TF-IDF & Sentence Embedding)</p>
    </div>""")

    with gr.Tabs():
        # --- TAB 1: NHẬP VĂN BẢN TRỰC TIẾP ---
        with gr.TabItem("📝 Kiểm tra trực tiếp"):
            text_input = gr.Textbox(
                label="📄 Nhập đoạn văn bản của bạn",
                placeholder="Dán nội dung cần kiểm tra vào đây...",
                lines=6, value=SAMPLE3
            )
            with gr.Row():
                method = gr.Dropdown(
                    ["TF-IDF (So sánh từ khóa)", "Embedding (So sánh ngữ nghĩa)", "Hybrid (Kết hợp cả hai)"],
                    value="Hybrid (Kết hợp cả hai)", label="🧠 Thuật toán"
                )
                score_box = gr.Textbox(label="📊 Điểm đạo văn tổng thể", interactive=False)
                
            btn = gr.Button("🔍 PHÂN TÍCH", variant="primary", size="lg")
            
            gr.Markdown("**Dữ liệu mẫu test nhanh:**")
            with gr.Row():
                ex1 = gr.Button("📌 Trùng lặp nguyên văn", size="sm")
                ex2 = gr.Button("📌 Diễn đạt lại (Paraphrase)", size="sm")
                ex3 = gr.Button("📌 Hoàn toàn mới", size="sm")
                
            # Khung kết quả chia làm 2 cột
            res = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1):
                    hl_box = gr.HighlightedText(
                        label="📝 Phân tích câu (Vùng bôi màu là đạo văn)",
                        color_map={"🔴 Đạo văn": "#ffcccc", "🟡 Nghi ngờ": "#fff2cc", "🟢 Hợp lệ": "#e6ffe6"},
                        combine_adjacent=True
                    )
                with gr.Column(scale=1):
                    match_res = gr.Markdown()

            # Mapping Event
            btn.click(fn=check_plagiarism, inputs=[text_input, method], outputs=[res, match_res, score_box, hl_box])
            ex1.click(fn=lambda: SAMPLE3, outputs=[text_input])
            ex2.click(fn=lambda: SAMPLE2, outputs=[text_input])
            ex3.click(fn=lambda: SAMPLE1, outputs=[text_input])

        # --- TAB 2: TẢI FILE LÊN ---
        with gr.TabItem("📁 Tải file tài liệu"):
            f1 = gr.File(label="📄 Upload file báo cáo (.txt, .docx, .pdf)", file_types=[".txt",".docx",".pdf"])
            method2 = gr.Dropdown(
                ["TF-IDF (So sánh từ khóa)", "Embedding (So sánh ngữ nghĩa)", "Hybrid (Kết hợp cả hai)"],
                value="Hybrid (Kết hợp cả hai)", label="🧠 Thuật toán"
            )
            btn2 = gr.Button("🔍 KIỂM TRA FILE", variant="primary", size="lg")
            
            # Khung kết quả chia làm 2 cột
            fr = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1):
                    fh = gr.HighlightedText(
                        label="📝 Quét nội dung File",
                        color_map={"🔴 Đạo văn": "#ffcccc", "🟡 Nghi ngờ": "#fff2cc", "🟢 Hợp lệ": "#e6ffe6"},
                        combine_adjacent=True
                    )
                with gr.Column(scale=1):
                    fm = gr.Markdown()
            fs = gr.Textbox(visible=False) # Ẩn đi vì không cần thiết hiển thị lại ở dạng ô text độc lập
            
            btn2.click(fn=check_from_file, inputs=[f1, method2], outputs=[fr, fm, fs, fh])

        # --- TAB 3: ĐÁNH GIÁ MODEL ---
        with gr.TabItem("📊 Đánh giá Model (Testing)"):
            gr.Markdown("Chạy quá trình Testing hệ thống trên tập dữ liệu `visp_test.xlsx`.")
            with gr.Row():
                ns = gr.Slider(50, 500, value=100, step=50, label="Số lượng mẫu đánh giá")
                et = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Ngưỡng phân loại đạo văn")
            eb = gr.Button("🚀 CHẠY KIỂM THỬ (EVALUATE)", variant="primary")
            eo = gr.Markdown()
            eb.click(fn=run_evaluation, inputs=[ns,et], outputs=[eo])

        # --- TAB 4: HƯỚNG DẪN ---
        with gr.TabItem("ℹ️ Thông tin & Hướng dẫn"):
            gr.Markdown("""## 📖 Hướng dẫn sử dụng

### 🎯 Quy trình hoạt động
1. Nhập văn bản hoặc tải file (.docx, .pdf, .txt) cần kiểm tra.
2. Lựa chọn thuật toán phân tích.
3. Bấm **"Phân tích"** để AI quét dữ liệu.
4. Xem kết quả các đoạn văn bản bị bôi đỏ và đối chiếu với bản gốc bên cột phải.

### 🧠 Giải thích thuật toán
| Phương pháp | Đặc điểm |
|---|---|
| **TF-IDF** | Tốc độ cực nhanh. Hoạt động dựa trên tần suất từ khóa. Rất giỏi bắt lỗi "Copy - Paste" y nguyên. |
| **Embedding** | Sử dụng Deep Learning (Sentence Transformer). Giỏi phát hiện "Paraphrase" (Đổi từ đồng nghĩa, xáo trộn câu). |
| **Hybrid** | Kết hợp (40% điểm TF-IDF + 60% điểm Embedding). Là thuật toán tối ưu và toàn diện nhất của hệ thống. |

### 📂 Nguồn dữ liệu (Dataset)
Sử dụng bộ dữ liệu Tiếng Việt **VISP (Vietnamese Paraphrase)** thu thập từ các nguồn báo chí và học thuật, bao quát nhiều chủ đề (Sức khỏe, Khoa học, Kinh tế...).
""")

    gr.HTML("""<div style='text-align:center;padding:20px;color:#95a5a6;border-top:1px solid #ecf0f1;margin-top:20px'>
        <p>🎓 Đồ án môn Lập Trình Trí Tuệ Nhân Tạo</p>
        <p>Công nghệ: Python | NLP | TF-IDF | Sentence Transformer | Gradio</p>
    </div>""")

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)