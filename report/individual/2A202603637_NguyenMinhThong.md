# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603637
- **Họ và tên:** Nguyễn Minh Thông
- **SEED** (in ở ô đầu notebook): 2341

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.878 | 0.919 | 0.119 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.955 | 0.974 | 0.540 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.900 | 0.976 | 16.379 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.952 | 0.947 | 12.475 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.967 | 0.974 | 11.523 | 0.000 |
| 4a. LLM nhỏ zero-shot (`gpt-6-luna`) | — | 0.950 | 5560.072 | 0.006 |
| 4b. LLM lớn zero-shot (`gpt-6.1-terra`) | — | 0.947 | 1374.965 | 0.063 |
| 4c. LLM nhỏ few-shot | — | 0.950 | 1179.041 | 0.022 |

Hàng LLM chỉ đo trên SUB (40 tin), nên so chất lượng giữa các cách bằng cột F1 SUB. Trên SUB, 1 tin sai làm F1 đổi khoảng 0.025, nên chênh dưới ~0.05 coi như hoà.

## 3. Dự đoán và thực tế

1. **`hard_eval` (50 case khó):** em đoán mọi model tụt mạnh, LLM chỉ còn ~75–85%, và nhóm khó nhất với LLM là prompt injection. Thực tế LLM nhỏ đạt 98%, bản có prompt phòng thủ đạt 100%, và đúng 100% ở nhóm prompt injection. Lệch vì em đánh giá thấp khả năng đọc hiểu ý định của LLM: các bẫy đánh vào từ khoá (né từ khoá, giọng tự nhiên) làm khó BoW chứ không làm khó một model đọc cả câu như người. Phần đoán cho BoW (~50–60%) thì khớp (50–54%).
2. **`dl_curve` / `dl_relu` (MLP so với LogReg):** em đoán MLP sẽ ngang LogReg, và bỏ ReLU thì mạng tương đương LogReg. Thực tế MLP đạt F1 0.955 so với 0.878, và bỏ ReLU vẫn đạt 0.955. Về mặt toán thì mạng không ReLU đúng là tuyến tính như LogReg, nhưng chênh lệch đến từ cách train: Adam + mini-batch train tới loss ~0, còn LogReg tự code chạy 300 epoch gradient descent vẫn chưa hội tụ (F1 trên train chỉ 0.917).
3. **`llm_models` (model lớn vs model nhỏ):** em đoán terra cao hơn một chút, đắt hơn ~10 lần và chậm hơn nhiều. Thực tế hai model hoà (0.947 so với 0.950, cùng sai 2/40 tin), và terra lại nhanh hơn (p50 1166 ms so với 3664 ms của luna). Chỉ phần giá là khớp (≈10 lần). Lệch vì với bài phân loại 1 từ đơn giản, model nhỏ đã đủ giỏi, còn độ trễ API phụ thuộc tải server lúc gọi hơn là kích thước model. Thêm nữa, kết quả LLM dao động giữa các lần chạy dù temperature=0 (F1 zero-shot của model nhỏ từng chỉ đạt 0.878), nên chênh 1–2 tin trên SUB 40 tin chưa đủ để kết luận.

## 4. Phân tích 50 case khó

- **Tụt nhiều nhất: BoW + LogReg**, F1 từ 0.919 (SUB) xuống 0.286, accuracy 50%. BoW + MLP cũng chỉ còn 54%. Embedding còn 64–68%, LLM 98–100%.
- **Nhóm làm BoW sai nhiều nhất:** spam giọng tự nhiên (BoW + LogReg đúng 0%), thao túng/lừa đảo (12%), né từ khoá (17%).
- **Case 1 (spam giọng tự nhiên):** *"Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed, just text me back"*. Tin không có từ khoá spam mạnh (free, win, call, txt). Chỉ có vài từ đẩy nhẹ về spam như "text", "to", còn lại là từ thường gặp trong tin cá nhân như "i", "me", "how", vốn có trọng số kéo về ham. BoW chỉ cộng trọng số từng từ nên tổng điểm vẫn nghiêng về ham, dù ý định là lừa việc làm.
- **Case 2 (spam giọng tự nhiên):** *"Thanks for being one of our loyal customers, as a thank you we've added a gift to your account, tap to unlock before Sunday"*. Mồi nhử có hạn chót nhưng không dùng chữ "free/win". Ngoài vài từ phổ biến như "your", "to", BoW không có từ khoá spam mạnh nào để bám vào, nên tổng điểm không vượt ngưỡng. Embedding và LLM tốt hơn vì nhìn được nghĩa của cả câu: "quà trong tài khoản + hạn chót + bấm để nhận" là mẫu lừa đảo quen thuộc.
- Embedding gặp khó ở nhóm "nói về spam" (25–75%): tin kể lại một vụ lừa đảo có chủ đề giống tin lừa đảo nên vector nằm gần vùng spam. Embedding nắm được chủ đề nhưng không nắm được ý định.

## 5. Thử thách cá nhân

**Đề #2 (theo SEED 2341):** Giảm vocab còn 100 từ và tăng lên 5000 từ (`build_vocab(..., max_size)`). Train lại, báo F1 test. Giải thích xu hướng.

> Chưa làm. Phần này phải tự làm, không dùng AI.

## 6. Phương án cho sản phẩm thật

- **Yêu cầu:** 10 triệu SMS/ngày (~116 tin/s), p95 < 50 ms/tin.
- **Đạt độ trễ:** cả 5 cách không dùng LLM: BoW + LogReg (0.12 ms), BoW + MLP (0.54 ms), 3a (16.4 ms), 3b (12.5 ms), 3c (11.5 ms). Cả 3 cách LLM (p95 1.2–5.6 s) đều vượt.
- **Chọn 3c Encoder + MLP head:** F1 test 0.967 (cao nhất), p95 11.5 ms, ~1,900 tin/s khi chạy cả lô, dư cho 116 tin/s. 3c cũng bền hơn BoW + MLP trên 50 case khó (68% so với 54%).
- **Đánh đổi chấp nhận:** 3c cần GPU. Ở lần chạy trước (encode chậm hơn ~30 lần), p95 của 3c là 58 ms, vượt SLA. Nếu không có GPU thì dùng BoW + MLP (F1 0.955, p95 0.54 ms). 3c vẫn dễ lọt spam kiểu mới (spam giọng tự nhiên chỉ đúng 40%), nên em đề xuất thêm một router: tin có p(spam) ∈ [0.3, 0.7] hoặc có dấu hiệu bất thường thì chuyển lên LLM nhỏ + prompt phòng thủ (đúng 50/50 case khó). Giữ tỉ lệ chuyển lên LLM dưới 5% thì p95 vẫn do đường nhanh quyết định.
- **Chi phí LLM nếu dùng cho toàn bộ:** 300 triệu tin/tháng × $0.006/1k tin ≈ $1,800/tháng với LLM nhỏ ($18,900 với LLM lớn), và không đạt SLA. Với router 5%, chi phí chỉ còn khoảng $90/tháng.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời/viết gì | Đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 4 · TODO 4.1 `call_openai` | Nhờ Claude viết hàm gọi Chat Completions theo docstring | Hàm dùng `client.chat.completions.create(...)` + `**provider_kwargs(max_tokens)`, đo latency bằng `time.perf_counter()` | Ô kiểm tra ✅ với API giả lập; gọi thử API thật với `gpt-6-luna`; đổi `max_tokens` của OpenAI từ 3 lên 16 vì model không trả đủ nhãn |
| Phần 4 · TODO 4.2–4.3 prompt + `parse_label` | Nhờ Claude viết prompt zero-shot/few-shot và hàm đọc nhãn | Prompt kết thúc bằng `Label:`; `parse_label` đổi "not spam" → "ham", trả None khi không rõ | Ô kiểm tra ✅; invalid = 0 ở cả 4a và 4b |
| Phần 4 · TODO 4.4 `llm_classify_batch` | Nhờ Claude viết vòng phân loại cả loạt + ước tính chi phí | Vòng lặp dùng `retry_with_backoff`, đếm `n_invalid`, tính USD theo `PRICE_PER_1M` | Ô kiểm tra ✅; chi phí model lớn ≈10 lần model nhỏ, khớp tỉ lệ giá |
| Câu trả lời Phần 0–5 + báo cáo này | Nhờ Claude Code rà notebook theo RUBRIC, sửa format, cập nhật số liệu cho khớp output lần chạy cuối, bổ sung các câu còn thiếu | Sửa và viết lại phần chữ trong các ô markdown và file báo cáo này; không sửa ô code nào | Số liệu được đối chiếu với output lần chạy cuối (bảng so sánh, bảng 50 case khó, biểu đồ) |

**Điều em hiểu ra hôm nay:** điểm cao trên tập test cùng phân phối không nói lên độ bền của model. BoW đạt F1 0.88–0.96 trên test thường nhưng chỉ còn 50–54% trên 50 case khó, còn LLM nhờ hiểu ý định nên gần như không tụt. Ngoài ra, chênh lệch giữa hai model có thể đến từ cách train chứ không phải kiến trúc (bỏ ReLU vẫn đạt 0.955), nên phải làm thí nghiệm loại trừ mới biết nguyên nhân thật.
