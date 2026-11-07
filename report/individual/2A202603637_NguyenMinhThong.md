# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603637
- **Họ và tên:** Nguyễn Minh Thông
- **SEED** (in ở ô đầu notebook): 2341

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.878 | 0.919 | 0.106 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.955 | 0.974 | 2.220 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.900 | 0.976 | 61.947 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.952 | 0.947 | 35.637 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.967 | 0.974 | 58.033 | 0.000 |
| 4a. LLM nhỏ zero-shot (`gpt-6-luna`) | — | 0.878 | 1789.378 | 0.006 |
| 4b. LLM lớn zero-shot (`gpt-6.1-terra`) | — | 0.923 | 1440.796 | 0.063 |
| 4c. LLM nhỏ few-shot | — | 0.950 | 1343.786 | 0.022 |

Hàng LLM chỉ đo trên SUB (40 tin), nên so chất lượng giữa các cách bằng cột F1 SUB. Trên SUB, 1 tin sai làm F1 đổi khoảng 0.025, nên chênh dưới ~0.05 coi như hoà.

## 3. Dự đoán và thực tế

1. **`hard_eval` (50 case khó):** em đoán mọi model tụt mạnh, LLM chỉ còn ~75–85%, và nhóm khó nhất với LLM là prompt injection. Thực tế LLM nhỏ đạt 98%, bản có prompt phòng thủ đạt 100%, và đúng 100% ở nhóm prompt injection. Lệch vì em đánh giá thấp khả năng đọc hiểu ý định của LLM: các bẫy đánh vào từ khoá (né từ khoá, giọng tự nhiên) làm khó BoW chứ không làm khó một model đọc cả câu như người. Phần đoán cho BoW (~50–60%) thì khớp (50–54%).
2. **`dl_curve` / `dl_relu` (MLP so với LogReg):** em đoán MLP sẽ ngang LogReg, và bỏ ReLU thì mạng tương đương LogReg. Thực tế MLP đạt F1 0.955 so với 0.878, và bỏ ReLU vẫn đạt 0.955. Về mặt toán thì mạng không ReLU đúng là tuyến tính như LogReg, nhưng chênh lệch đến từ cách train: Adam + mini-batch train tới loss ~0, còn LogReg tự code chạy 300 epoch gradient descent vẫn chưa hội tụ (F1 trên train chỉ 0.917).
3. **`llm_fewshot`:** em đoán few-shot làm F1 gần như không đổi. Ở lần chạy cuối, few-shot đạt 0.950 so với 0.878 của zero-shot (+0.072, sai 2 tin so với 5 tin). Nhưng ở một lần chạy trước, kết quả lại ngược lại (zero-shot 0.950, few-shot 0.927). Lệch chủ yếu do API LLM không hoàn toàn tất định dù temperature=0 và SUB chỉ có 40 tin, nên chênh 2–3 tin chưa đủ để kết luận. Phần đoán chi phí tăng 2–4 lần thì khớp (≈3.7 lần).

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
- **Đạt độ trễ:** BoW + LogReg (0.11 ms), BoW + MLP (2.2 ms), 3b Encoder + LogReg (35.6 ms). 3a (61.9 ms), 3c (58.0 ms) và cả 3 cách LLM (1.3–1.8 s) đều vượt.
- **Chọn BoW + MLP:** F1 test 0.955 (ngang 3b là 0.952), p95 2.2 ms, ~7,900 tin/s khi chạy cả lô, nên một máy CPU đã dư cho 116 tin/s. 3b chỉ ~55 tin/s nên cần ít nhất 3 tiến trình song song.
- **Đánh đổi chấp nhận:** BoW + MLP chỉ đúng 54% trên 50 case khó, dễ lọt spam kiểu mới. Vì vậy em đề xuất thêm một router: tin có p(spam) ∈ [0.3, 0.7] hoặc có dấu hiệu né từ khoá thì chuyển lên LLM nhỏ + prompt phòng thủ (đúng 50/50 case khó). Giữ tỉ lệ chuyển lên LLM dưới 5% thì p95 vẫn do đường nhanh quyết định.
- **Chi phí LLM nếu dùng cho toàn bộ:** 300 triệu tin/tháng × $0.006/1k tin ≈ $1,800/tháng với LLM nhỏ ($18,900 với LLM lớn), và không đạt SLA. Với router 5%, chi phí chỉ còn khoảng $90/tháng.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời/viết gì | Đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 4 · TODO 4.1 `call_openai` | Nhờ Claude viết hàm gọi Chat Completions theo docstring | Hàm dùng `client.chat.completions.create(...)` + `**provider_kwargs(max_tokens)`, đo latency bằng `time.perf_counter()` | Ô kiểm tra ✅ với API giả lập; gọi thử API thật với `gpt-6-luna`; đổi `max_tokens` của OpenAI từ 3 lên 16 vì model không trả đủ nhãn |
| Phần 4 · TODO 4.2–4.3 prompt + `parse_label` | Nhờ Claude viết prompt zero-shot/few-shot và hàm đọc nhãn | Prompt kết thúc bằng `Label:`; `parse_label` đổi "not spam" → "ham", trả None khi không rõ | Ô kiểm tra ✅; invalid = 0 ở cả 4a và 4b |
| Phần 4 · TODO 4.4 `llm_classify_batch` | Nhờ Claude viết vòng phân loại cả loạt + ước tính chi phí | Vòng lặp dùng `retry_with_backoff`, đếm `n_invalid`, tính USD theo `PRICE_PER_1M` | Ô kiểm tra ✅; chi phí model lớn ≈10 lần model nhỏ, khớp tỉ lệ giá |
| Câu trả lời Phần 0–5 + báo cáo này | Nhờ Claude Code rà notebook theo RUBRIC, sửa format, cập nhật số liệu cho khớp output lần chạy cuối, bổ sung các câu còn thiếu | Sửa và viết lại phần chữ trong các ô markdown và file báo cáo này; không sửa ô code nào | Số liệu được đối chiếu với output lần chạy cuối (bảng so sánh, bảng 50 case khó, biểu đồ) |

**Điều em hiểu ra hôm nay:** điểm cao trên tập test cùng phân phối không nói lên độ bền của model. BoW đạt F1 0.88–0.96 trên test thường nhưng chỉ còn 50–54% trên 50 case khó, còn LLM nhờ hiểu ý định nên gần như không tụt. Ngoài ra, chênh lệch giữa hai model có thể đến từ cách train chứ không phải kiến trúc (bỏ ReLU vẫn đạt 0.955), nên phải làm thí nghiệm loại trừ mới biết nguyên nhân thật.
