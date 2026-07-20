
# Phần 2: Tôi đã xây dựng mô hình Face Swap như thế nào?

> Thời lượng dự kiến: 4–5 phút  
> Phong cách: giải thích kỹ thuật đơn giản, kết hợp sơ đồ và kết quả thực tế

## 0:00–0:35 — Bài toán tôi muốn giải quyết

**[Hình ảnh]** Sơ đồ Source → Model → Result (có thể tái dùng layout từ `SourceTargetRoles` / ảnh trong `video_graphics/assets/images/{source,target,result}.jpg`).

**Lời thoại:**

Mô hình của tôi nhận hai đầu vào.

Đầu tiên là nguồn **source** — ảnh chứa danh tính mà tôi muốn lấy. Thứ hai là mục tiêu **target** — ảnh hoặc video có khuôn mặt cần được thay thế.

Kết quả cần giữ được những đặc điểm nhận dạng của source, nhưng vẫn phải đi theo góc mặt, biểu cảm và ánh sáng của target. Đồng thời, vùng da sau khi ghép phải hòa vào khung hình đủ tự nhiên.

## 0:35–1:10 — Tôi đã xây dựng những gì?

**[Hình ảnh]** Quay màn hình cấu trúc project: `data/` (raw → processed), `src/`, `scripts/train.py`, `scripts/` inference, `outputs/training`, `outputs/inference`. Config chính: `config/config.yaml` (`image_size: 256`, `batch_size: 8`).

**Lời thoại:**

Tôi đã xây dựng một hệ thống gồm quy trình chuẩn bị dữ liệu, huấn luyện mô hình và xử lý ảnh, video.

Dữ liệu huấn luyện ban đầu là bộ dữ liệu **LFW**, viết tắt của *Labeled Faces in the Wild*. Đây là tập hợp ảnh khuôn mặt của nhiều người trong các điều kiện chụp khác nhau.

Mọi ảnh được tiền xử lý về kích thước 256 nhân 256 pixel trước khi đưa vào huấn luyện.

## 1:10–2:15 — Mô hình hoạt động như thế nào?

**[Hình ảnh]** Sơ đồ pipeline: Source → FaceNet → identity embedding; Target → U-Net Generator → swapped face; Discriminator (PatchGAN) chấm thật/giả; loss identity = cosine similarity FaceNet(result, source). (Scene Manim chưa có — cần vẽ riêng.)

**Lời thoại:**

Hệ thống của tôi có ba thành phần chính.

Thành phần đầu tiên là **FaceNet**. Đây là một mô hình đã được huấn luyện trước để biến khuôn mặt thành một dãy số đại diện cho danh tính. Tôi sử dụng FaceNet như một bộ trích xuất đặc trưng và không huấn luyện lại nó.

Thành phần thứ hai là **Generator**, được tôi xây dựng theo kiến trúc U-Net. Generator nhận thông tin danh tính từ source và các đặc điểm từ target để tạo ra khuôn mặt mới.

U-Net có các đường nối giữa phần nén và phần khôi phục ảnh. Nhờ vậy, mô hình có thể giữ lại những chi tiết quan trọng như vị trí mắt, miệng và cấu trúc khuôn mặt.

Thành phần cuối cùng là **Discriminator**, sử dụng kiến trúc PatchGAN. Nhiệm vụ của nó là quan sát từng vùng nhỏ của ảnh và đánh giá xem vùng đó trông thật hay giả.

Hai mạng này được huấn luyện đối kháng với nhau. Generator cố gắng tạo ảnh ngày càng thật hơn, còn Discriminator cố gắng phát hiện ảnh do Generator tạo ra.

Ngoài việc làm ảnh trông chân thực, mô hình còn phải giữ đúng danh tính của source. Vì vậy, trong quá trình huấn luyện, tôi so sánh đặc trưng FaceNet của ảnh kết quả với ảnh source bằng độ tương đồng cosine.

## 2:15–2:55 — Quá trình huấn luyện

**[Hình ảnh]** Time-lapse terminal khi chạy `scripts/train.py`; Task Manager/GPU; biểu đồ từ `outputs/training/` (loss & identity accuracy theo epoch, xem `training_metrics.csv`).

**Lời thoại:**

Tôi huấn luyện mô hình trên GPU NVIDIA RTX 5070 Laptop với 8 GB VRAM. Với giới hạn bộ nhớ này, ảnh được giữ ở kích thước 256 nhân 256 và batch size là 8.

Sau mỗi epoch, hệ thống tự động lưu loss, độ chính xác danh tính và biểu đồ huấn luyện. Điều này giúp tôi phát hiện khi mô hình không còn tiến bộ hoặc bắt đầu học lệch.

Sau 39 epoch, validation loss tốt nhất đạt khoảng 1 phẩy 87 ở epoch 36. Độ chính xác danh tính, được đo bằng độ tương đồng giữa embedding của source và khuôn mặt kết quả, đạt khoảng 90 phần trăm.

Con số đó cho thấy mô hình đã học được khá nhiều đặc điểm danh tính, nhưng nó không có nghĩa là 90 phần trăm kết quả đều hoàn hảo hoặc không thể phân biệt với ảnh thật.

## 2:55–3:45 — Kết quả và giới hạn

**[Hình ảnh]** So sánh: kết quả ổn ở ~480p (`outputs/inference/`, ví dụ `swap_mbappe.jpg`) vs lỗi khi phóng lên 720p+ — khoanh vùng mảng trắng / chi tiết vỡ quanh mặt.

**Lời thoại:**

Ở ảnh và video có độ phân giải thấp đến trung bình, đặc biệt khoảng 480p trở xuống, kết quả giữ được danh tính tương đối tốt và hòa trộn khá ổn với tư thế cũng như màu da của target.

Tuy nhiên, mô hình hiện tại vẫn còn một giới hạn rõ ràng.

Vì khuôn mặt chỉ được xử lý ở kích thước 256 nhân 256, khi phóng trở lại video 720p hoặc cao hơn, một số vùng có thể xuất hiện mảng trắng và chi tiết bị vỡ. Lỗi này dễ thấy hơn khi khuôn mặt chiếm diện tích lớn trong khung hình.

Tôi có thể cải thiện bằng cách huấn luyện ở kích thước 512, nhưng cách này cần nhiều VRAM và thời gian hơn. Một hướng khác là thêm bước tinh chỉnh sau face swap, chẳng hạn một mạng tăng độ phân giải hoặc hiệu chỉnh màu chỉ dành cho vùng mặt.

## 3:45–4:20 — Kết thúc và giới thiệu series

**[Hình ảnh]** Montage ngắn: ảnh LFW/processed, snippet code Generator–Discriminator, biểu đồ `outputs/training/`, vài kết quả `outputs/inference/`. Có thể tái dùng nhịp từ `Part1Outro`.

**Lời thoại:**

Đây là lần đầu tiên tôi tự xây dựng và huấn luyện một hệ thống AI phức tạp như thế này, nên chắc chắn vẫn còn những quyết định chưa tối ưu và nhiều thứ cần cải thiện.

Trong những phần tiếp theo, tôi sẽ đi sâu hơn vào cách chuẩn bị dữ liệu, kiến trúc Generator và Discriminator, quá trình huấn luyện, những lần mô hình thất bại và cách tôi xử lý ảnh lẫn video.

Nếu bạn có kinh nghiệm về computer vision hoặc GAN, hãy để lại góp ý ở phần bình luận. Còn nếu bạn muốn theo dõi toàn bộ quá trình tôi biến ý tưởng này thành một hệ thống chạy được, hãy đăng ký kênh và đón xem phần tiếp theo.

Và tất nhiên, công nghệ này chỉ nên được sử dụng khi có sự đồng ý của người liên quan — không dùng để giả mạo, lừa đảo hoặc phát tán thông tin sai lệch.

Hẹn gặp lại các bạn trong video tiếp theo.