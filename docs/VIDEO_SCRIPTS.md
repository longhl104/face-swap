# FACE SWAP — Phần 1: Face Swap là gì và mô hình AI học như thế nào?

> Thời lượng lời thoại sau khi cắt khoảng lặng: 3 phút 06 giây  
> Phong cách: kể chuyện, dễ hiểu, có xen kẽ demo và hình minh họa

## 0:00–0:20 — Mở đầu

**[Hình ảnh/âm thanh tư liệu]** Một loạt đoạn tin tức ngắn về deepfake và các vụ giả mạo trên mạng. Giữ nguyên lời dẫn trong đoạn tin:

> Thời gian gần đây, Công an thành phố Hà Nội đã cảnh báo về thủ đoạn sử dụng công nghệ deepfake để giả mạo video, hình ảnh và giọng nói của người thân nhằm lừa đảo, chiếm đoạt tài sản. Chỉ trong quý đầu năm nay, các vụ lừa đảo sử dụng công nghệ deepfake đã khiến thế giới thiệt hại 200 triệu USD. Điều này cho thấy quy mô và mức độ đáng báo động của cuộc khủng hoảng này. Không chỉ ở Việt Nam, một sự việc tương tự cũng xảy ra với hai người dẫn chương trình nổi tiếng của kênh truyền hình Sky tại Anh: hình ảnh của họ bị giả mạo để quảng bá cho một casino trực tuyến.

Cuối đoạn tin, màn hình chuyển sang video do mô hình của tôi xử lý.

**Lời thoại:**

Những vụ việc vừa rồi cho thấy deepfake không còn chỉ là một hiệu ứng giải trí. Khi hình ảnh, video và giọng nói có thể bị giả mạo một cách thuyết phục, công nghệ này có thể gây ra những hậu quả rất thật.

Nhưng những nội dung giả mạo ấy được tạo ra như thế nào? Vì deepfake là một chủ đề rất rộng, trước tiên trong series này, tôi sẽ tập trung vào một kỹ thuật phổ biến: dùng mô hình trí tuệ nhân tạo, hay AI, để đưa khuôn mặt của một người vào ảnh hoặc video của người khác, còn gọi là face swap. Và để tìm hiểu cách nó hoạt động, tôi đã tự xây dựng một mô hình.

**[Chèn trước/sau]** Hiển thị ảnh hoặc video gốc bên trái, kết quả face swap bên phải.

Như các bạn có thể thấy, mô hình của tôi đã thay khuôn mặt của Mbappé bằng khuôn mặt của Messi, đồng thời vẫn giữ được biểu cảm, góc mặt và ánh sáng của ảnh Mbappé, giúp kết quả trông tương đối tự nhiên. Tôi sẽ đi sâu vào cách mô hình này hoạt động trong các phần tiếp theo của series.

Trước hết, deepfake và face swap khác nhau như thế nào?

## 0:20–1:30 — Deepfake và Face Swap

**[Hình ảnh]** Dòng chữ “Deep Learning + Fake = Deepfake”, sau đó hiện các ví dụ: đổi mặt, nhép môi, giả giọng nói.

**Lời thoại:**

Chúng ta cần phân biệt hai khái niệm thường bị dùng lẫn với nhau: **deepfake** và **face swap**.

Deepfake là khái niệm rộng, chỉ những nội dung giả hoặc bị chỉnh sửa bằng kỹ thuật học sâu, hay deep learning.

Học sâu là một phương pháp giúp máy tính học từ lượng lớn dữ liệu bằng mạng nơ-ron gồm nhiều lớp. Mạng nơ-ron là một mô hình toán học lấy cảm hứng từ cách các tế bào thần kinh kết nối với nhau. Nó gồm nhiều nút xử lý: mỗi nút nhận các con số, biến đổi chúng rồi truyền kết quả sang lớp tiếp theo.

Với hình ảnh, các lớp đầu có thể học những đặc điểm đơn giản như đường nét và màu sắc; các lớp sâu hơn dần nhận ra những đặc điểm phức tạp như mắt, mũi hoặc cả khuôn mặt. Chính khả năng nhận biết và tái tạo những đặc điểm này là nền tảng để deepfake tạo ra nội dung giả mạo ngày càng thuyết phục.

Vì vậy, deepfake có thể xuất hiện dưới nhiều dạng như hình ảnh, video hoặc âm thanh. Nó có thể khiến một người trông như đang nói điều họ chưa từng nói, bắt chước giọng nói, thay đổi biểu cảm hoặc đưa khuôn mặt của họ vào một video khác.

Face swap là một bài toán cụ thể hơn: chuyển danh tính khuôn mặt của một người sang khuôn mặt của người khác. Vì vậy, face swap có thể được xem là một ứng dụng của deepfake, nhưng deepfake không chỉ có face swap.

Để dễ theo dõi, trong video này tôi gọi người cung cấp **danh tính khuôn mặt** là **người nguồn**, còn người cung cấp **biểu cảm, góc mặt, ánh sáng và màu da** là **người đích**.

Việc “ghép” không đơn giản là cắt khuôn mặt từ ảnh này rồi dán lên ảnh khác. Mô hình phải chuyển những đặc điểm nhận dạng của người nguồn, đồng thời cố gắng bảo toàn biểu cảm và các đặc điểm hình ảnh của người đích.

Ví dụ, nếu người đích đang cười và quay mặt sang trái, kết quả vẫn phải giữ nụ cười cùng góc quay đó, nhưng mang danh tính của người nguồn. Nói ngắn gọn: **danh tính đến từ người nguồn, còn biểu cảm và bối cảnh khuôn mặt đến từ người đích**.

## 1:30–2:15 — Bản chất của mô hình AI

**[Hình ảnh]** Một bức ảnh được chuyển thành các con số, đi qua nhiều lớp của mạng nơ-ron rồi tạo ra ảnh kết quả.

**Lời thoại:**

Trước khi đi sâu vào face swap, chúng ta cần hiểu bản chất của một mô hình AI.

Nói đơn giản, mô hình AI là một hệ thống toán học gồm rất nhiều tham số. Nó không nhìn và hiểu khuôn mặt giống con người. Với mô hình, mỗi bức ảnh chỉ là một tập hợp các con số biểu diễn màu sắc và độ sáng của từng điểm ảnh.

Trong quá trình huấn luyện, mô hình được xem rất nhiều ví dụ. Sau mỗi lần tạo kết quả, hệ thống đo xem kết quả sai lệch bao nhiêu so với mục tiêu, rồi điều chỉnh các tham số một chút. Quá trình này được lặp lại hàng nghìn lần để mô hình dần học được các quy luật trong dữ liệu.

Khi đã huấn luyện xong, mô hình sử dụng những quy luật đó để xử lý một hình ảnh mới. Vì vậy, nó không lưu sẵn rồi cắt và dán từng khuôn mặt, mà học mối liên hệ giữa các điểm ảnh với danh tính, biểu cảm, góc mặt và màu sắc.

## 2:15–2:45 — Kết thúc phần 1

**[Hình ảnh]** Flash lại vài kết quả face swap, sau đó hiện title card “Phần 2”.

**Lời thoại:**

Đây là lần đầu tiên tôi làm video theo dạng này, nên nếu có điều gì cần bổ sung hoặc chỉnh sửa, các bạn hãy để lại góp ý ở phần bình luận nhé.

Trong phần tiếp theo, tôi sẽ nói cụ thể về dữ liệu, kiến trúc mô hình, quá trình huấn luyện và những kết quả mà tôi đạt được.

Hẹn gặp lại các bạn trong phần 2.

---

# Phần 2: Tôi đã xây dựng mô hình Face Swap như thế nào?

> Thời lượng dự kiến: 4–5 phút  
> Phong cách: giải thích kỹ thuật đơn giản, kết hợp sơ đồ và kết quả thực tế

## 0:00–0:35 — Bài toán tôi muốn giải quyết

**[Hình ảnh]** Minh họa hai đầu vào: “Source” và “Target”, sau đó đi qua mô hình để tạo “Result”.

**Lời thoại:**

Mô hình của tôi nhận hai đầu vào.

Đầu tiên là nguồn **source** — ảnh chứa danh tính mà tôi muốn lấy. Thứ hai là mục tiêu **target** — ảnh hoặc video có khuôn mặt cần được thay thế.

Kết quả cần giữ được những đặc điểm nhận dạng của source, nhưng vẫn phải đi theo góc mặt, biểu cảm và ánh sáng của target. Đồng thời, vùng da sau khi ghép phải hòa vào khung hình đủ tự nhiên.

## 0:35–1:10 — Tôi đã xây dựng những gì?

**[Hình ảnh]** Quay màn hình cấu trúc project, lướt nhanh qua các phần chuẩn bị dữ liệu, huấn luyện và xử lý ảnh, video.

**Lời thoại:**

Tôi đã xây dựng một hệ thống gồm quy trình chuẩn bị dữ liệu, huấn luyện mô hình và xử lý ảnh, video.

Dữ liệu huấn luyện ban đầu là bộ dữ liệu **LFW**, viết tắt của *Labeled Faces in the Wild*. Đây là tập hợp ảnh khuôn mặt của nhiều người trong các điều kiện chụp khác nhau.

Mọi ảnh được tiền xử lý về kích thước 256 nhân 256 pixel trước khi đưa vào huấn luyện.

## 1:10–2:15 — Mô hình hoạt động như thế nào?

**[Hình ảnh]** Sơ đồ đơn giản: Source → FaceNet → Identity; Target → U-Net Generator → Swapped Face; kết quả → Discriminator.

**Lời thoại:**

Hệ thống của tôi có ba thành phần chính.

Thành phần đầu tiên là **FaceNet**. Đây là một mô hình đã được huấn luyện trước để biến khuôn mặt thành một dãy số đại diện cho danh tính. Tôi sử dụng FaceNet như một bộ trích xuất đặc trưng và không huấn luyện lại nó.

Thành phần thứ hai là **Generator**, được tôi xây dựng theo kiến trúc U-Net. Generator nhận thông tin danh tính từ source và các đặc điểm từ target để tạo ra khuôn mặt mới.

U-Net có các đường nối giữa phần nén và phần khôi phục ảnh. Nhờ vậy, mô hình có thể giữ lại những chi tiết quan trọng như vị trí mắt, miệng và cấu trúc khuôn mặt.

Thành phần cuối cùng là **Discriminator**, sử dụng kiến trúc PatchGAN. Nhiệm vụ của nó là quan sát từng vùng nhỏ của ảnh và đánh giá xem vùng đó trông thật hay giả.

Hai mạng này được huấn luyện đối kháng với nhau. Generator cố gắng tạo ảnh ngày càng thật hơn, còn Discriminator cố gắng phát hiện ảnh do Generator tạo ra.

Ngoài việc làm ảnh trông chân thực, mô hình còn phải giữ đúng danh tính của source. Vì vậy, trong quá trình huấn luyện, tôi so sánh đặc trưng FaceNet của ảnh kết quả với ảnh source bằng độ tương đồng cosine.

## 2:15–2:55 — Quá trình huấn luyện

**[Hình ảnh]** Time-lapse màn hình terminal, GPU hoạt động và biểu đồ loss qua từng epoch.

**Lời thoại:**

Tôi huấn luyện mô hình trên GPU NVIDIA RTX 5070 Laptop với 8 GB VRAM. Với giới hạn bộ nhớ này, ảnh được giữ ở kích thước 256 nhân 256 và batch size là 8.

Sau mỗi epoch, hệ thống tự động lưu loss, độ chính xác danh tính và biểu đồ huấn luyện. Điều này giúp tôi phát hiện khi mô hình không còn tiến bộ hoặc bắt đầu học lệch.

Sau 34 epoch, validation loss tốt nhất đạt khoảng 2 phẩy 03 ở epoch 31. Độ chính xác danh tính, được đo bằng độ tương đồng giữa embedding của source và khuôn mặt kết quả, đạt khoảng 89 phần trăm.

Con số đó cho thấy mô hình đã học được khá nhiều đặc điểm danh tính, nhưng nó không có nghĩa là 89 phần trăm kết quả đều hoàn hảo hoặc không thể phân biệt với ảnh thật.

## 2:55–3:45 — Kết quả và giới hạn

**[Hình ảnh]** Cho xem lần lượt một kết quả tốt ở độ phân giải thấp và một kết quả lỗi ở độ phân giải cao. Khoanh vùng mảng trắng.

**Lời thoại:**

Ở ảnh và video có độ phân giải thấp đến trung bình, đặc biệt khoảng 480p trở xuống, kết quả giữ được danh tính tương đối tốt và hòa trộn khá ổn với tư thế cũng như màu da của target.

Tuy nhiên, mô hình hiện tại vẫn còn một giới hạn rõ ràng.

Vì khuôn mặt chỉ được xử lý ở kích thước 256 nhân 256, khi phóng trở lại video 720p hoặc cao hơn, một số vùng có thể xuất hiện mảng trắng và chi tiết bị vỡ. Lỗi này dễ thấy hơn khi khuôn mặt chiếm diện tích lớn trong khung hình.

Tôi có thể cải thiện bằng cách huấn luyện ở kích thước 512, nhưng cách này cần nhiều VRAM và thời gian hơn. Một hướng khác là thêm bước tinh chỉnh sau face swap, chẳng hạn một mạng tăng độ phân giải hoặc hiệu chỉnh màu chỉ dành cho vùng mặt.

## 3:45–4:20 — Kết thúc và giới thiệu series

**[Hình ảnh]** Montage ngắn: dữ liệu, code, biểu đồ huấn luyện và kết quả.

**Lời thoại:**

Đây là lần đầu tiên tôi tự xây dựng và huấn luyện một hệ thống AI phức tạp như thế này, nên chắc chắn vẫn còn những quyết định chưa tối ưu và nhiều thứ cần cải thiện.

Trong những phần tiếp theo, tôi sẽ đi sâu hơn vào cách chuẩn bị dữ liệu, kiến trúc Generator và Discriminator, quá trình huấn luyện, những lần mô hình thất bại và cách tôi xử lý ảnh lẫn video.

Nếu bạn có kinh nghiệm về computer vision hoặc GAN, hãy để lại góp ý ở phần bình luận. Còn nếu bạn muốn theo dõi toàn bộ quá trình tôi biến ý tưởng này thành một hệ thống chạy được, hãy đăng ký kênh và đón xem phần tiếp theo.

Và tất nhiên, công nghệ này chỉ nên được sử dụng khi có sự đồng ý của người liên quan — không dùng để giả mạo, lừa đảo hoặc phát tán thông tin sai lệch.

Hẹn gặp lại các bạn trong video tiếp theo.

Credits:
https://vtv.vn/video/canh-bao-thu-doan-su-dung-cong-nghe-deepfake-de-lua-dao-727508.htm
https://vtv.vn/video/gia-tang-lua-dao-bang-deepfake-tren-toan-cau-108670985.htm
https://vtv.vn/video/canh-giac-247-deepfake-bay-lua-dao-bang-ai-108689797.htm
https://www.youtube.com/watch?v=u7JcFm3oZoc
https://pixabay.com/music/electronic-tension-action-intro-rising-threat-464751/