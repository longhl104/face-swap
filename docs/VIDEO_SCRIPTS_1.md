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

**[Chèn trước/sau]** Ảnh/video gốc (target, ví dụ Mbappé) bên trái; kết quả face swap bên phải (`outputs/inference/swap_mbappe.jpg` hoặc clip video đã swap). Source identity: Messi.

Như các bạn có thể thấy, mô hình của tôi đã thay khuôn mặt của Mbappé bằng khuôn mặt của Messi, đồng thời vẫn giữ được biểu cảm, góc mặt và ánh sáng của ảnh Mbappé, giúp kết quả trông tương đối tự nhiên. Tôi sẽ đi sâu vào cách mô hình này hoạt động trong các phần tiếp theo của series.

Trước hết, deepfake và face swap khác nhau như thế nào?

## 0:20–1:30 — Deepfake và Face Swap

**[Hình ảnh]** Lần lượt các clip Manim trong `video_graphics/`:
1. `DeepfakeEquation` — “Deep Learning + Fake = Deepfake”
2. `DeepfakeMindmap` — sơ đồ các dạng deepfake, nhấn Face Swap
3. `NeuralNetwork` — mạng nơ-ron nhiều lớp (đường nét → mắt/mũi → khuôn mặt)
4. `DeepfakeForms` — deepfake dưới dạng hình ảnh / video / âm thanh và những gì nó làm được
5. `FaceSwapDefinition` — chuyển danh tính từ A sang B; Face Swap nằm trong tập Deepfake
6. `SourceTargetRoles` — người nguồn vs người đích; không phải cắt & dán

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

**[Hình ảnh]** Có thể tái dùng / cắt ngắn `NeuralNetwork`, hoặc quay màn hình: ảnh → ma trận số → qua các lớp → ảnh kết quả. Nhấn mạnh: mô hình làm việc trên điểm ảnh (số), không “nhìn” như người.

**Lời thoại:**

Trước khi đi sâu vào face swap, chúng ta cần hiểu bản chất của một mô hình AI.

Nói đơn giản, mô hình AI là một hệ thống toán học gồm rất nhiều tham số. Nó không nhìn và hiểu khuôn mặt giống con người. Với mô hình, mỗi bức ảnh chỉ là một tập hợp các con số biểu diễn màu sắc và độ sáng của từng điểm ảnh.

Trong quá trình huấn luyện, mô hình được xem rất nhiều ví dụ. Sau mỗi lần tạo kết quả, hệ thống đo xem kết quả sai lệch bao nhiêu so với mục tiêu, rồi điều chỉnh các tham số một chút. Quá trình này được lặp lại hàng nghìn lần để mô hình dần học được các quy luật trong dữ liệu.

Khi đã huấn luyện xong, mô hình sử dụng những quy luật đó để xử lý một hình ảnh mới. Vì vậy, nó không lưu sẵn rồi cắt và dán từng khuôn mặt, mà học mối liên hệ giữa các điểm ảnh với danh tính, biểu cảm, góc mặt và màu sắc.

## 2:15–2:45 — Kết thúc phần 1

**[Hình ảnh]** `Part1Outro` (`video_graphics/scenes/part1_outro.py`): flash vài kết quả từ `assets/images/results/`, CTA góp ý bình luận, preview chủ đề phần 2, rồi title card “PHẦN 2”.

**Lời thoại:**

Đây là lần đầu tiên tôi làm video theo dạng này, nên nếu có điều gì cần bổ sung hoặc chỉnh sửa, các bạn hãy để lại góp ý ở phần bình luận nhé.

Trong phần tiếp theo, tôi sẽ nói cụ thể về dữ liệu, kiến trúc mô hình, quá trình huấn luyện và những kết quả mà tôi đạt được.

Hẹn gặp lại các bạn trong phần 2.

---


Credits:
https://vtv.vn/video/canh-bao-thu-doan-su-dung-cong-nghe-deepfake-de-lua-dao-727508.htm
https://vtv.vn/video/gia-tang-lua-dao-bang-deepfake-tren-toan-cau-108670985.htm
https://vtv.vn/video/canh-giac-247-deepfake-bay-lua-dao-bang-ai-108689797.htm
https://www.youtube.com/watch?v=u7JcFm3oZoc
https://pixabay.com/music/electronic-tension-action-intro-rising-threat-464751/
https://pixabay.com/music/ambient-documentary-background-music-462075/