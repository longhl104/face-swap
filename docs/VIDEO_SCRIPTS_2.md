# Phần 2: Tôi đã xây dựng mô hình Face Swap như thế nào?

> Thời lượng dự kiến: 4–5 phút  
> Phong cách: giải thích kỹ thuật đơn giản, kết hợp sơ đồ và kết quả thực tế

## 0:00–0:35 — Bài toán tôi muốn giải quyết

Ở phần trước, chúng ta đã nói về deepfake, face swap, và bản chất của mô hình AI. Trong phần này, tôi sẽ kể cụ thể cách tôi đã xây dựng mô hình face swap của mình — bắt đầu từ bài toán mà mô hình cần giải quyết.

Mô hình của tôi nhận hai đầu vào.

Đầu tiên là nguồn **source** — ảnh chứa danh tính mà tôi muốn lấy. Thứ hai là mục tiêu **target** — ảnh hoặc video có khuôn mặt cần được thay thế.

Kết quả cần giữ được những đặc điểm nhận dạng của source, nhưng vẫn phải đi theo góc mặt, biểu cảm và ánh sáng của target. Đồng thời, vùng da sau khi ghép phải hòa vào khung hình đủ tự nhiên.

## 0:35–1:55 — Mô hình hoạt động như thế nào?

Nếu nghĩ theo cách thông thường, face swap cần làm được ba việc: biết khuôn mặt “là ai”, vẽ lại khuôn mặt đó lên người đích, và kiểm tra kết quả có trông thật hay không. Hệ thống của tôi cũng đi đúng theo logic đó, với ba thành phần chính.

Thứ nhất, tôi cần một cách đo danh tính. Nếu tự dạy mô hình “nhận mặt” từ đầu, sẽ rất lâu và dễ lệch. Vì vậy tôi dùng **FaceNet** — một mô hình đã được huấn luyện trước để biến khuôn mặt thành một dãy số đại diện cho danh tính. Tôi giữ FaceNet cố định, chỉ dùng như một bộ trích xuất đặc trưng, không huấn luyện lại nó.

Thứ hai, tôi cần một phần “vẽ lại” khuôn mặt. Đó là **Generator**, được tôi xây dựng theo kiến trúc U-Net. Generator nhận thông tin danh tính từ source và các đặc điểm từ target — góc mặt, biểu cảm, ánh sáng — để tạo ra khuôn mặt mới.

Tại sao lại U-Net? Vì nếu chỉ nén ảnh rồi khôi phục một cách thô, mô hình dễ làm mất vị trí mắt, miệng và cấu trúc khuôn mặt. U-Net có các đường nối giữa phần nén và phần khôi phục, giúp giữ lại những chi tiết không gian đó. Tôi sẽ giải thích kỹ hơn về kiến trúc U-Net ở các phần sau của series.

Thứ ba, chỉ có Generator thì chưa đủ. Nếu để mô hình tự tạo ảnh rồi tự chấm điểm, kết quả thường bị mờ hoặc “giả giả”. Vì vậy tôi thêm **Discriminator** — một bộ phận đóng vai trò giám khảo. Nó dùng kiến trúc PatchGAN: thay vì nhìn cả ảnh một lần rồi bảo “thật” hay “giả”, nó quan sát từng vùng nhỏ. Cách này hợp với face swap, vì lỗi thường lộ ở da, mắt hay đường viền khuôn mặt. Cách PatchGAN hoạt động chi tiết cũng sẽ được tôi đi sâu hơn ở các phần tiếp theo.

Hai mạng Generator và Discriminator được huấn luyện đối kháng với nhau. Generator cố gắng tạo ảnh ngày càng thật hơn, còn Discriminator cố gắng phát hiện ảnh do Generator tạo ra — giống như một người liên tục cải thiện bài làm trước một giám khảo ngày càng tinh mắt.

Cuối cùng, ảnh trông thật vẫn chưa đủ: kết quả còn phải giữ đúng danh tính của source. Vì vậy, trong quá trình huấn luyện, tôi so sánh đặc trưng FaceNet của ảnh kết quả với ảnh source bằng độ tương đồng cosine. Nói ngắn gọn: FaceNet giữ “là ai”, Generator tạo ảnh, Discriminator ép ảnh phải chân thực.

## 1:55–2:40 — Dữ liệu và chuẩn bị

Ba thành phần đó cần dữ liệu để học. Tôi đã xây dựng một quy trình chuẩn bị dữ liệu, huấn luyện mô hình và xử lý ảnh, video.

Dữ liệu huấn luyện ban đầu là bộ dữ liệu **LFW**, viết tắt của *Labeled Faces in the Wild*. Đây là tập hợp ảnh khuôn mặt của nhiều người trong các điều kiện chụp khác nhau — góc mặt, ánh sáng và biểu cảm đều đa dạng.

Tôi chọn LFW vì ba lý do thực tế. Thứ nhất, đây là bộ dữ liệu công khai, dễ tải về và đã được cộng đồng dùng lâu năm cho các bài toán nhận dạng khuôn mặt, nên tôi không phải mất thời gian thu thập hay gán nhãn từ đầu. Thứ hai, quy mô của nó vừa đủ để huấn luyện trên máy cá nhân với GPU 8 GB VRAM, trong khi các bộ lớn hơn như CelebA hay FFHQ đòi hỏi nhiều dung lượng lưu trữ và thời gian huấn luyện hơn. Thứ ba, vì ảnh được chụp “ngoài đời thật” chứ không chỉ trong studio, mô hình có cơ hội học sự biến thiên mà face swap thực tế cũng sẽ gặp.

Tất nhiên, LFW không phải lựa chọn tối ưu tuyệt đối cho face swap chuyên sâu — nhưng với lần đầu tự xây dựng và huấn luyện toàn bộ hệ thống, ưu tiên của tôi là một bộ dữ liệu đủ đa dạng, đủ nhỏ để lặp lại thí nghiệm nhanh, rồi mới tính đến mở rộng sau.

Mọi ảnh được tiền xử lý về kích thước 256 nhân 256 pixel trước khi đưa vào huấn luyện.

Tôi chọn kích thước này vì đây là điểm cân bằng giữa chất lượng và khả năng huấn luyện trên máy của tôi. Với GPU chỉ có 8 GB VRAM, nếu tăng lên 512 nhân 512 thì bộ nhớ sẽ đầy nhanh hơn nhiều, buộc phải giảm batch size hoặc thậm chí không chạy được ổn định. Batch size là số ảnh mà mô hình xử lý cùng lúc trong mỗi bước học — giống như học viên nhìn nhiều ví dụ trước khi chỉnh lại cách hiểu của mình. Batch lớn hơn thường giúp việc học ổn định hơn, nhưng cũng tốn nhiều bộ nhớ GPU hơn. Ở 256, tôi vẫn giữ được batch size 8 — đủ lớn để quá trình học ổn định hơn — đồng thời mỗi epoch chạy đủ nhanh để thử nghiệm và điều chỉnh. Ảnh nhỏ hơn nữa, chẳng hạn 128, sẽ tiết kiệm bộ nhớ hơn nhưng dễ làm mất chi tiết quanh mắt, miệng và đường viền khuôn mặt — những phần quan trọng với face swap. Vì vậy, 256 là lựa chọn thực dụng: đủ chi tiết để học danh tính, nhưng vẫn vừa với phần cứng hiện có.

## 2:40–3:20 — Quá trình huấn luyện

Tôi huấn luyện mô hình trên GPU NVIDIA RTX 5070 Laptop với 8 GB VRAM. Với giới hạn bộ nhớ này, ảnh được giữ ở kích thước 256 nhân 256 và batch size là 8.

Sau mỗi epoch, hệ thống tự động lưu loss, độ chính xác danh tính và biểu đồ huấn luyện. Điều này giúp tôi phát hiện khi mô hình không còn tiến bộ hoặc bắt đầu học lệch.

Sau 39 epoch, validation loss tốt nhất đạt khoảng 1 phẩy 87 ở epoch 36. Độ chính xác danh tính, được đo bằng độ tương đồng giữa embedding của source và khuôn mặt kết quả, đạt khoảng 90 phần trăm.

Con số đó cho thấy mô hình đã học được khá nhiều đặc điểm danh tính, nhưng nó không có nghĩa là 90 phần trăm kết quả đều hoàn hảo hoặc không thể phân biệt với ảnh thật.

## 3:20–4:10 — Kết quả và giới hạn

Ở ảnh và video có độ phân giải thấp đến trung bình, đặc biệt khoảng 480p trở xuống, kết quả giữ được danh tính tương đối tốt và hòa trộn khá ổn với tư thế cũng như màu da của target.

Tuy nhiên, mô hình hiện tại vẫn còn một giới hạn rõ ràng.

Vì khuôn mặt chỉ được xử lý ở kích thước 256 nhân 256, khi phóng trở lại video 720p hoặc cao hơn, một số vùng có thể xuất hiện mảng trắng và chi tiết bị vỡ. Lỗi này dễ thấy hơn khi khuôn mặt chiếm diện tích lớn trong khung hình.

Tôi có thể cải thiện bằng cách huấn luyện ở kích thước 512, nhưng cách này cần nhiều VRAM và thời gian hơn. Một hướng khác là thêm bước tinh chỉnh sau face swap, chẳng hạn một mạng tăng độ phân giải hoặc hiệu chỉnh màu chỉ dành cho vùng mặt.

## 4:10–4:45 — Kết thúc và giới thiệu series

Đây là lần đầu tiên tôi tự xây dựng và huấn luyện một hệ thống AI phức tạp như thế này, nên chắc chắn vẫn còn những quyết định chưa tối ưu và nhiều thứ cần cải thiện.

Trong những phần tiếp theo, tôi sẽ đi sâu hơn vào cách chuẩn bị dữ liệu, kiến trúc Generator và Discriminator, quá trình huấn luyện, những lần mô hình thất bại và cách tôi xử lý ảnh lẫn video.

Nếu bạn có kinh nghiệm về computer vision hoặc GAN, hãy để lại góp ý ở phần bình luận. Còn nếu bạn muốn theo dõi toàn bộ quá trình tôi biến ý tưởng này thành một hệ thống chạy được, hãy đăng ký kênh và đón xem phần tiếp theo.

Và tất nhiên, công nghệ này chỉ nên được sử dụng khi có sự đồng ý của người liên quan — không dùng để giả mạo, lừa đảo hoặc phát tán thông tin sai lệch.

Hẹn gặp lại các bạn trong video tiếp theo.
