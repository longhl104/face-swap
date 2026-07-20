# FACE SWAP — Phần 2: Chuẩn bị xây dựng mô hình Face Swap

> Thời lượng dự kiến: khoảng 6–7 phút  
> Phong cách: giải thích kỹ thuật đơn giản, kết hợp sơ đồ và quay màn hình code  
> Phạm vi phần này: bài toán → khung mô hình (FaceNet + Generator) → dữ liệu & tiền xử lý. Huấn luyện, loss và kết quả để dành cho các phần sau.

## 0:00–0:35 — Bài toán chúng ta muốn giải quyết

Ở phần trước, chúng ta đã nói về deepfake, face swap, và bản chất của mô hình AI. Trong phần này, chúng ta xây dựng mô hình face swap — nhưng chưa đi vào huấn luyện. Trước hết cần rõ bài toán, khung mô hình, rồi chuẩn bị dữ liệu cho đúng.

Mô hình của chúng ta nhận hai đầu vào.

Đầu tiên là nguồn **source** — ảnh chứa danh tính mà chúng ta muốn lấy. Thứ hai là mục tiêu **target** — ảnh hoặc video có khuôn mặt cần được thay thế.

Kết quả cần giữ được những đặc điểm nhận dạng của source, nhưng vẫn phải đi theo góc mặt, biểu cảm và ánh sáng của target. Đồng thời, vùng da sau khi ghép phải hòa vào khung hình đủ tự nhiên.

## 0:35–1:55 — Mô hình hoạt động như thế nào?

Nếu nghĩ theo cách thông thường, face swap cần làm được hai việc: biết khuôn mặt “là ai”, và vẽ lại khuôn mặt đó lên người đích. Người đích chính là target — người cung cấp góc mặt, biểu cảm, ánh sáng và khung hình mà khuôn mặt mới sẽ được đặt vào.

Thứ nhất, chúng ta cần một cách nhận biết danh tính khuôn mặt. Nếu tự dạy mô hình “nhận mặt” từ đầu, sẽ rất lâu và dễ lệch. Vì vậy chúng ta dùng **FaceNet** — một mô hình đã được huấn luyện trước để biến khuôn mặt thành một dãy số đại diện cho danh tính.

Thứ hai, chúng ta cần phần “vẽ lại” khuôn mặt. Đó là **Generator**. Generator nhận thông tin danh tính từ source và các đặc điểm từ target — góc mặt, biểu cảm, ánh sáng — để tạo ra khuôn mặt mới.

Generator không giữ nguyên từng điểm ảnh rồi chỉnh từng chỗ như Photoshop. Một bức ảnh có quá nhiều điểm ảnh để mô hình xử lý theo kiểu đó. Cách phổ biến là mô hình **nén** ảnh xuống thành một bản tóm tắt nhỏ — chỉ giữ những gì quan trọng như “đang cười, quay trái, ánh sáng thế nào” — rồi từ bản tóm tắt đó **vẽ lại** khuôn mặt mới.

Nếu chỉ nén rồi khôi phục một cách đơn giản, danh tính có thể còn, nhưng vị trí mắt, miệng và cấu trúc khuôn mặt dễ mất. Vì vậy Generator của chúng ta dùng kiến trúc **U-Net**: chúng ta có các đường nối tắt giữa phần nén và phần vẽ lại — giống như vừa gửi bản tóm tắt, vừa kèm theo vài “ảnh phác thảo” độ phân giải cao hơn để khi vẽ lại vẫn biết mắt và miệng nằm đâu. Kiến trúc này vốn được đề xuất để khoanh vùng tế bào trên ảnh kính hiển vi trong y học; về sau được dùng rộng trong các bài toán ảnh, kể cả face swap.

Đó là hai thành phần chính trong mô hình của chúng ta. Hệ thống còn có thêm Discriminator và quá trình huấn luyện với nhiều hàm mất mát hay còn gọi là loss functions; những phần đó sẽ được nói kỹ ở các video sau. Phần này tập trung vào việc chuẩn bị: chọn dữ liệu và đưa ảnh về dạng mô hình có thể học được.

## 1:55–2:40 — Dữ liệu và chuẩn bị

Mô hình của chúng ta cần dữ liệu để học. Trước khi huấn luyện, chúng ta cần chọn bộ dữ liệu phù hợp và tiền xử lý ảnh cho đúng.

Trước khi nói về dữ liệu, cần biết cấu hình máy chúng ta dùng để huấn luyện: tôi đang có con laptop ASUS proart với GPU NVIDIA RTX 5070 Laptop với 8 GB VRAM. Giới hạn bộ nhớ này sẽ ảnh hưởng trực tiếp đến các lựa chọn phía sau.

Dữ liệu huấn luyện ban đầu là bộ dữ liệu **LFW**, viết tắt của *Labeled Faces in the Wild*. Đây là tập hợp ảnh khuôn mặt của nhiều người trong các điều kiện chụp khác nhau — góc mặt, ánh sáng và biểu cảm đều đa dạng.

> https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

Chúng ta chọn LFW vì ba lý do thực tế. Thứ nhất, đây là bộ dữ liệu công khai, dễ tải về và đã được cộng đồng dùng lâu năm cho các bài toán nhận dạng khuôn mặt, nên không phải mất thời gian thu thập hay gán nhãn từ đầu. Thứ hai, quy mô của nó vừa đủ để huấn luyện trên máy cá nhân với GPU 8 GB VRAM, trong khi các bộ lớn hơn như CelebA hay FFHQ đòi hỏi nhiều dung lượng lưu trữ và thời gian huấn luyện hơn. Thứ ba, vì ảnh được chụp “ngoài đời thật” chứ không chỉ trong studio, mô hình có cơ hội học sự biến thiên mà face swap thực tế cũng sẽ gặp.

Tất nhiên, LFW không phải lựa chọn tối ưu tuyệt đối cho face swap chuyên sâu — nhưng với lần đầu tự xây dựng và huấn luyện toàn bộ hệ thống, ưu tiên của chúng ta là một bộ dữ liệu đủ đa dạng, đủ nhỏ để lặp lại thí nghiệm nhanh, rồi mới tính đến mở rộng sau.

Vậy bộ dữ liệu đã chọn rồi — còn bước đưa ảnh LFW về dạng mô hình có thể học được diễn ra như thế nào? Hãy cùng nhìn vào đoạn code tiền xử lý chính.

Chúng ta sẽ không đi sâu vào phần thiết lập môi trường hay toàn bộ cấu trúc project — chỉ cần quan tâm những đoạn code quan trọng, nơi quyết định dữ liệu được xử lý thế nào. Về cơ bản, tôi viết hệ thống bằng Python và dùng một số thư viện phổ biến cho xử lý ảnh và xây dựng mô hình trí tuệ nhân tạo. Code đầy đủ nằm trên GitHub nếu các bạn muốn xem thêm.

## 2:40–3:40 — Code tiền xử lý

**[Hình ảnh/Code]** Quay màn hình lần lượt:
1. `scripts/preprocess_dataset.py` — hai đoạn vòng lặp bên dưới
2. `src/data/preprocess.py` — các đoạn quan trọng của `FacePreprocessor` bên dưới

Chỉ highlight phần chính (không cần hiện cả file). Code đầy đủ: https://github.com/longhl104/face-swap

```python
# 1) Thu thập ảnh thô và chuẩn bị thư mục đầu ra
paths = StoragePaths()
raw = paths.raw_data
out = paths.processed_data

image_paths = collect_image_paths(raw)
```

```python
# 2) Với mỗi ảnh: phát hiện khuôn mặt → crop/resize → lưu .npy
with FacePreprocessor() as preprocessor:
    for img_path in tqdm(image_paths, desc="Preprocessing faces"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        face = preprocessor.process_image(image)
        if face is None:
            continue

        rel = img_path.relative_to(raw)
        out_path = out / rel.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, face.astype(np.float32))
```

```python
# 3) FacePreprocessor: khởi tạo bộ phát hiện YuNet (OpenCV)
self.image_size = load_config()["image_size"]
model_path = _ensure_yunet_model()
self._detector = cv2.FaceDetectorYN.create(
    str(model_path), "", (320, 320), 0.6, 0.3, 5000
)
```

```python
# 4) detect_face: tìm khuôn mặt lớn nhất trong ảnh
_, faces = self._detector.detect(image)
best = max(faces, key=lambda f: f[2] * f[3])
x, y, fw, fh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
return FaceRegion(bbox=(x, y, fw, fh), landmarks=landmarks)
```

```python
# 5) align_crop: cắt vùng mặt → làm thành hình vuông → resize
crop = image[y1:y2, x1:x2]
side = max(ch, cw)
square = cv2.copyMakeBorder(
    crop, top, bottom, left, right,
    borderType=cv2.BORDER_REFLECT_101,
)
face = cv2.resize(square, (self.image_size, self.image_size))
```

```python
# 6) process_image: ghép detect + crop thành một bước
region = self.detect_face(image)
if region is None:
    return None
return self.crop_and_align(image, region)
```

**Lời thoại (kèm code):**

Phần tiền xử lý nằm trong `scripts/preprocess_dataset.py`. Đoạn đầu chỉ làm một việc đơn giản: lấy đường dẫn ảnh thô từ `data/raw`, rồi chuẩn bị thư mục `data/processed` để chứa kết quả.

Đoạn chính là vòng lặp. Với mỗi ảnh, chúng ta đọc file bằng OpenCV, đưa qua `FacePreprocessor`, rồi lưu thành file `.npy`.

Vậy `FacePreprocessor` làm gì bên trong? Sang `src/data/preprocess.py`. Khi khởi tạo, lớp này đọc `image_size` từ config và nạp bộ phát hiện khuôn mặt **YuNet** của OpenCV.

Tiếp theo là `detect_face`: YuNet quét ảnh, rồi chúng ta chọn khuôn mặt lớn nhất — vì trong ảnh LFW thường chỉ quan tâm một mặt chính.

Sau khi có hộp bao quanh mặt, `align_crop` cắt vùng đó ra, làm thành hình vuông bằng `BORDER_REFLECT_101` — phản chiếu mép ảnh thay vì thêm viền đen — rồi resize về kích thước đã chọn, ví dụ 128 hoặc 256.

Vì sao lại hình vuông, chứ không để chữ nhật theo đúng tỉ lệ hộp mặt? Vì mạng nơ-ron của chúng ta nhận đầu vào có kích thước cố định — một cạnh nhân một cạnh, ví dụ 256×256 — và mỗi batch phải gồm nhiều ảnh cùng một shape để xử lý song song trên GPU. Nếu mỗi khuôn mặt giữ tỉ lệ dài-rộng khác nhau, sẽ phải resize méo hoặc thêm logic phức tạp hơn rất nhiều. Hình vuông là lựa chọn đơn giản và phổ biến: đủ chỗ cho cả khuôn mặt, dễ đưa về cùng một kích thước, và khớp với cách hầu hết mô hình ảnh — kể cả FaceNet và Generator — được thiết kế.

Cuối cùng, `process_image` chỉ là bước nối: phát hiện mặt, rồi crop/align; nếu không tìm thấy khuôn mặt thì trả về `None` và ảnh đó bị bỏ qua trong vòng lặp ở script.

Tại sao chúng ta cần bước này? Vì ảnh gốc trong LFW không đồng đều: khuôn mặt có thể nhỏ hoặc lớn, lệch sang một bên, và xung quanh còn nền, tóc, vai hoặc cả khung cảnh. Mô hình face swap học trên khuôn mặt, không phải trên cả bức ảnh. Nếu đưa nguyên ảnh vào, mạng sẽ lãng phí dung lượng để học nền thừa, đồng thời mỗi mẫu lại khác nhau về vị trí và tỉ lệ mặt — khiến việc học khó ổn định hơn nhiều.

## 3:40–4:20 — Kết thúc phần 2

**[Hình ảnh]** Có thể flash vài ảnh LFW trước/sau preprocess (raw → crop vuông 256), hoặc title card “PHẦN 3”. CTA góp ý bình luận.

**Lời thoại:**

Tóm lại, trước khi huấn luyện, chúng ta đã làm rõ ba việc. Một: mô hình nhận source và target, giữ danh tính của source nhưng đi theo góc mặt và biểu cảm của target. Hai: khung cơ bản gồm FaceNet để lấy danh tính và Generator dạng U-Net để vẽ lại khuôn mặt. Ba: chọn bộ dữ liệu LFW, rồi tiền xử lý — phát hiện mặt, làm thành hình vuông, resize và lưu thành `.npy`.

Đây mới là bước chuẩn bị. Trong phần tiếp theo, chúng ta sẽ đi vào quá trình huấn luyện, các hàm mất mát chính, và những kết quả ban đầu.

Nếu có chỗ nào chưa rõ các bạn để lại góp ý ở phần bình luận nhé.

Hẹn gặp lại các bạn trong phần 3.

---

Credits:
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
https://github.com/longhl104/face-swap
https://arxiv.org/abs/1505.04597

