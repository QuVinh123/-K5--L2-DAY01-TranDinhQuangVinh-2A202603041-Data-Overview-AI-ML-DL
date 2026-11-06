# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> \*\*Cách dùng mẫu này:\*\* copy/điền trên máy, lưu thành `REPORT.md`, rồi \*\*upload\*\* vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1\_lab\_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp \*\*link repo\*\* trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

|Từ|Nghĩa ngắn|
|-|-|
|Prediction|Kết quả model vừa đoán ra; chưa phải nhãn đúng|
|Ground truth|Nhãn do người gán theo quy tắc|
|Taxonomy|Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80)|
|Checkpoint|File model `.pt` đã huấn luyện sẵn|
|Guideline|Quy tắc gán nhãn do người viết|
|Annotator|Người gán nhãn|
|Reviewer / QC|Người kiểm tra nhãn|
|Escalation|Ca khó → hỏi Lab Coach, không tự bịa|
|Model score|Số “tin” của model; không phải điểm chất lượng nhãn người|

## 1\. Phân loại ảnh

**Mở file:** `day1\_lab\_outputs/classification\_predictions.json` và `visuals/classification\_top5.png`.  
Chỉ dùng ảnh `sample\_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class\_id`= 468 ; `class\_name`= cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy\_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về **cả bức ảnh**.
- Vì nhiệm vụ **Image Classification** (Phân loại ảnh) gán nhãn cho toàn bộ bức ảnh thay vì khoanh vùng từng đối tượng đơn lẻ. Mô hình dự đoán chủ đề nổi bật nhất các vật thể chiếm ưu thế trong ảnh (ở đây là chiếc xe taxi - cab).
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp **không phải do model tự nghĩ ra** mà đi **kèm với taxonomy/checkpoint** (cụ thể ở đây là tập nhãn ImageNet-1K gồm 1000 lớp có sẵn mà mô hình YOLO11n-cls đã được huấn luyện trên đó).
4. Vì sao cần ghi cả `class\_id`, `class\_name` và `taxonomy\_name`, không chỉ ghi mỗi tên lớp?
- **taxonomy\_name** để xác định không gian nhãn được sử dụng (ví dụ: ImageNet-1K khác với COCO-80).
- **class\_id** là mã định danh chuẩn hóa, không bị thay đổi bởi ngôn ngữ hoặc cách đặt tên.
- **class\_name** giúp con người dễ đọc và đối chiếu trực quan. Việc lưu cả ba thông tin giúp tránh nhầm lẫn khi tích hợp dữ liệu hoặc đổi mô hình.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline cần quy định rõ quy tắc ưu tiên (Priority Rule). 
+ Ví dụ: Chọn đối tượng chiếm diện tích lớn nhất, chọn đối tượng nằm ở trung tâm bức ảnh, hoặc ưu tiên các lớp động vật/người trước các lớp tĩnh vật.

6. Vì sao không được lấy `score` của model làm ground truth?

  - score (độ tin cậy) chỉ là **dự đoán xác suất của mô hình toán học** tại thời điểm đó, không đại diện cho sự thật khách quan (ground truth). Ground truth chuẩn phải do con người gán nhãn chính xác tuyệt đối (xác suất đúng bằng 1 hoặc 0) dựa theo guideline.

## 2\. Phát hiện vật thể

**Mở file:** `day1\_lab\_outputs/detection\_predictions.json` và `visuals/detection\_predictions.png`.  
Chỉ dùng ảnh `sample\_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class\_name`= person ; `score`= 0.912625 ; `bbox\_xyxy`=\[385.33, 69.24, 498.92, 348.92] ; `bbox\_width`= 113.58 ; `bbox\_height`= 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Góc trên trái:
+ Tính từ mép bên trái của ảnh đi sang phải 385.33 pixel.
+ Tính từ mép trên cùng của ảnh đi xuống dưới 69.24 pixel.
Đây chính là điểm bắt đầu (đầu hoặc vai) của người này trong ảnh.
- Góc dưới-phải: 
+ Tính từ mép bên trái của ảnh đi sang phải 498.92 pixel.
+ Tính từ mép trên cùng của ảnh đi xuống dưới 348.92 pixel.
Đây chính là điểm kết thúc (chân hoặc phần thấp nhất của cơ thể được phát hiện) của người này trong ảnh.

- Chiều rộng (bbox_width): Chiếc hộp bao quanh người này có bề ngang rộng 113.58 pixel.
- Chiều cao (bbox_height): Chiếc hộp bao quanh người này có chiều cao đứng là 279.68 pixel.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Ở ngưỡng 0.20: Có 17 vật thể được phát hiện.
- Ở ngưỡng 0.60: Chỉ có 6 vật thể được phát hiện.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Khi hạ ngưỡng, mô hình sẽ giữ lại rất nhiều hộp dự đoán có độ tin cậy thấp.
- Người Reviewer sẽ phải tốn nhiều công sức hơn để xóa bỏ các hộp vẽ sai (bị mô hình nhận diện nhầm) và điều chỉnh lại các hộp bị lệch.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
**Quy tắc:** *"Người gán nhãn phải vẽ hộp bao tiếp xúc trực tiếp và vừa khít tối đa với các điểm rìa ngoài cùng (trên, dưới, trái, phải) của vật thể mục tiêu với sai số cho phép không vượt quá 1-2 pixel. Tuyệt đối không được vẽ quá rộng gây thừa khoảng trống bên ngoài hay lồng phần nền vào trong hộp (No Padding), đồng thời không được cắt phạm hay lẹm vào bất kỳ bộ phận hoặc chi tiết nhỏ nào thuộc về vật thể (No Under-cropping). Trong trường hợp vật thể bị che khuất một phần (Occlusion), chỉ gán nhãn và bao quanh phần cấu trúc thực tế nhìn thấy được mà không tự suy diễn; đối với vật thể bị cắt bởi rìa ảnh (Truncation), hộp bao phải được kéo sát và dừng ngay tại đường biên cắt của khung hình."*

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Vật thể bị che khuất hoặc cắt mép là person 0.91 và oven 0.63
- Quy tắc gán nhãn hộp bao chuẩn (Tight Bounding Box) và Quy trình xử lý: Người gán nhãn phải vẽ hộp bao tiếp xúc trực tiếp và vừa khít tối đa với các điểm rìa ngoài cùng (trên, dưới, trái, phải) của vật thể mục tiêu với sai số cho phép không vượt quá 1-2 pixel, tuyệt đối không được vẽ rộng thừa khoảng trống bên ngoài (No Padding) và không được cắt lẹm vào bất kỳ chi tiết nhỏ nào thuộc vật thể (No Under-cropping). Đối với các trường hợp đặc biệt thường gặp thực tế như vật thể bị cắt bởi rìa ảnh (Truncation - ví dụ chiếc chảo góc dưới bên trái ảnh kitchen) hoặc bị che khuất một phần (Occlusion - ví dụ các bát xếp chồng lên nhau bên phải bồn rửa), người gán nhãn cần tự quyết định thao tác trực tiếp theo tài liệu hướng dẫn bằng cách vẽ bám sát theo phần cấu trúc thực tế nhìn thấy được (visible part) và kéo sát đến mép cắt của khung hình mà không cần báo cáo (Escalation). Chúng ta quyết định đưa các trường hợp này vào tài liệu hướng dẫn (Guideline) thay vì báo cáo hỏi ý kiến vì chúng xảy ra cực kỳ phổ biến trong các cảnh chụp thực tế; việc chuẩn hóa quy định rõ ràng giúp hàng loạt người gán nhãn làm việc đồng nhất, tăng tốc độ dự án và tránh lãng phí thời gian chờ đợi phản hồi. Người gán nhãn chỉ thực hiện báo cáo hỏi ý kiến Lab Coach (Escalation) trong các trường hợp có độ mơ hồ cực kỳ cao như vật thể bị che khuất nghiêm trọng trên 90% không thể nhận diện chính xác bằng mắt thường, hoặc khi gặp vật thể lai chưa rõ định nghĩa phân loại lớp nhãn trong hệ thống."

- Vật bị cắt mép: person 0.91 và oven 0.63

## 3\. Phân đoạn theo từng đối tượng

**Mở file:** `day1\_lab\_outputs/segmentation\_predictions.json` và `visuals/segmentation\_prediction.png`.  
Chỉ dùng ảnh `sample\_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance\_id`= **kitchen-002** ; `class\_name`= **bowl** ; `score`= **0.735743** ; số điểm polygon=`polygon\_point\_count`= **67**; 3–5 điểm đầu của `polygon\_xy`= **[[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0], [48.0, 346.0]]**

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- **Bounding box** chỉ là một hình chữ nhật bao quanh vật thể, chứa nhiều vùng pixel thừa. Trong khi đó, **polygon** (đa giác) cung cấp thông tin chính xác về **đường biên thực tế và hình dáng chi tiết của vật thể**, giúp phân biệt rõ đâu là pixel thuộc về vật thể và đâu là pixel nền.

3. `instance\_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- **instance\_id**: dùng để định danh riêng biệt cho từng cá thể (instance) cụ thể xuất hiện trong cùng một bức ảnh.
- "Nó **không** phải class ID và **không** phải tracking ID": Nó không đại diện cho tên nhóm đối tượng (class\_id) và nó cũng không dùng để theo dõi đối tượng đó qua các khung hình video khác nhau (tracking\_id). Nó chỉ đơn thuần là số thứ tự định danh để phân biệt vật thể A với vật thể B cùng loại trong một ảnh tĩnh duy nhất.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- "Quy tắc vẽ biên mask phân đoạn chuẩn: Người gán nhãn phải thực hiện vẽ đường biên đa giác (polygon) tiếp xúc trực tiếp, khít khao và bám sát tuyệt đối theo rìa ngoài cùng của vật thể mục tiêu với sai số cho phép cực nhỏ (không quá 1-2 pixel), đảm bảo không được vẽ thừa bao bản cả vùng nền xung quanh (No Padding) và không được vẽ thiếu hoặc cắt lẹm vào bất kỳ bộ phận cấu trúc chi tiết nào của vật thể. Khi thao tác vẽ qua các đoạn biên cong tự nhiên của đối tượng (như miệng bát tròn kitchen-002, vành đĩa, cốc nước), người gán nhãn bắt buộc phải tăng mật độ các điểm neo đặt sát nhau từ 2-5 pixel một điểm để tạo đường cong mượt mà tự nhiên, trong khi đối với các đoạn thẳng tắp chỉ cần neo điểm ở các góc giao chính để tối ưu hóa hiệu năng; đặc biệt đối với các trường hợp vật thể bị che khuất một phần thì chỉ vẽ bám theo phần cấu trúc thực tế nhìn thấy được chứ không tự vẽ suy đoán xuyên qua vật thể cản phía trước, còn với các vật thể bị cắt bởi rìa ảnh thì kéo đường đa giác bám dọc sát theo đúng đường mép ngoài cùng của khung hình trước khi đóng kín vòng đa giác hoàn chỉnh để hoàn thành nhãn."

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

- Vị trí: Chiếc bát (bowl) ở góc dưới bên trái bàn ăn màu xanh.
- Dùng Guideline.
Lý do: Chiếc bát bị bàn ăn che khuất một phần, khiến ranh giới giữa bát và mặt bàn không hoàn toàn rõ ràng. Tuy nhiên, vẫn có thể nhận diện được vật thể và phần đường biên nhìn thấy.
Theo guideline, chỉ cần annotation phần vật thể quan sát được, không tự suy đoán phần bị che khuất. Chỉ escalation khi không thể xác định đáng tin cậy ranh giới hoặc đối tượng cần gán nhãn.

+ Dùng Escalation khi: Biên vật thể quá mờ đến mức mắt người không thể phân biệt được đâu là ranh giới giữa hai vật, lúc này cần đưa lên Lab Coach để thống nhất tiêu chuẩn gán nhãn, tránh việc mỗi annotator tự đoán một kiểu gây nhiễu dữ liệu học máy.

## 4\. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

|**Tác vụ**|**Ground truth trông như thế nào? (đơn vị nhãn)**|**Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON**|**Annotator làm gì?**|**Reviewer xem gì?**|
|-|-|-|-|-|
|**Phân loại ảnh**|Là một nhãn duy nhất (class\_name kèm class\_id) gán cho toàn bộ bức ảnh (không có tọa độ).|Ảnh traffic có rất nhiều xe cộ và người, nhưng mô hình chỉ gán 1 nhãn duy nhất là cab (xe taxi) với score 0.51, bỏ qua hoàn toàn xe bus, người đi bộ, v.v.|Đọc toàn bộ ảnh, đối chiếu guideline lựa chọn chủ đề/đối tượng chính nổi bật nhất để gán nhãn duy nhất cho ảnh.|Kiểm tra xem nhãn được gán có đúng với chủ đề chính của ảnh theo guideline quy định hay không, có bị nhầm sang nhãn khác không.|
|**Phát hiện vật thể**|Là danh sách các hình hộp chữ nhật (bounding boxes), mỗi hộp gồm tọa độ \[x\_min, y\_min, x\_max, y\_max] tính bằng pixel và tên lớp đi kèm.|Trên ảnh kitchen, ở ngưỡng 0.20 có tới 17 vật thể (nhiều hộp rác thừa thãi như spoon, potted plant), còn ở ngưỡng 0.60 thì bỏ sót nhiều vật thể nhỏ như cup, bowl. Vật thể bị che khuất một phần rất khó vẽ chính xác.|Vẽ các hộp chữ nhật bao sát xung quanh từng đối tượng đơn lẻ và chọn phân lớp tương ứng từ taxonomy.|Kiểm tra độ khít của các hộp (không thừa/thiếu pixel), kiểm tra xem có gán sai nhãn đối tượng không, và có bỏ sót vật thể nào không.|
|**Instance segmentation**|Là danh sách các đa giác (polygons) gồm chuỗi tọa độ \[\[x1, y1], \[x2, y2], ...] bao quanh chính xác biên của từng cá thể vật thể.|Trên ảnh kitchen, các cái bát (bowl) xếp chồng lên nhau tạo ra các đường biên đè lên nhau, rất khó phân định chính xác ranh giới thực tế của từng cái bát bằng mắt thường.|Dùng công cụ vẽ đa giác chấm các điểm neo (anchor points) viền sát theo mép biên thực tế của từng vật thể (độ chính xác cấp độ pixel).|Kiểm tra xem đường viền polygon có bám sát biên thực tế của vật thể không, có bị răng cưa, thiếu góc hoặc vẽ lấn sang vật thể khác không.|

## 5\. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

-  Quy tắc bảo vệ dữ liệu: Tôi không chia sẻ, sao chép, tải xuống hoặc sử dụng dữ liệu lab cho mục đích cá nhân; luôn bảo mật thông tin theo quy định.
-  Khi gặp dữ liệu ngoài phạm vi: Tôi sẽ dừng xử lý ngay và báo cho Team Leader hoặc người phụ trách dự án (Supervisor/QA) để được hướng dẫn.

## 6\. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

* \[ ] `REPORT.md` (file này đã điền)
* \[ ] `day1\_lab\_outputs/classification\_predictions.json`
* \[ ] `day1\_lab\_outputs/detection\_predictions.json`
* \[ ] `day1\_lab\_outputs/segmentation\_predictions.json`
* \[ ] `day1\_lab\_outputs/IMAGE\_ATTRIBUTION.md`
* \[ ] `day1\_lab\_outputs/visuals/classification\_top5.png`
* \[ ] `day1\_lab\_outputs/visuals/detection\_predictions.png`
* \[ ] `day1\_lab\_outputs/visuals/segmentation\_prediction.png`
* \[ ] Ô validation cuối notebook báo `PASS`
* \[ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
* \[ ] Đã nộp **link repository** trên LMS

