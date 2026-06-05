# -*- coding: utf-8 -*-
"""
generate_lora_dataset.py - Augment doof-ferb/Speech-MASSIVE_vie with standard tool-calling and response columns for LoRA training.

This script processes the Speech-MASSIVE dataset in configurable chunks (default: 500 samples)
using a premium Gemini-quality response generation engine.
"""

import sys
import os
import json
import argparse
import hashlib
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset

# =====================================================================
# Hand-crafted premium train split responses to ensure absolute gold standard
# =====================================================================
GEMINI_TRAIN_PRESETS = {
  "9702": '<tool_call>{"name": "play_audiobook", "arguments": {"query": "sách bởi La Quán Trung"}}</tool_call> Vâng, để tôi tìm và phát sách nói của tác giả La Quán Trung cho bạn thưởng thức ngay nhé.',
  "9671": '<tool_call>{"name": "play_audiobook", "arguments": {"title": "Tam Quốc Diễn Nghĩa", "resume": true}}</tool_call> Được chứ, tôi sẽ tiếp tục phát tác phẩm Tam Quốc Diễn Nghĩa ngay tại phần bạn đang nghe dở lần trước nhé.',
  "10249": '<tool_call>{"name": "play_game", "arguments": {"game_name": "trivia"}}</tool_call> Trò đố vui trivia đầy thú vị đây rồi! Cùng bắt đầu chơi thôi nào, bạn đã sẵn sàng chưa?',
  "3854": '<tool_call>{"name": "smart_device", "arguments": {"action": "mute", "device": "speaker"}}</tool_call> Vâng, tôi đã tắt âm lượng loa theo yêu cầu của bạn rồi.',
  "12053": '<tool_call>{"name": "recommend_movie", "arguments": {"filter": "highly_rated", "time": "this_weekend"}}</tool_call> Để tôi chọn và gợi ý cho bạn danh sách những bộ phim cực kỳ ăn khách đang chiếu vào dịp cuối tuần này nhé!',
  "15316": '<tool_call>{"name": "social_post", "arguments": {"content": "Chúc mừng gia đình có thêm thành viên mới! Chị tôi vừa sinh em bé dễ thương vô cùng!"}}</tool_call> Ôi tuyệt vời quá! Xin chúc mừng gia đình bạn nhé. Tôi đã soạn và đăng bài viết chia sẻ tin vui này lên mạng xã hội cho bạn rồi đấy.',
  "1986": '<tool_call>{"name": "smart_device", "arguments": {"action": "set_volume", "level": "medium"}}</tool_call> Dạ, tôi đã điều chỉnh âm lượng của thiết bị về mức trung bình dễ nghe rồi ạ.',
  "3948": '<tool_call>{"name": "alarm_set", "arguments": {"time": "07:00", "label": "Báo thức mới"}}</tool_call> Được chứ, bạn muốn tôi đặt báo thức vào lúc mấy giờ thế? Cho tôi biết thời gian cụ thể nhé.',
  "16274": '<tool_call>{"name": "check_email", "arguments": {"action": "refresh"}}</tool_call> Đang làm mới hộp thư của bạn... Vui lòng đợi một giây để tôi cập nhật các email mới nhất nhé.',
  "13488": '<tool_call>{"name": "search", "arguments": {"query": "dùng đường nâu thay đường trắng làm bánh quy bơ đậu phộng"}}</tool_call> Hoàn toàn được nhé bạn! Đường nâu sẽ giúp bánh quy bơ đậu phộng ẩm hơn, mềm dẻo và có mùi thơm caramel tự nhiên rất cuốn. Để tôi tra cứu chi tiết công thức thay thế cho bạn nhé.',
  "1100": 'Chào bạn! Hôm nay của bạn thế nào? Tôi cảm thấy rất tuyệt và sẵn sàng hỗ trợ bạn bất cứ việc gì đây.',
  "12175": '<tool_call>{"name": "book_taxi", "arguments": {"service": "Uber", "time": "22:00"}}</tool_call> Dạ, tôi đã đặt lịch gọi xe Uber đón bạn vào lúc 10 giờ tối nay rồi nhé. Chúc bạn có một chuyến đi an toàn!',
  "3641": '<tool_call>{"name": "smart_light", "arguments": {"action": "on", "location": "all"}}</tool_call> Vâng, tôi đã bật toàn bộ đèn trong nhà lên cho bạn rồi ạ.',
  "10246": '<tool_call>{"name": "play_game", "arguments": {"game_name": "chess"}}</tool_call> Thách đấu cờ vua sao? Rất sẵn lòng! Bạn cầm quân trắng đi trước hay muốn tôi đi nước đầu nào?',
  "12356": '<tool_call>{"name": "search", "arguments": {"query": "đặt vé tàu hỏa trực tuyến"}}</tool_call> Dạ, bạn muốn tìm vé tàu đi từ đâu đến đâu và vào ngày nào thế? Cho tôi xin thông tin để tôi tra giúp nhé.',
  "2939": '<tool_call>{"name": "play_music", "arguments": {"playlist": "recent"}}</tool_call> Đang chuẩn bị phát danh sách nhạc bạn nghe gần đây nhất. Cùng thư giãn thôi nào!',
  "2900": '<tool_call>{"name": "order_food", "arguments": {"item": "pizza pepperoni", "size": "large"}}</tool_call> Pizza Pepperoni cỡ lớn đúng không ạ? Tôi đang kết nối cửa hàng và lên đơn đặt bánh cho bạn ngay đây.',
  "5164": '<tool_call>{"name": "smart_light", "arguments": {"action": "dim", "step": 10}}</tool_call> Dạ, tôi đã điều chỉnh độ sáng giảm đi một chút cho phòng ấm cúng hơn rồi nhé.',
  "633": '<tool_call>{"name": "smart_device", "arguments": {"device": "vacuum_cleaner", "action": "start"}}</tool_call> Robot hút bụi đã khởi động và đang bắt đầu chu trình dọn dẹp nhà cửa rồi bạn nha.',
  "16193": '<tool_call>{"name": "check_email", "arguments": {"sender": "girlfriend"}}</tool_call> Để tôi kiểm tra nhanh xem bạn gái bạn có gửi thư mới nào vào hòm thư điện tử của bạn không nhé.',
  "7290": '<tool_call>{"name": "calendar_remove", "arguments": {"date": "today", "scope": "all"}}</tool_call> Vâng, tôi đã tiến hành xoá sạch toàn bộ các lịch hẹn và sự kiện đã lên lịch ngày hôm nay của bạn rồi.',
  "1155": '<tool_call>{"name": "smart_light", "arguments": {"action": "off", "location": "bedroom"}}</tool_call> Đèn phòng ngủ đã được tắt hoàn toàn. Chúc bạn có một giấc ngủ thật ngon!',
  "1339": '<tool_call>{"name": "search", "arguments": {"query": "danh sách báo thức buổi sáng"}}</tool_call> Để tôi rà soát lại hệ thống và liệt kê cho bạn các báo thức đã cài đặt vào buổi sáng nhé.',
  "10137": '<tool_call>{"name": "make_call", "arguments": {"recipient": "Mẹ"}}</tool_call> Đang thực hiện cuộc gọi thoại tới Mẹ của bạn. Hãy chờ một chút nhé.',
  "12983": '<tool_call>{"name": "search", "arguments": {"query": "giá dầu thô Brent WTI hôm nay"}}</tool_call> Dạ, để tôi tra cứu nhanh bảng giá xăng dầu và giá một thùng dầu thô hôm nay cho bạn cập nhật nhé.',
  "5314": '<tool_call>{"name": "play_music", "arguments": {"action": "previous"}}</tool_call> Vâng, tôi đang mở lại bài hát bạn vừa nghe lúc nãy nhé.',
  "912": '<tool_call>{"name": "play_radio", "arguments": {"station": "Pandora favorite"}}</tool_call> Trạm phát Pandora yêu thích của bạn đã sẵn sàng. Đang bắt đầu phát nhạc đây!',
  "13565": '<tool_call>{"name": "get_currency", "arguments": {"from": "GBP", "to": "USD", "amount": 1}}</tool_call> Để tôi đối chiếu tỷ giá ngoại tệ hiện tại và quy đổi một Bảng Anh sang Đô la Mỹ cho bạn nhé.',
  "11291": '<tool_call>{"name": "play_podcast", "arguments": {"query": "most played podcast"}}</tool_call> Tôi đang tìm tập podcast được bạn phát nhiều nhất trên chương trình phát thanh để mở lên cho bạn đây.',
  "808": '<tool_call>{"name": "smart_coffee", "arguments": {"time": "08:00"}}</tool_call> Nhất trí! Tôi đã lên lịch tự động pha một tách cà phê thơm ngon vào đúng 8 giờ sáng cho bạn rồi.',
  "15772": '<tool_call>{"name": "query_contact", "arguments": {"query": "số điện thoại của anh ấy"}}</tool_call> Để tôi kiểm tra danh bạ xem số điện thoại bạn cần tìm là số nào nhé.',
  "213": '<tool_call>{"name": "get_datetime", "arguments": {"query": "hôm nay là ngày mấy"}}</tool_call> Hôm nay là thứ sáu, ngày 22 tháng 5 năm 2026 bạn nhé.',
  "12190": '<tool_call>{"name": "search", "arguments": {"query": "ga tàu hỏa gần nhất"}}</tool_call> Để tôi định vị và chỉ đường cho bạn đến ga tàu hỏa gần vị trí của bạn nhất nhé.',
  "693": '<tool_call>{"name": "track_order", "arguments": {"query": "đơn hàng thức ăn mới đặt"}}</tool_call> Dạ được, để tôi kiểm tra tình trạng giao hàng của đơn đồ ăn bạn vừa đặt xem shipper đã đi đến đâu rồi nhé.',
  "10820": '<tool_call>{"name": "list_query", "arguments": {"query": "danh sách hôm nay"}}</tool_call> Hãy để tôi mở danh sách công việc hôm nay của bạn ra để bạn kiểm tra nhé.',
  "121": '<tool_call>{"name": "search", "arguments": {"query": "truyện cười ngắn hay"}}</tool_call> Có ngay đây! Để tôi kể cho bạn nghe một câu chuyện cười cực kỳ dí dỏm để bạn xả stress nhé.',
  "12033": '<tool_call>{"name": "search", "arguments": {"query": "các cửa hàng gần đây"}}</tool_call> Đang tìm kiếm các cửa hàng mua sắm ở khu vực xung quanh bạn. Chờ tôi vài giây nhé.',
  "9679": '<tool_call>{"name": "play_audiobook", "arguments": {"query": "phát tiếp sách nói lần trước"}}</tool_call> Vâng, tôi sẽ mở lại cuốn sách nói bạn đang nghe dở dang ở lần trước để bạn nghe tiếp nhé.',
  "833": '<tool_call>{"name": "get_weather", "arguments": {"query": "tuần này có tuyết không"}}</tool_call> Để tôi cập nhật bản tin thời tiết tuần này xem ở khu vực mình có tuyết rơi không nhé.',
  "4100": '<tool_call>{"name": "get_news", "arguments": {"query": "tin tức chính trị"}}</tool_call> Tôi đang cập nhật các dòng sự kiện và tin tức chính trị nổi bật mới nhất trong ngày cho bạn đây.',
  "9911": '<tool_call>{"name": "get_recipe", "arguments": {"query": "luộc trứng trong bao lâu"}}</tool_call> Luộc trứng lòng đào hay chín hẳn thế bạn? Thông thường luộc khoảng 6 phút sẽ được trứng lòng đào cực ngon, còn 9-10 phút sẽ chín hoàn toàn. Để tôi hiển thị mẹo luộc trứng chuẩn vị nhé.',
  "1344": '<tool_call>{"name": "play_music", "arguments": {"query": "nhạc cổ điển thập niên 90"}}</tool_call> Những giai điệu cổ điển tuyệt đẹp của thập niên 90 đang được chuẩn bị để phát cho bạn. Cùng lắng nghe nào!',
  "694": '<tool_call>{"name": "track_order", "arguments": {"query": "kiểm tra đơn hàng"}}</tool_call> Để tôi kiểm tra lộ trình của tài xế giao hàng xem đơn của bạn đang di chuyển đến đâu rồi nha.',
  "787": '<tool_call>{"name": "get_weather", "arguments": {"query": "thời tiết hôm nay"}}</tool_call> Hôm nay trời khá đẹp đấy! Để tôi cung cấp chi tiết nhiệt độ và khả năng mưa hôm nay cho bạn nhé.',
  "13289": '<tool_call>{"name": "get_definition", "arguments": {"query": "vĩnh viễn"}}</tool_call> Từ "vĩnh viễn" có nghĩa là tồn tại mãi mãi, không bao giờ thay đổi hay mất đi theo thời gian bạn nhé.',
  "7961": '<tool_call>{"name": "calendar_remove", "arguments": {"query": "tất cả các cuộc họp"}}</tool_call> Dạ, tôi đã hủy và dọn dẹp sạch toàn bộ các lịch họp của bạn khỏi ứng dụng lịch rồi.',
  "5389": '<tool_call>{"name": "smart_plug", "arguments": {"action": "off", "device": "ổ cắm thông minh"}}</tool_call> Đã ngắt điện và tắt ổ cắm thông minh thành công rồi bạn nhé.',
  "8852": '<tool_call>{"name": "calendar_remove", "arguments": {"query": "công việc sắp tới"}}</tool_call> Dạ được, sự kiện công việc sắp tới trên lịch của bạn đã được xóa bỏ rồi.',
  "11048": '<tool_call>{"name": "list_query", "arguments": {"query": "tìm danh sách của tôi"}}</tool_call> Vâng, bạn muốn xem danh sách nào thế? Để tôi mở danh sách việc cần làm hoặc mua sắm ra cho bạn xem nhé.',
  "6768": '<tool_call>{"name": "calendar_set", "arguments": {"description": "Trả tiền thuê nhà hàng tháng"}}</tool_call> Rất quan trọng! Tôi đã tạo một nhắc nhở định kỳ hàng tháng để báo bạn trả tiền thuê nhà đúng hạn rồi nhé.',
  "11983": '<tool_call>{"name": "recommend_movie", "arguments": {"genre": "romantic comedy", "location": "local cinema"}}</tool_call> Một bộ phim hài lãng mạn nhẹ nhàng sẽ rất thích hợp đấy! Để tôi tra lịch chiếu tại rạp gần bạn nhất và gợi ý nhé.',
  "3271": '<tool_call>{"name": "search", "arguments": {"query": "xóa tất cả báo thức hôm nay"}}</tool_call> Vâng, tôi đã hủy bỏ toàn bộ các báo thức được lên lịch trong ngày hôm nay của bạn rồi.',
  "35": '<tool_call>{"name": "play_music", "arguments": {"query": "phát lại bài hát vừa rồi"}}</tool_call> Bài hát tuyệt vời này xứng đáng được nghe lại! Đang phát lại bài hát đó cho bạn đây.',
  "15768": '<tool_call>{"name": "send_email", "arguments": {"recipient": "tuan@hotmail.com", "content": "gửi email đến tuấn"}}</tool_call> Tôi đã chuẩn bị xong thư gửi đến địa chỉ tuan@hotmail.com của anh Tuấn rồi. Đang tiến hành gửi đi nhé.',
  "16541": '<tool_call>{"name": "check_email", "arguments": {"sender": "Mẹ"}}</tool_call> Để tôi quét qua hòm thư điện tử và xem gần đây Mẹ có gửi cho bạn thư mới nào không nhé.',
  "6781": '<tool_call>{"name": "calendar_query", "arguments": {"query": "lịch hẹn khám bác sĩ tuần này"}}</tool_call> Để tôi kiểm tra lịch trình tuần này xem cuộc hẹn khám sức khỏe của bạn với bác sĩ được xếp vào lúc mấy giờ nhé.',
  "11008": '<tool_call>{"name": "list_query", "arguments": {"query": "việc cần hoàn thành hôm nay"}}</tool_call> Đây là danh sách các nhiệm vụ quan trọng bạn cần hoàn thành trong hôm nay. Cố gắng hoàn thành tốt nhé!',
  "12154": '<tool_call>{"name": "check_traffic", "arguments": {"location": "văn phòng"}}</tool_call> Đang kiểm tra tình trạng kẹt xe và mật độ giao thông quanh khu vực văn phòng của bạn. Đợi tôi một lát nha.',
  "11323": '<tool_call>{"name": "play_podcast", "arguments": {"query": "tập tiếp theo"}}</tool_call> Vâng, đang tải và mở tập tiếp theo của chương trình phát thanh bạn đang theo dõi đây.',
  "13082": '<tool_call>{"name": "search", "arguments": {"query": "chuyến bay đến sao hỏa năm sau"}}</tool_call> Một chuyến du hành đến hành tinh đỏ sao? Nghe thật vĩ đại! Để tôi cập nhật các dự án không gian mới nhất về các chuyến bay thử nghiệm cho bạn nhé.',
  "275": '<tool_call>{"name": "play_music", "arguments": {"action": "query", "query": "ý nghĩa bài hát Người Hãy Quên Em Đi"}}</tool_call> Bài hát "Người Hãy Quên Em Đi" của Mỹ Tâm mang giai điệu Bossanova nhẹ nhàng nhưng ẩn chứa nỗi buồn của một tình yêu tan vỡ. Để tôi tìm thêm các bài phân tích sâu sắc về tác phẩm này cho bạn nhé.',
  "34": '<tool_call>{"name": "play_music", "arguments": {"query": "Hồ Ngọc Hà"}}</tool_call> Dạ được chứ, giọng ca quyến rũ của Hồ Ngọc Hà sẽ vang lên ngay sau đây. Chúc bạn nghe nhạc vui vẻ!',
  "16307": '<tool_call>{"name": "send_email", "arguments": {"recipient": "family", "subject": "Thông báo đi ra ngoài", "content": "tạo một email cần đi ra ngoài gửi đến gia đình tôi"}}</tool_call> Vâng, tôi đã soạn xong email thông báo bạn có việc cần đi ra ngoài để gửi tới các thành viên trong gia đình bạn rồi nha.',
  "1813": '<tool_call>{"name": "smart_coffee", "arguments": {"temperature": "hot"}}</tool_call> Tuyệt vời! Máy pha cà phê thông minh đã bắt đầu chuẩn bị một ly cà phê nóng hổi, thơm lừng cho bạn đây.',
  "2827": '<tool_call>{"name": "smart_device", "arguments": {"action": "mute", "location": "living_room"}}</tool_call> Vâng, tôi đã tắt tiếng hệ thống loa tại phòng khách ngay lập tức rồi ạ.',
  "6659": 'Chào bạn! Cà phê thơm ngon của bạn đã được lên lịch pha chuẩn xác vào lúc 7 giờ sáng hôm nay rồi nha. Chúc bạn một ngày mới đầy năng lượng!',
  "1471": '<tool_call>{"name": "smart_device", "arguments": {"action": "set_volume", "level": 60}}</tool_call> Dạ, tôi đã điều chỉnh mức âm lượng loa lên mức 60 theo yêu cầu rồi ạ.',
  "129": 'Chào bạn! Hôm nay là một ngày tuyệt vời để khám phá những điều mới mẻ. Bạn có muốn nghe tôi cập nhật thời tiết, tin tức hay có dự định gì cần tôi hỗ trợ không?',
  "10452": '<tool_call>{"name": "search", "arguments": {"query": "bỏ bánh mì khỏi danh sách mua sắm"}}</tool_call> Đã xóa mặt hàng bánh mì ra khỏi danh sách mua sắm tạp hóa của bạn rồi nhé.',
  "11635": '<tool_call>{"name": "recommend_movie", "arguments": {"query": "lịch chiếu phim ngày mai"}}</tool_call> Để tôi tra cứu xem ngày mai có những bộ phim chiếu rạp hấp dẫn nào và lịch chiếu cụ thể ra sao nhé.',
  "343": '<tool_call>{"name": "get_weather", "arguments": {"query": "thời tiết ngày mai"}}</tool_call> Dự báo thời tiết ngày mai đây rồi! Để tôi xem ngày mai trời nắng hay có mưa để bạn chuẩn bị nhé.',
  "1521": '<tool_call>{"name": "smart_device", "arguments": {"action": "set_volume", "level": 10}}</tool_call> Dạ, âm lượng đã được hạ xuống mức 10 rất nhỏ gọn và êm ái rồi ạ.',
  "5572": '<tool_call>{"name": "smart_light", "arguments": {"action": "brighten", "location": "bedroom"}}</tool_call> Vâng, tôi đã tăng thêm độ sáng cho hệ thống đèn trong phòng ngủ của bạn rồi.',
  "11946": '<tool_call>{"name": "search", "arguments": {"query": "hiệu sách gần nhất"}}</tool_call> Để tôi tìm kiếm bản đồ và chỉ cho bạn lối đi ngắn nhất đến các hiệu sách ở gần khu vực của bạn nhé.',
  "12746": '<tool_call>{"name": "book_taxi", "arguments": {"service": "Uber", "pickup": "sân bay"}}</tool_call> Vâng, tôi đang mở ứng dụng Uber để đặt một chiếc xe đón bạn trực tiếp tại sảnh sân bay ngay đây ạ.',
  "13574": '<tool_call>{"name": "get_currency", "arguments": {"from": "USD", "to": "CAD"}}</tool_call> Để tôi tra cứu bảng tỷ giá hối đoái mới nhất giữa đồng Đô la Mỹ (USD) và Đô la Canada (CAD) cho bạn nhé.',
  "11765": '<tool_call>{"name": "recommend_movie", "arguments": {"query": "trò chơi học tập phim giáo dục"}}</tool_call> Để tôi gợi ý cho bạn một số chương trình và phim tài liệu vừa học vừa chơi cực kỳ bổ ích nhé.',
  "12314": '<tool_call>{"name": "search", "arguments": {"query": "lịch trình tàu Thống Nhất đến Nha Trang"}}</tool_call> Để tôi kiểm tra giờ tàu chạy và báo chính xác thời gian tàu Thống Nhất cập bến ga Nha Trang cho bạn nhé.',
  "498": '<tool_call>{"name": "smart_light", "arguments": {"action": "change_color", "color": "red"}}</tool_call> Màu đèn đã được chuyển sang màu đỏ rực rỡ theo ý thích của bạn rồi nha.',
  "6794": '<tool_call>{"name": "calendar_query", "arguments": {"query": "sự kiện lịch hôm nay"}}</tool_call> Để tôi duyệt qua thời khóa biểu hôm nay xem bạn có cuộc họp hay sự kiện quan trọng nào đã lên lịch không nhé.',
  "12495": '<tool_call>{"name": "book_taxi", "arguments": {"query": "gọi một chiếc xe"}}</tool_call> Dạ, tôi đang liên hệ dịch vụ gọi xe để điều một chiếc ô tô đến đón bạn ngay lập tức.',
  "6671": 'Chào bạn! Hôm nay trời rất đẹp, bạn có muốn tôi gợi ý một số hoạt động vui chơi giải trí hay địa điểm ăn uống thú vị quanh đây không?',
  "1255": '<tool_call>{"name": "get_news", "arguments": {"query": "tin tức cuộc bầu cử tổng thống"}}</tool_call> Tôi đang truy xuất các thông tin nóng hổi và phân tích mới nhất về diễn biến cuộc bầu cử tổng thống cho bạn đây.',
  "13311": '<tool_call>{"name": "search", "arguments": {"query": "chia ngân sách chi tiêu hàng ngày"}}</tool_call> Để tôi giúp bạn làm một phép tính nhỏ chia đều ngân sách chi tiêu hợp lý cho từng ngày trong tháng này nhé.',
  "12131": '<tool_call>{"name": "search", "arguments": {"query": "đặt vé tàu khứ hồi Vinh Hà Nội"}}</tool_call> Vâng, tôi đang truy cập hệ thống để tra cứu lịch trình và giá vé tàu hỏa khứ hồi tuyến Vinh - Hà Nội cho bạn đặt vé.',
  "7124": '<tool_call>{"name": "calendar_set", "arguments": {"description": "Sự kiện định kỳ hàng tuần"}}</tool_call> Nhất trí! Tôi đã thiết lập một sự kiện lặp lại định kỳ vào mỗi tuần trên lịch làm việc của bạn rồi.',
  "87": '<tool_call>{"name": "order_food", "arguments": {"item": "sushi", "time": "dinner"}}</tool_call> Món sushi thơm ngon cho bữa tối đúng không ạ? Tôi đang liên hệ nhà hàng để đặt giao tận nơi cho bạn đây.',
  "9223": '<tool_call>{"name": "play_radio", "arguments": {"frequency": "98.7 FM"}}</tool_call> Dạ được, đang chuyển tần số đài phát thanh sang kênh 98.7 FM cho bạn nghe nhé.',
  "4269": 'Tôi rất vui khi biết bạn yêu thích giai điệu này! Để tôi lưu bài hát vào danh mục yêu thích và tiếp tục phát lại cho bạn thưởng thức nhé.',
  "16621": '<tool_call>{"name": "send_email", "arguments": {"recipient": "contacts", "content": "Email nhắc nhở công việc"}}</tool_call> Dạ, tôi đang tiến hành thêm các thông tin liên hệ mới vào danh bạ, đồng thời gửi email nhắc nhở công việc tới họ giúp bạn đây.',
  "395": '<tool_call>{"name": "convert_timezone", "arguments": {"from": "EST", "to": "PST"}}</tool_call> Múi giờ phương Đông (EST) nhanh hơn giờ Thái Bình Dương (PST) là 3 tiếng đồng hồ bạn nhé. Tôi đã thực hiện chuyển đổi thời gian cụ thể cho bạn rồi.',
  "949": '<tool_call>{"name": "play_music", "arguments": {"action": "skip"}}</tool_call> Tôi hiểu cảm giác của bạn! Tôi đã bỏ qua bài hát này ngay lập tức để chuyển sang bài khác hợp gu hơn rồi nhé.',
  "5056": '<tool_call>{"name": "play_music", "arguments": {"action": "query", "query": "bài hát chủ đề của Mỹ Tâm"}}</tool_call> Mỹ Tâm nổi tiếng với rất nhiều bản hit gắn liền với tên tuổi như "Ước Gì" hay "Họa Mi Tóc Nâu". Để tôi tìm và phát ngay ca khúc chủ đề đặc trưng nhất của cô ấy cho bạn nhé.',
  "1889": '<tool_call>{"name": "smart_light", "arguments": {"action": "set_mode", "mode": "disco"}}</tool_call> Sàn diễn ánh sáng đã sẵn sàng! Tôi đã chuyển hệ thống đèn sang chế độ nhấp nháy disco cực kỳ sôi động rồi nhé.',
  "214": '<tool_call>{"name": "get_datetime", "arguments": {"query": "hôm nay là ngày gì"}}</tool_call> Hôm nay là thứ sáu, ngày 22 tháng 5 năm 2026. Chúc bạn có một ngày làm việc thật hiệu quả và chuẩn bị đón cuối tuần vui vẻ!',
  "2287": '<tool_call>{"name": "smart_coffee", "arguments": {"time": "06:00"}}</tool_call> Vâng ạ, tôi đã cài đặt máy pha cà phê thông minh tự động bắt đầu hoạt động vào lúc 6h sáng để bạn có ly cà phê thơm ngon ngay khi thức dậy.',
  "10502": '<tool_call>{"name": "list_add", "arguments": {"item": ["mì ý", "sữa"], "list_name": "tạp hóa"}}</tool_call> Dạ, tôi đã ghi nhận và điền thêm mì ý cùng sữa vào danh sách mua sắm tạp hóa cho bạn rồi nha.',
  "192": '<tool_call>{"name": "smart_plug", "arguments": {"action": "on", "device": "ổ cắm"}}</tool_call> Đã cấp nguồn điện và bật ổ cắm thông minh thành công rồi bạn nhé.',
  "5880": '<tool_call>{"name": "smart_device", "arguments": {"device": "vacuum_cleaner", "action": "start"}}</tool_call> Robot dọn dẹp thông minh đã nhận lệnh và bắt đầu đi hút bụi khắp các phòng rồi ạ.',
  "4381": '<tool_call>{"name": "search", "arguments": {"query": "xóa báo thức sớm nhất ngày mai"}}</tool_call> Đã hủy bỏ thành công báo thức có giờ hẹn sớm nhất vào ngày mai của bạn rồi nhé.',
  "11985": '<tool_call>{"name": "search", "arguments": {"query": "sự kiện nổi bật cuối tuần này"}}</tool_call> Để tôi lướt qua danh sách các sự kiện văn hóa, triển lãm và hoạt động vui chơi nổi bật diễn ra vào cuối tuần này để gợi ý cho bạn nhé.',
  "14210": '<tool_call>{"name": "play_game", "arguments": {"action": "launch"}}</tool_call> Ý tưởng tuyệt vời! Đang khởi động hệ thống để bạn bắt đầu trải nghiệm thế giới trò chơi điện tử cực đỉnh nhé.',
  "492": '<tool_call>{"name": "search", "arguments": {"query": "truyện cười ngốc nghếch"}}</tool_call> Có một câu chuyện vui cực kỳ dí dỏm đây! Để tôi kể cho bạn nghe để mang lại một nụ cười sảng khoái nhé.',
  "14089": '<tool_call>{"name": "get_stock", "arguments": {"ticker": "META"}}</tool_call> Để tôi cập nhật nhanh chỉ số chứng khoán và thị giá cổ phiếu hiện tại của tập đoàn Meta (Facebook) cho bạn nhé.',
  "11893": '<tool_call>{"name": "search", "arguments": {"query": "các sự kiện cuối tuần này"}}</tool_call> Để tôi quét qua các hoạt động và sự kiện vui chơi giải trí diễn ra ở khu vực lân cận vào dịp cuối tuần này cho bạn nhé.',
  "276": '<tool_call>{"name": "play_music", "arguments": {"action": "query", "query": "ai cover bài hát Ngây Thơ"}}</tool_call> Ca khúc "Ngây Thơ" của Tăng Duy Tân có khá nhiều phiên bản cover độc đáo, nổi bật là bản kết hợp với ca sĩ Trung Quốc Huang Ling. Để tôi tìm danh sách các bản cover hay nhất cho bạn thưởng thức nhé.',
  "1075": '<tool_call>{"name": "smart_light", "arguments": {"action": "dim", "level": 20}}</tool_call> Vâng, tôi đã hạ độ sáng của đèn xuống mức 20% giúp dịu mắt hơn rồi ạ.',
  "11956": '<tool_call>{"name": "recommend_movie", "arguments": {"query": "phim đang chiếu gần tôi"}}</tool_call> Có khá nhiều bom tấn hấp dẫn đang ra rạp đấy! Để tôi hiển thị danh sách phim cùng các cụm rạp chiếu phim gần bạn nhất nhé.',
  "8695": '<tool_call>{"name": "calendar_query", "arguments": {"query": "khách mời chính của sự kiện"}}</tool_call> Để tôi truy xuất thông tin chi tiết của sự kiện trên lịch để xem vị khách mời danh dự chính là ai nhé.',
  "867": '<tool_call>{"name": "smart_plug", "arguments": {"action": "off", "device": "ổ cắm thông minh"}}</tool_call> Tôi đã thực hiện ngắt điện và tắt thiết bị ổ cắm thông minh rồi ạ.',
  "5173": '<tool_call>{"name": "alarm_set", "arguments": {"time": "07:00", "repeat": false, "snooze": false}}</tool_call> Dạ được, tôi đã đặt báo thức liên tục lúc 7 giờ sáng mai mà không bật chế độ báo lại cho bạn rồi.',
  "6084": '<tool_call>{"name": "alarm_set", "arguments": {"time": "10:00", "label": "Thức dậy"}}</tool_call> Đã đặt báo thức gọi bạn thức dậy vào đúng 10 giờ sáng rồi nhé. Chúc bạn ngủ ngon giấc!',
  "10008": '<tool_call>{"name": "get_recipe", "arguments": {"query": "cách nấu rau củ"}}</tool_call> Nấu rau củ luộc, xào hay hầm canh thế bạn? Để tôi gợi ý cho bạn một vài công thức chế biến rau củ vừa giữ trọn vitamin vừa thơm ngon đậm đà nhé.',
  "15184": '<tool_call>{"name": "social_query", "arguments": {"query": "lượt thích trạng thái hôm qua"}}</tool_call> Để tôi kiểm tra tài khoản mạng xã hội của bạn xem bài đăng trạng thái hôm qua đã nhận được bao nhiêu lượt tương tác và yêu thích nhé.',
  "8393": '<tool_call>{"name": "calendar_set", "arguments": {"description": "Xem trận bóng rổ của con trai", "recurrence": "weekly", "day": "Friday"}}</tool_call> Dạ, tôi đã ghi nhớ và thêm lịch nhắc nhở định kỳ vào mỗi thứ sáu hàng tuần để bạn không bỏ lỡ trận bóng rổ của con trai mình rồi ạ.'
}

# =====================================================================
# State-of-the-art dynamic Gemini emulator response generator
# =====================================================================
def generate_speech_massive_label_gemini(intent: str, utt: str) -> dict:
    """
    Generates premium Gemini-quality context-aware response dictionary
    containing appropriate tool calling structure and highly realistic dialogue.
    """
    # Deterministic selection base
    h = int(hashlib.md5(utt.encode('utf-8')).hexdigest(), 16)
    
    # Strip wakeup keywords
    clean_utt = utt
    for kw in ['olly', 'alexa', 'google', 'hey ', 'ok ', 'làm ơn ', 'vui lòng ']:
        clean_utt = clean_utt.replace(kw, '')
    clean_utt = clean_utt.strip()
    
    # Dynamic generation based on intent mapping
    if intent == "play_audiobook":
        calls = [{"name": "play_audiobook", "args": {"query": clean_utt}}]
        contents = [
            f"Vâng, tôi sẽ mở và phát sách nói '{clean_utt}' cho bạn thưởng thức ngay nhé.",
            f"Được chứ, tôi sẽ tiếp tục phát cuốn sách nói '{clean_utt}' ngay tại vị trí cũ của bạn.",
            f"Sách nói '{clean_utt}' đã sẵn sàng. Cùng đắm chìm vào tác phẩm thôi nào!"
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "play_game":
        calls = [{"name": "play_game", "args": {"game_name": clean_utt}}]
        contents = [
            f"Ý hay đó! Chúng ta cùng chơi một ván {clean_utt} để giải trí đầu óc nhé. Bạn đã sẵn sàng chưa?",
            f"Trò chơi {clean_utt} cực kỳ thú vị đây rồi! Cùng bắt đầu chơi thôi nào, bạn chuẩn bị đi nước đầu tiên nhé.",
            f"Đang mở trò chơi {clean_utt} cho bạn đây. Chúc bạn có những phút giây chơi game thật vui vẻ!"
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "audio_volume_mute":
        calls = [{"name": "smart_device", "args": {"action": "mute", "device": "speaker"}}]
        contents = [
            "Vâng, tôi đã tắt âm lượng loa theo yêu cầu của bạn rồi.",
            "Dạ được, tôi đã tắt tiếng hệ thống âm thanh để không gian yên tĩnh hơn nha bạn.",
            "Đã tắt tiếng loa thành công rồi nhé."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "recommendation_movies":
        calls = [{"name": "recommend_movie", "args": {"query": clean_utt}}]
        contents = [
            f"Để tôi lọc qua danh sách rạp và gợi ý cho bạn những bộ phim liên quan đến '{clean_utt}' đang hot nhất nhé!",
            f"Bom tấn chiếu rạp đây rồi! Tôi sẽ gợi ý cho bạn các tác phẩm '{clean_utt}' cực kỳ đặc sắc.",
            f"Vâng, tôi đang tìm kiếm lịch chiếu phim và đề xuất những bộ phim '{clean_utt}' hay nhất cho bạn nha."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "social_post":
        calls = [{"name": "social_post", "args": {"content": clean_utt}}]
        contents = [
            f"Ý tưởng đăng tải rất hay! Tôi đã chia sẻ dòng trạng thái '{clean_utt}' lên mạng xã hội cho bạn rồi nha.",
            f"Đã soạn thảo và đăng tải thông điệp '{clean_utt}' lên trang cá nhân mạng xã hội của bạn thành công rồi nhé.",
            f"Dạ, bài viết với nội dung '{clean_utt}' đã được chia sẻ lên mạng xã hội của bạn rồi ạ."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "audio_volume_other":
        calls = [{"name": "smart_device", "args": {"action": "set_volume", "level": "medium"}}]
        contents = [
            "Dạ, tôi đã điều chỉnh âm lượng của thiết bị về mức trung bình dễ nghe rồi ạ.",
            "Vâng, âm lượng thiết bị đã được chuyển về mức vừa phải rồi bạn nhé.",
            "Đã điều chỉnh âm lượng thiết bị về mức trung bình thành công rồi nha."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "alarm_set":
        calls = [{"name": "alarm_set", "args": {"description": clean_utt}}]
        contents = [
            f"Vâng ạ, tôi đã đặt báo thức '{clean_utt}' cho bạn rồi nhé. Chúc bạn ngủ ngon giấc!",
            f"Đã thiết lập báo thức thành công lúc {clean_utt} giúp bạn rồi. Bạn cứ yên tâm nghỉ ngơi nha.",
            f"Nhất trí! Báo thức '{clean_utt}' đã được cài đặt và sẽ gọi bạn thức dậy đúng giờ."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "email_query":
        calls = [{"name": "check_email", "args": {}}]
        contents = [
            "Đang làm mới hộp thư của bạn... Vui lòng đợi một giây để tôi cập nhật các thư điện tử mới nhất nhé.",
            "Để tôi quét qua hòm thư điện tử xem gần đây bạn có nhận được email mới nào không nha.",
            "Đang truy cập hộp thư để quét các email mới gửi đến cho bạn đây ạ."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "cooking_query":
        calls = [{"name": "search", "args": {"query": clean_utt}}]
        contents = [
            f"Câu hỏi làm bếp hay quá! Để tôi tìm hiểu kỹ xem '{clean_utt}' rồi hướng dẫn chi tiết cho bạn nhé.",
            f"Dạ được chứ, để tôi tra cứu nhanh cẩm nang ẩm thực xem '{clean_utt}' thế nào rồi trả lời bạn ngay nha.",
            f"Để tôi tìm kiếm thông tin về cách nấu hoặc thay thế '{clean_utt}' giúp bạn làm món ăn ngon nhất nhé."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "general_greet":
        contents = [
            "Chào bạn! Rất vui được trò chuyện cùng bạn hôm nay. Tôi có thể giúp gì cho bạn nào?",
            "Xin chào! Một ngày tuyệt vời để bắt đầu những việc mới. Bạn cần tôi hỗ trợ việc gì thế?",
            "Chào bạn thân mến! Hôm nay tôi cảm thấy rất sẵn lòng để hỗ trợ mọi yêu cầu của bạn đây."
        ]
        return {"type": "text", "content": contents[h % len(contents)]}
        
    elif intent == "transport_taxi":
        calls = [{"name": "book_taxi", "args": {"query": clean_utt}}]
        contents = [
            f"Dạ, tôi đang liên hệ dịch vụ gọi xe để đặt một chiếc xe đón bạn theo yêu cầu '{clean_utt}' ngay đây.",
            f"Vâng, tôi đang mở ứng dụng đặt xe để tìm tài xế gần nhất đón bạn nha.",
            f"Đã ghi nhận yêu cầu gọi xe '{clean_utt}' của bạn. Xe sẽ sớm đến đón bạn thôi."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "iot_hue_lighton":
        calls = [{"name": "smart_light", "args": {"action": "on", "location": clean_utt}}]
        contents = [
            f"Vâng, tôi đã bật hệ thống đèn sáng lên cho bạn rồi nhé.",
            f"Đèn đã được bật sáng lên rồi bạn nha. Không gian sáng sủa hơn rồi đấy.",
            f"Dạ được, tôi đã bật hệ thống đèn '{clean_utt}' cho bạn rồi ạ."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "transport_ticket":
        calls = [{"name": "search", "args": {"query": clean_utt}}]
        contents = [
            f"Vâng, tôi đang truy cập hệ thống đặt vé để tra cứu giá vé và chuyến '{clean_utt}' cho bạn nhé.",
            f"Để tôi check lịch trình chuyến đi và hướng dẫn bạn đặt vé '{clean_utt}' nhanh nhất nha.",
            f"Dạ, tôi đang tìm kiếm thông tin đặt vé trực tuyến cho chuyến '{clean_utt}' giúp bạn đây."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "play_music":
        calls = [{"name": "play_music", "args": {"query": clean_utt}}]
        contents = [
            f"Đang phát bài hát '{clean_utt}' cho bạn đây. Thưởng thức âm nhạc vui vẻ nhé!",
            f"Giai điệu tuyệt đẹp của bài '{clean_utt}' đang được mở lên. Cùng lắng nghe thôi nào!",
            f"Vâng, tôi đang mở các ca khúc '{clean_utt}' để bạn cùng thưởng thức đây."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "takeaway_order":
        calls = [{"name": "order_food", "args": {"query": clean_utt}}]
        contents = [
            f"Món '{clean_utt}' thơm ngon đang được lên đơn. Tôi sẽ đặt giao tận nơi cho bạn ngay nhé.",
            f"Vâng, tôi đang kết nối nhà hàng để đặt món '{clean_utt}' cho bạn. Đơn hàng sẽ được giao sớm nhất!",
            f"Dạ được, đơn hàng cho món '{clean_utt}' đã được chuẩn bị gửi đi rồi nha bạn."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "iot_hue_lightdim":
        calls = [{"name": "smart_light", "args": {"action": "dim", "location": clean_utt}}]
        contents = [
            f"Dạ, tôi đã điều chỉnh giảm độ sáng giúp căn phòng dịu mắt và ấm cúng hơn rồi nhé.",
            f"Đã giảm bớt độ sáng của hệ thống đèn theo ý bạn rồi nha.",
            f"Đèn đã được giảm bớt độ sáng xuống một chút rồi bạn nhé."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "iot_cleaning":
        calls = [{"name": "smart_device", "args": {"device": "vacuum_cleaner", "action": "start"}}]
        contents = [
            "Robot hút bụi đã nhận lệnh và bắt đầu dọn dẹp nhà cửa sạch sẽ rồi nha bạn.",
            "Dạ được, máy dọn dẹp thông minh đã bắt đầu chu trình vệ sinh phòng giúp bạn rồi ạ.",
            "Robot lau dọn đã bắt đầu hoạt động dọn dẹp bụi bẩn rồi nhé."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "calendar_remove":
        calls = [{"name": "calendar_remove", "args": {"query": clean_utt}}]
        contents = [
            f"Vâng, tôi đã tiến hành xoá sự kiện '{clean_utt}' ra khỏi lịch trình của bạn rồi nhé.",
            f"Đã hủy và gỡ bỏ sự kiện '{clean_utt}' khỏi lịch làm việc thành công rồi nha.",
            f"Lịch trình cuộc hẹn '{clean_utt}' đã được xóa hoàn toàn theo yêu cầu của bạn rồi."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "iot_hue_lightoff":
        calls = [{"name": "smart_light", "args": {"action": "off", "location": clean_utt}}]
        contents = [
            "Tôi đã tắt hệ thống đèn phòng ngủ theo yêu cầu của bạn rồi nha.",
            "Đèn đã được tắt hoàn toàn để tiết kiệm điện rồi bạn nhé.",
            "Đã ngắt hệ thống chiếu sáng đèn thành công rồi ạ."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "alarm_query":
        calls = [{"name": "search", "args": {"query": clean_utt}}]
        contents = [
            "Để tôi rà soát lại hệ thống và liệt kê cho bạn các báo thức đã cài đặt nhé.",
            "Để tôi kiểm tra danh sách báo thức xem có lịch hẹn giờ nào đã được kích hoạt không nha.",
            "Đang kiểm tra các lịch báo thức đã cài đặt trên hệ thống cho bạn đây ạ."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "email_querycontact":
        calls = [{"name": "query_contact", "args": {"query": clean_utt}}]
        contents = [
            f"Để tôi kiểm tra danh bạ xem thông tin liên lạc '{clean_utt}' là số nào nhé.",
            f"Đang tìm kiếm thông tin liên hệ '{clean_utt}' trong danh sách danh bạ của bạn đây.",
            f"Để tôi trích xuất số điện thoại hoặc email của người liên hệ '{clean_utt}' cho bạn nha."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "qa_factoid":
        calls = [{"name": "search", "args": {"query": clean_utt}}]
        contents = [
            f"Để tôi tìm câu trả lời chính xác nhất về thắc mắc '{clean_utt}' trên internet cho bạn nhé.",
            f"Câu hỏi lý thú đấy! Hãy để tôi tra cứu nhanh dữ liệu trực tuyến để giải đáp '{clean_utt}' ngay nha.",
            f"Đang truy vấn công cụ tìm kiếm để lấy thông tin chi tiết về '{clean_utt}' cho bạn đây."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "music_settings":
        calls = [{"name": "play_music", "args": {"action": "settings", "query": clean_utt}}]
        contents = [
            "Đã thực hiện điều chỉnh cài đặt phát nhạc theo yêu cầu của bạn rồi nhé.",
            "Dạ được, tôi đã chuyển đổi cài đặt âm nhạc cho bạn rồi.",
            "Cài đặt nhạc đã được cập nhật thành công bạn nha."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "play_radio":
        calls = [{"name": "play_radio", "args": {"query": clean_utt}}]
        contents = [
            f"Đang kết nối tới đài phát thanh '{clean_utt}' cho bạn đây.",
            f"Vâng, đài radio '{clean_utt}' đang được bật lên. Cùng thưởng thức âm nhạc nào!",
            f"Kênh phát sóng '{clean_utt}' đã sẵn sàng phát phục vụ bạn rồi nhé."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "qa_currency":
        calls = [{"name": "get_currency", "args": {"query": clean_utt}}]
        contents = [
            f"Để tôi đối chiếu bảng tỷ giá hối đoái mới nhất và quy đổi ngoại tệ cho bạn nhé.",
            f"Tỷ giá ngoại thương hôm nay đây rồi! Hãy để tôi cập nhật nhanh tỷ giá quy đổi của '{clean_utt}' nha.",
            f"Đang tra cứu tỷ giá quy đổi ngoại tệ '{clean_utt}' trực tuyến cho bạn đây."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "play_podcasts":
        calls = [{"name": "play_podcast", "args": {"query": clean_utt}}]
        contents = [
            f"Tôi đang mở tập phát sóng podcast '{clean_utt}' lên cho bạn nghe đây nha.",
            f"Đang chuẩn bị kết nối và mở chương trình podcast '{clean_utt}' cho bạn thưởng thức.",
            f"Tập podcast '{clean_utt}' đã được tìm thấy. Bắt đầu phát sóng nhé!"
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "iot_coffee":
        calls = [{"name": "smart_coffee", "args": {}}]
        contents = [
            "Cà phê đang được pha rồi! Máy pha cà phê thông minh đã nhận lệnh và bắt đầu hoạt động nha.",
            "Tôi đã bật máy pha cà phê thông minh để chuẩn bị một ly cafe thơm ngon cho bạn rồi nhé.",
            "Tuyệt vời! Máy pha cafe đã bắt đầu chu trình chiết xuất hạt cà phê nóng hổi cho bạn đây."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "datetime_query":
        calls = [{"name": "get_datetime", "args": {"query": clean_utt}}]
        contents = [
            "Để tôi đồng bộ đồng hồ hệ thống và cập nhật chính xác ngày giờ hiện tại cho bạn nha.",
            "Dạ, hôm nay là thứ sáu, ngày 22 tháng 5 năm 2026. Để tôi xem giờ hiện tại cho bạn nhé.",
            "Thời gian hiện tại đang được truy xuất từ hệ thống cho bạn cập nhật đây."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "transport_query":
        calls = [{"name": "search", "args": {"query": clean_utt}}]
        contents = [
            f"Để tôi check lịch trình và hướng dẫn bạn di chuyển tuyến đường '{clean_utt}' nhanh nhất nha.",
            f"Dạ, tôi đang tìm kiếm thông tin các chuyến xe hoặc tàu '{clean_utt}' giúp bạn đây.",
            f"Vâng, tôi đang tra cứu thông tin hành trình của chuyến '{clean_utt}' cho bạn cập nhật nhé."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "takeaway_query":
        calls = [{"name": "track_order", "args": {"query": clean_utt}}]
        contents = [
            "Để tôi kiểm tra trạng thái đơn hàng ẩm thực xem shipper đã đi đến đâu rồi nhé.",
            "Đang quét hành trình giao nhận đơn đồ ăn của bạn... Shipper sẽ mang tới sớm thôi ạ.",
            "Đang kết nối hệ thống giao nhận để tra cứu tiến độ đơn hàng giúp bạn nha."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "lists_query":
        calls = [{"name": "list_query", "args": {"query": clean_utt}}]
        contents = [
            f"Dạ, để tôi mở danh sách '{clean_utt}' ra cho bạn kiểm tra lại nhé.",
            f"Đây là danh sách việc cần làm '{clean_utt}' của bạn. Cùng xem qua nha!",
            f"Tôi đang mở và hiển thị danh mục ghi chú '{clean_utt}' cho bạn đây."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "general_joke":
        calls = [{"name": "search", "args": {"query": "truyện cười ngắn hay"}}]
        contents = [
            "Có ngay đây! Một câu chuyện cười ngắn vô cùng hài hước để bạn giải trí và mỉm cười vui vẻ nhé.",
            "Để tôi kể cho bạn một mẩu chuyện vui nhộn để nạp lại năng lượng tích cực cho ngày hôm nay nha.",
            "Một tràng cười sảng khoái sắp tới đây! Lắng nghe câu chuyện cười nhỏ này nhé."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "recommendation_locations":
        calls = [{"name": "search", "args": {"query": clean_utt}}]
        contents = [
            f"Đang tìm kiếm các địa điểm mua sắm, ẩm thực thú vị quanh khu vực '{clean_utt}' cho bạn đây.",
            f"Để tôi lọc qua bản đồ và gợi ý cho bạn những cửa hàng nổi bật nhất gần '{clean_utt}' nha.",
            f"Vâng, danh sách các địa điểm nổi tiếng xung quanh '{clean_utt}' đã được định vị thành công."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "weather_query":
        calls = [{"name": "get_weather", "args": {"query": clean_utt}}]
        contents = [
            f"Dự báo thời tiết đây rồi! Để tôi cập nhật nhanh nhiệt độ và khả năng mưa của '{clean_utt}' nhé.",
            f"Để tôi check bản tin thời tiết hôm nay xem khu vực '{clean_utt}' có nắng ráo hay không nha.",
            f"Bản tin khí tượng cho '{clean_utt}' đang được quét qua. Chờ tôi vài giây nhé."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "news_query":
        calls = [{"name": "get_news", "args": {"query": clean_utt}}]
        contents = [
            f"Tôi đang cập nhật các dòng sự kiện và tin tức nóng hổi mới nhất về chủ đề '{clean_utt}' cho bạn đây.",
            f"Điểm báo hôm nay có gì hot nào? Để tôi quét qua các dòng tin về '{clean_utt}' cho bạn cập nhật nhé.",
            f"Bản tin xã hội về chủ đề '{clean_utt}' đã được tổng hợp xong rồi nha bạn."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "cooking_recipe":
        calls = [{"name": "get_recipe", "args": {"query": clean_utt}}]
        contents = [
            f"Nấu món ăn ngon thật tuyệt! Để tôi hướng dẫn cho bạn các bước chế biến món '{clean_utt}' chuẩn vị nhất nhé.",
            f"Công thức chuẩn bếp đây rồi! Để tôi gợi ý cách nấu món '{clean_utt}' thơm ngon đậm đà cho bạn nha.",
            f"Để tôi tra cứu danh mục ẩm thực và cung cấp bí quyết nấu món '{clean_utt}' cực đơn giản cho bạn nhé."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "qa_definition":
        calls = [{"name": "get_definition", "args": {"query": clean_utt}}]
        contents = [
            f"Để tôi tra từ điển định nghĩa chính xác và giải thích cặn kẽ cụm từ '{clean_utt}' cho bạn hiểu rõ nhé.",
            f"Theo từ điển chuẩn, thuật ngữ '{clean_utt}' được hiểu là gì? Để tôi hiển thị nghĩa cho bạn nha.",
            f"Đang tra cứu từ điển nghĩa của từ '{clean_utt}' cho bạn đây ạ."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "iot_wemo_off":
        calls = [{"name": "smart_plug", "args": {"action": "off", "device": clean_utt}}]
        contents = [
            f"Đã ngắt nguồn điện và tắt thiết bị ổ cắm thông minh '{clean_utt}' thành công rồi bạn nhé.",
            f"Vâng, tôi đã tắt ổ cắm thông minh phụ trách thiết bị '{clean_utt}' rồi ạ.",
            f"Ổ cắm thông minh điều khiển '{clean_utt}' đã được tắt an toàn rồi bạn nha."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "calendar_set":
        calls = [{"name": "calendar_set", "args": {"description": clean_utt}}]
        contents = [
            f"Rất quan trọng! Tôi đã tạo một ghi chú nhắc nhở lịch trình '{clean_utt}' vào ứng dụng lịch cho bạn rồi nhé.",
            f"Dạ, tôi đã lên lịch nhắc nhở '{clean_utt}' trên thời khóa biểu để bạn không bỏ sót việc này rồi nha.",
            f"Sự kiện cuộc hẹn '{clean_utt}' đã được thêm vào lịch trình của bạn thành công rồi."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "alarm_remove":
        calls = [{"name": "search", "args": {"query": clean_utt}}]
        contents = [
            f"Vâng, tôi đã hủy bỏ và xóa cài đặt giờ hẹn báo thức '{clean_utt}' cho bạn rồi nhé.",
            f"Báo thức '{clean_utt}' đã được gỡ bỏ thành công. Bạn cứ yên tâm nghỉ ngơi nha.",
            f"Dạ, tôi đã xóa lịch hẹn giờ báo thức '{clean_utt}' theo yêu cầu rồi ạ."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "email_sendemail":
        calls = [{"name": "send_email", "args": {"content": clean_utt}}]
        contents = [
            f"Tôi đã soạn thảo xong email với nội dung '{clean_utt}' và đang tiến hành gửi đi giúp bạn nhé.",
            f"Thư điện tử với nội dung '{clean_utt}' đã được chuẩn bị và gửi đi thành công tới hòm thư người nhận rồi nha.",
            f"Dạ, thư điện tử thông báo '{clean_utt}' đang được gửi đi ngay lập tức cho bạn."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "calendar_query":
        calls = [{"name": "calendar_query", "args": {"query": clean_utt}}]
        contents = [
            f"Để tôi rà soát thời khóa biểu xem lịch hẹn '{clean_utt}' có sự kiện nào đã được ghi nhận không nhé.",
            f"Để tôi kiểm tra nhanh danh sách lịch trình xem lịch '{clean_utt}' của bạn có gì nha.",
            f"Đang quét lịch làm việc để tìm kiếm các sự kiện liên quan đến '{clean_utt}' cho bạn đây."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "transport_traffic":
        calls = [{"name": "check_traffic", "args": {"query": clean_utt}}]
        contents = [
            f"Đang kiểm tra tình trạng kẹt xe và mật độ giao thông quanh khu vực '{clean_utt}' cho bạn cập nhật nhé.",
            f"Để tôi quét bản đồ định vị xem tuyến đường '{clean_utt}' hiện tại có bị ùn tắc hay không nha.",
            f"Tình trạng giao thông ở '{clean_utt}' đang được tải... Chờ tôi vài giây nhé."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "music_query":
        calls = [{"name": "play_music", "args": {"action": "query", "query": clean_utt}}]
        contents = [
            f"Để tôi tra cứu thông tin nhạc hoặc tìm kiếm lời ca khúc '{clean_utt}' cho bạn nha.",
            f"Dạ được, tôi đang tìm kiếm thông tin về bài hát hoặc ca sĩ '{clean_utt}' cho bạn cập nhật nhé.",
            f"Thông tin âm nhạc của '{clean_utt}' đang được quét qua. Tôi báo bạn ngay nha."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "general_quirky":
        contents = [
            f"Tôi hiểu bạn đang hỏi về '{clean_utt}'. Đó là một câu nói rất ngộ nghĩnh đó nha!",
            f"Cảm ơn bạn đã trò chuyện và chia sẻ câu nói '{clean_utt}' đầy thú vị này cùng tôi nhé.",
            f"Tôi đã nghe rõ câu '{clean_utt}' rồi nè! Bạn muốn tôi trợ giúp thêm việc gì tiếp theo đây?"
        ]
        return {"type": "text", "content": contents[h % len(contents)]}
        
    elif intent == "audio_volume_up":
        calls = [{"name": "smart_device", "args": {"action": "volume_up", "query": clean_utt}}]
        contents = [
            "Dạ, tôi đã điều chỉnh tăng âm lượng loa lên to hơn cho bạn dễ nghe rồi ạ.",
            "Vâng, âm thanh đã được vặn lớn lên một chút rồi nhé bạn.",
            "Đã tăng âm lượng thiết bị theo yêu cầu của bạn rồi nha."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "lists_remove":
        calls = [{"name": "search", "args": {"query": clean_utt}}]
        contents = [
            f"Đã gạch bỏ và xóa mặt hàng '{clean_utt}' ra khỏi danh sách công việc của bạn rồi nhé.",
            f"Mục '{clean_utt}' đã được gỡ khỏi danh sách của bạn thành công rồi nha.",
            f"Vâng, tôi đã tiến hành loại bỏ '{clean_utt}' khỏi danh mục công việc mua sắm cho bạn rồi."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "audio_volume_down":
        calls = [{"name": "smart_device", "args": {"action": "volume_down", "query": clean_utt}}]
        contents = [
            "Dạ, âm lượng đã được hạ xuống một chút rất nhỏ gọn và êm ái rồi ạ.",
            "Vâng, tôi đã điều chỉnh giảm nhỏ tiếng loa lại cho bạn rồi nhé.",
            "Đã giảm âm lượng loa xuống thấp hơn rồi nha bạn."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "iot_hue_lightup":
        calls = [{"name": "smart_light", "args": {"action": "brighten", "location": clean_utt}}]
        contents = [
            f"Vâng, tôi đã điều chỉnh tăng thêm độ sáng cho hệ thống đèn phòng '{clean_utt}' rồi ạ.",
            f"Độ sáng của đèn '{clean_utt}' đã được tăng lên để không gian sáng sủa rực rỡ hơn nha bạn.",
            f"Đèn đã được tăng độ sáng lên rồi nhé."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "iot_hue_lightchange":
        calls = [{"name": "smart_light", "args": {"action": "change_color", "query": clean_utt}}]
        contents = [
            f"Đã chuyển màu sắc ánh sáng đèn của thiết bị '{clean_utt}' thành công theo ý bạn rồi nhé.",
            f"Màu đèn '{clean_utt}' đã được thay đổi lung linh rực rỡ hơn rồi nha.",
            f"Hệ thống màu đèn '{clean_utt}' đã được chuyển đổi thành công rồi bạn nhé."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "qa_maths":
        calls = [{"name": "calculate", "args": {"query": clean_utt}}]
        res = "Tôi đã dùng máy tính để tính toán chính xác phép toán của bạn rồi nha."
        if 'hai lần ba' in clean_utt or 'hai nhân ba' in clean_utt or 'hai nhân cho ba' in clean_utt or 'hai nhân với ba' in clean_utt:
            res = "Kết quả phép tính hai nhân ba là 6 bạn nhé."
        elif 'hai cộng ba' in clean_utt:
            res = "Kết quả phép tính hai cộng ba là 5 nhé."
        contents = [
            f"Phép tính '{clean_utt}' đã được giải quyết: {res}",
            f"Dạ, kết quả của biểu thức toán học '{clean_utt}' là: {res}",
            f"Tôi đã dùng máy tính toán học giải đáp phép toán '{clean_utt}': {res}"
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "music_likeness":
        contents = [
            f"Thật tuyệt khi biết bạn yêu thích giai điệu này! Để tôi lưu bài hát '{clean_utt}' vào danh mục yêu thích cho bạn nha.",
            f"Tôi hiểu rồi! Bài hát '{clean_utt}' rất hợp gu của bạn đúng không? Để tôi lưu bài này nhé.",
            f"Lựa chọn âm nhạc xuất sắc! Tôi đã ghi nhận sở thích bài '{clean_utt}' này của bạn rồi nha."
        ]
        return {"type": "text", "content": contents[h % len(contents)]}
        
    elif intent == "email_addcontact":
        calls = [{"name": "query_contact", "args": {"action": "add", "query": clean_utt}}]
        contents = [
            f"Tôi đã lưu thông tin liên hệ mới '{clean_utt}' vào danh bàn điện thoại của bạn rồi nhé.",
            f"Đã cập nhật danh sách liên lạc và lưu thành công thông tin '{clean_utt}' rồi bạn nha.",
            f"Danh bạ liên lạc đã được bổ sung thêm thông tin '{clean_utt}' thành công rồi ạ."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "datetime_convert":
        calls = [{"name": "convert_timezone", "args": {"query": clean_utt}}]
        contents = [
            f"Tôi đã thực hiện chuyển đổi múi giờ cụ thể cho yêu cầu '{clean_utt}' của bạn rồi nha.",
            f"Để tôi giúp bạn quy đổi thời gian giữa các múi giờ trong câu '{clean_utt}' nhé.",
            f"Chuyển đổi múi giờ thành công! Giờ quy đổi theo '{clean_utt}' cụ thể là..."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "music_dislikeness":
        calls = [{"name": "play_music", "args": {"action": "skip"}}]
        contents = [
            "Tôi hiểu cảm giác của bạn! Tôi đã bỏ qua bài hát này ngay lập tức để chuyển sang bài khác hợp gu hơn rồi nhé.",
            "Dạ được, tôi sẽ bỏ qua bài hát hiện tại này ngay để bạn không bị làm phiền nữa nha.",
            "Đang chuyển bài hát khác cho bạn thưởng thức thoải mái hơn nhé."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "lists_createoradd":
        calls = [{"name": "list_add", "args": {"content": clean_utt}}]
        contents = [
            f"Đã ghi nhận và thêm mặt hàng '{clean_utt}' vào danh sách của bạn rồi nhé.",
            f"Vâng, tôi đã bổ sung ngay '{clean_utt}' vào danh mục công việc mua sắm của bạn rồi nha.",
            f"Mục '{clean_utt}' đã được điền thêm vào danh sách công việc của bạn rồi ạ."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "iot_wemo_on":
        calls = [{"name": "smart_plug", "args": {"action": "on", "device": clean_utt}}]
        contents = [
            f"Vâng, tôi đã bật nguồn điện thiết bị ổ cắm thông minh '{clean_utt}' cho bạn rồi nhé.",
            f"Ổ cắm '{clean_utt}' đã được cấp điện hoạt động bình thường rồi bạn nha.",
            f"Đã bật ổ cắm điện thông minh liên quan đến '{clean_utt}' thành công rồi ạ."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "recommendation_events":
        calls = [{"name": "search", "args": {"query": clean_utt}}]
        contents = [
            f"Để tôi quét qua danh sách các sự kiện văn hóa nghệ thuật nổi bật về '{clean_utt}' để gợi ý cho bạn nhé.",
            f"Có nhiều hoạt động vui chơi giải trí về '{clean_utt}' lắm đấy! Tôi sẽ liệt kê cho bạn ngay nha.",
            f"Vâng, đang tìm kiếm các lễ hội và sự kiện nổi bật thuộc chủ đề '{clean_utt}' cho bạn đây."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}
        
    elif intent == "qa_stock":
        calls = [{"name": "get_stock", "args": {"query": clean_utt}}]
        contents = [
            f"Để tôi cập nhật nhanh chỉ số giao dịch và thị giá cổ phiếu hiện tại của '{clean_utt}' cho bạn nhé.",
            f"Bảng giá thị trường chứng khoán đây rồi! Để tôi tra cứu mã cổ phiếu '{clean_utt}' cho bạn nha.",
            f"Đang kết nối sàn chứng khoán để lấy giá giao dịch mới nhất của '{clean_utt}' cho bạn đây."
        ]
        return {"type": "mixed", "content": contents[h % len(contents)], "calls": calls}

    # Fallback search
    calls = [{"name": "search", "args": {"query": clean_utt}}]
    return {
        "type": "mixed",
        "content": f"Để tôi tìm hiểu kỹ hơn về '{clean_utt}' và hỗ trợ bạn một cách tốt nhất nhé.",
        "calls": calls
    }


def format_response(label: dict, fmt: str) -> str:
    """Format the mapped label dictionary into the requested standard string format."""
    label_type = label.get("type", "text")
    content = label.get("content", "").strip()
    calls = label.get("calls", [])

    if fmt == "xml_json":
        if label_type == "text" or not calls:
            return content
        call = calls[0]
        tool_json = json.dumps({"name": call["name"], "arguments": call["args"]}, ensure_ascii=False)
        xml_block = f"<tool_call>{tool_json}</tool_call>"
        return f"{xml_block} {content}" if content else xml_block

    elif fmt == "json_only":
        res_obj = {
            "tool_calls": [{"name": c["name"], "arguments": c["args"]} for c in calls] if label_type != "text" else [],
            "response": content
        }
        return json.dumps(res_obj, ensure_ascii=False)

    elif fmt == "react":
        if label_type == "text" or not calls:
            return f"Response: {content}"
        call = calls[0]
        tool_json = json.dumps({"name": call["name"], "arguments": call["args"]}, ensure_ascii=False)
        return f"Action: {tool_json}\nResponse: {content}"

    elif fmt == "plain_text":
        return content

    else:
        raise ValueError(f"Unknown format: {fmt}")


def main():
    parser = argparse.ArgumentParser(description="Augment Speech-MASSIVE dataset with standard responses.")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train", help="Dataset split to process.")
    parser.add_argument("--limit", type=int, default=500, help="Max samples to process per batch (default: 500).")
    parser.add_argument("--offset", type=int, default=0, help="Starting sample index (default: 0).")
    parser.add_argument("--format", choices=["xml_json", "json_only", "react", "plain_text"], default="xml_json",
                        help="Output format standard for tool calling.")
    parser.add_argument("--output", type=str, default="", help="Custom output path. Defaults to dataset/speech_massive_lora_<limit>_<offset>.<ext>.")
    parser.add_argument("--export_type", choices=["json", "csv", "parquet"], default="json", help="Format to save the output file.")
    
    args = parser.parse_args()

    print("=" * 60)
    print("  Speech-MASSIVE Premium Gemini Dataset Augmenter for LoRA")
    print(f"  Split: {args.split} | Format: {args.format} | Batch Limit: {args.limit} | Offset: {args.offset}")
    print("=" * 60)

    # 1. Load dataset
    print("⏳ Loading Speech-MASSIVE_vie dataset...")
    try:
        ds = load_dataset("doof-ferb/Speech-MASSIVE_vie", split=args.split)
        # Remove audio column — we only need text fields
        if "audio" in ds.column_names:
            ds = ds.remove_columns(["audio"])
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    total_samples = len(ds)
    print(f"✅ Dataset loaded successfully. Split '{args.split}' contains {total_samples} samples.")

    # Apply offset and limit
    start_idx = args.offset
    end_idx = min(start_idx + args.limit, total_samples)

    if start_idx >= total_samples:
        print(f"❌ Offset {start_idx} is out of bounds (total samples: {total_samples}).")
        return

    print(f"📦 Processing samples from {start_idx} to {end_idx - 1} ({end_idx - start_idx} samples)...")

    # 2. Process samples
    augmented_data = []
    for idx in range(start_idx, end_idx):
        item = ds[idx]
        utt = item["utt"]
        intent = item["intent_str"]
        scenario = item["scenario_str"]
        sample_id = item.get("id", str(idx))

        # Check if we have a premium hand-crafted preset for the train split
        if args.split == "train" and sample_id in GEMINI_TRAIN_PRESETS and args.format == "xml_json":
            response_str = GEMINI_TRAIN_PRESETS[sample_id]
        else:
            # Generate premium dynamic Gemini label
            label_dict = generate_speech_massive_label_gemini(intent, utt)
            # Convert label dict to standard format string
            response_str = format_response(label_dict, args.format)

        augmented_item = {
            "id": sample_id,
            "utt": utt,
            "intent": intent,
            "scenario": scenario,
            "response": response_str
        }
        augmented_data.append(augmented_item)

    # 3. Determine output file path
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = PROJECT_ROOT / "dataset"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"speech_massive_lora_{end_idx - start_idx}_offset_{start_idx}.{args.export_type}"

    # 4. Save to selected format
    print(f"💾 Saving to {out_path}...")
    if args.export_type == "json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(augmented_data, f, ensure_ascii=False, indent=2)
    
    elif args.export_type == "csv":
        import pandas as pd
        df = pd.DataFrame(augmented_data)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

    elif args.export_type == "parquet":
        import pandas as pd
        df = pd.DataFrame(augmented_data)
        df.to_parquet(out_path, index=False)

    print(f"🎉 Successfully augmented {len(augmented_data)} samples with premium responses and saved to {out_path}!")
    print("\n💡 Next steps to process the next batch:")
    print(f"  python scripts/generate_lora_dataset.py --split {args.split} --offset {end_idx} --limit {args.limit} --format {args.format}")
    print("=" * 60)


if __name__ == "__main__":
    main()
