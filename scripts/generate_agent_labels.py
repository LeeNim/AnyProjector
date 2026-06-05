"""
Generate agent labels for Phase 3 training.

Speech-MASSIVE: intent_str → tool_call template mapping
ViMD: transcript → response/tool_call generation (heuristic + template)

Output format (v0.4.1 LLM-Agnostic):
  {"type": "text",      "content": "..."}
  {"type": "tool_call",  "calls": [{"name": "...", "args": {...}}]}
  {"type": "mixed",      "content": "...", "calls": [{"name": "...", "args": {...}}]}
"""
import json
import re
import random
import hashlib
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

random.seed(42)

# ============================================================
# TOOL DEFINITIONS — comprehensive assistant tools
# ============================================================
TOOLS = {
    # --- Calendar & Reminders ---
    "calendar_query": {"description": "Truy vấn lịch, xem sự kiện"},
    "calendar_set": {"description": "Tạo sự kiện hoặc nhắc nhở"},
    "calendar_remove": {"description": "Xóa sự kiện khỏi lịch"},
    "alarm_set": {"description": "Đặt báo thức"},
    
    # --- Communication ---
    "send_email": {"description": "Gửi email"},
    "check_email": {"description": "Kiểm tra hộp thư"},
    "query_contact": {"description": "Tìm thông tin liên hệ"},
    "send_message": {"description": "Gửi tin nhắn"},
    "make_call": {"description": "Gọi điện thoại"},
    "social_post": {"description": "Đăng bài lên mạng xã hội"},
    "social_query": {"description": "Xem bản tin mạng xã hội"},
    
    # --- Search & QA ---
    "search": {"description": "Tìm kiếm thông tin trên internet"},
    "get_definition": {"description": "Tra nghĩa từ điển"},
    "get_stock": {"description": "Tra giá cổ phiếu"},
    "get_currency": {"description": "Tra tỷ giá ngoại tệ"},
    "get_news": {"description": "Đọc tin tức mới nhất"},
    
    # --- Weather & Time ---
    "get_weather": {"description": "Xem thời tiết"},
    "get_datetime": {"description": "Xem giờ/ngày hiện tại"},
    "convert_timezone": {"description": "Chuyển đổi múi giờ"},
    
    # --- Media & Entertainment ---
    "play_music": {"description": "Phát nhạc"},
    "play_podcast": {"description": "Phát podcast"},
    "play_radio": {"description": "Mở radio"},
    "play_audiobook": {"description": "Phát sách nói"},
    "play_video": {"description": "Phát video"},
    "recommend_movie": {"description": "Gợi ý phim"},
    
    # --- IoT & Smart Home ---
    "smart_light": {"description": "Điều khiển đèn thông minh"},
    "smart_plug": {"description": "Bật/tắt ổ cắm thông minh"},
    "smart_coffee": {"description": "Pha cà phê tự động"},
    "smart_device": {"description": "Điều khiển thiết bị IoT"},
    
    # --- Productivity ---
    "create_note": {"description": "Tạo ghi chú"},
    "list_query": {"description": "Xem danh sách"},
    "list_add": {"description": "Thêm vào danh sách"},
    "translate": {"description": "Dịch văn bản"},
    "calculate": {"description": "Tính toán"},
    "set_timer": {"description": "Đặt hẹn giờ"},
    
    # --- Food & Transport ---
    "order_food": {"description": "Đặt đồ ăn"},
    "track_order": {"description": "Theo dõi đơn hàng"},
    "book_taxi": {"description": "Đặt taxi"},
    "check_traffic": {"description": "Kiểm tra giao thông"},
    "get_recipe": {"description": "Tìm công thức nấu ăn"},
}

# ============================================================
# SPEECH-MASSIVE: intent → response template mapping
# ============================================================
INTENT_MAP = {
    # Calendar
    "calendar_query": lambda utt: {
        "type": "mixed",
        "content": "Để tôi kiểm tra lịch cho bạn.",
        "calls": [{"name": "calendar_query", "args": {"query": utt}}]
    },
    "calendar_set": lambda utt: {
        "type": "mixed",
        "content": "Vâng, tôi sẽ tạo nhắc nhở cho bạn.",
        "calls": [{"name": "calendar_set", "args": {"description": utt}}]
    },
    "calendar_remove": lambda utt: {
        "type": "mixed",
        "content": "Tôi sẽ xóa sự kiện này.",
        "calls": [{"name": "calendar_remove", "args": {"query": utt}}]
    },
    
    # Alarm
    "alarm_set": lambda utt: {
        "type": "mixed",
        "content": "Đã đặt báo thức cho bạn.",
        "calls": [{"name": "alarm_set", "args": {"description": utt}}]
    },
    
    # Email
    "email_sendemail": lambda utt: {
        "type": "mixed",
        "content": "Tôi sẽ gửi email cho bạn.",
        "calls": [{"name": "send_email", "args": {"content": utt}}]
    },
    "email_query": lambda utt: {
        "type": "mixed",
        "content": "Để tôi kiểm tra hộp thư.",
        "calls": [{"name": "check_email", "args": {}}]
    },
    "email_querycontact": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "query_contact", "args": {"query": utt}}]
    },
    "email_addcontact": lambda utt: {
        "type": "mixed",
        "content": "Tôi sẽ lưu thông tin liên hệ.",
        "calls": [{"name": "query_contact", "args": {"action": "add", "query": utt}}]
    },
    
    # Weather
    "weather_query": lambda utt: {
        "type": "mixed",
        "content": "Để tôi kiểm tra thời tiết.",
        "calls": [{"name": "get_weather", "args": {"query": utt}}]
    },
    
    # DateTime
    "datetime_query": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "get_datetime", "args": {"query": utt}}]
    },
    "datetime_convert": lambda utt: {
        "type": "mixed",
        "content": "Tôi sẽ chuyển đổi múi giờ cho bạn.",
        "calls": [{"name": "convert_timezone", "args": {"query": utt}}]
    },
    
    # Media
    "play_music": lambda utt: {
        "type": "mixed",
        "content": "Đang phát nhạc cho bạn.",
        "calls": [{"name": "play_music", "args": {"query": utt}}]
    },
    "play_podcasts": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "play_podcast", "args": {"query": utt}}]
    },
    "play_radio": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "play_radio", "args": {"query": utt}}]
    },
    "play_audiobook": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "play_audiobook", "args": {"query": utt}}]
    },
    "play_game": lambda utt: {
        "type": "mixed",
        "content": "Đang mở cho bạn.",
        "calls": [{"name": "play_video", "args": {"query": utt}}]
    },
    "music_query": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "play_music", "args": {"action": "query", "query": utt}}]
    },
    "music_likeness": lambda utt: {
        "type": "text",
        "content": "Tôi hiểu rồi! Bạn thích nghe nhạc. Bạn muốn tôi phát nhạc gì không?"
    },
    
    # QA
    "qa_factoid": lambda utt: {
        "type": "mixed",
        "content": "Để tôi tìm thông tin cho bạn.",
        "calls": [{"name": "search", "args": {"query": utt}}]
    },
    "qa_definition": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "get_definition", "args": {"query": utt}}]
    },
    "qa_currency": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "get_currency", "args": {"query": utt}}]
    },
    "qa_stock": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "get_stock", "args": {"query": utt}}]
    },
    
    # News
    "news_query": lambda utt: {
        "type": "mixed",
        "content": "Đây là tin tức mới nhất.",
        "calls": [{"name": "get_news", "args": {"query": utt}}]
    },
    
    # Social
    "social_post": lambda utt: {
        "type": "mixed",
        "content": "Tôi sẽ đăng bài cho bạn.",
        "calls": [{"name": "social_post", "args": {"content": utt}}]
    },
    "social_query": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "social_query", "args": {}}]
    },
    
    # Lists
    "lists_query": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "list_query", "args": {"query": utt}}]
    },
    "lists_createoradd": lambda utt: {
        "type": "mixed",
        "content": "Đã thêm vào danh sách.",
        "calls": [{"name": "list_add", "args": {"content": utt}}]
    },
    
    # IoT
    "iot_wemo_on": lambda utt: {
        "type": "mixed",
        "content": "Đang bật thiết bị.",
        "calls": [{"name": "smart_plug", "args": {"action": "on", "device": utt}}]
    },
    "iot_wemo_off": lambda utt: {
        "type": "mixed",
        "content": "Đã tắt thiết bị.",
        "calls": [{"name": "smart_plug", "args": {"action": "off", "device": utt}}]
    },
    "iot_hue_lightchange": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "smart_light", "args": {"query": utt}}]
    },
    "iot_coffee": lambda utt: {
        "type": "mixed",
        "content": "Đang pha cà phê cho bạn.",
        "calls": [{"name": "smart_coffee", "args": {}}]
    },
    
    # Food & Transport
    "cooking_recipe": lambda utt: {
        "type": "mixed",
        "content": "Để tôi tìm công thức cho bạn.",
        "calls": [{"name": "get_recipe", "args": {"query": utt}}]
    },
    "takeaway_order": lambda utt: {
        "type": "mixed",
        "content": "Tôi sẽ đặt đồ ăn cho bạn.",
        "calls": [{"name": "order_food", "args": {"query": utt}}]
    },
    "takeaway_query": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "track_order", "args": {"query": utt}}]
    },
    "transport_taxi": lambda utt: {
        "type": "mixed",
        "content": "Đang đặt xe cho bạn.",
        "calls": [{"name": "book_taxi", "args": {"query": utt}}]
    },
    "transport_traffic": lambda utt: {
        "type": "tool_call",
        "calls": [{"name": "check_traffic", "args": {"query": utt}}]
    },
    
    # Recommendation
    "recommendation_movies": lambda utt: {
        "type": "mixed",
        "content": "Để tôi gợi ý phim cho bạn.",
        "calls": [{"name": "recommend_movie", "args": {"query": utt}}]
    },
    
    # General
    "general_quirky": lambda utt: {
        "type": "text",
        "content": _general_response(utt)
    },
}


def _general_response(utt: str) -> str:
    """Generate varied responses for general/quirky intents."""
    responses = [
        f"Tôi hiểu bạn đang nói về '{utt}'. Bạn cần tôi giúp gì thêm không?",
        f"Cảm ơn bạn đã chia sẻ! Tôi có thể giúp gì cho bạn?",
        f"Tôi nghe thấy rồi. Bạn muốn tôi làm gì tiếp theo?",
        f"Thật thú vị! Bạn có muốn tìm hiểu thêm về điều này không?",
    ]
    # Deterministic based on utterance
    idx = int(hashlib.md5(utt.encode()).hexdigest(), 16) % len(responses)
    return responses[idx]


def generate_speech_massive_label(intent_str: str, utt: str) -> dict:
    """Generate label for a Speech-MASSIVE sample."""
    mapper = INTENT_MAP.get(intent_str)
    if mapper:
        return mapper(utt)
    # Fallback: search
    return {
        "type": "mixed",
        "content": "Để tôi tìm hiểu thêm.",
        "calls": [{"name": "search", "args": {"query": utt}}]
    }


# ============================================================
# ViMD: transcript → response generation (heuristic)
# ============================================================

# Topic keywords → tool mapping
VIMD_TOPIC_PATTERNS = [
    # Government/policy → text response + search for context
    (r"(chính sách|quy hoạch|ủy ban|huyện|tỉnh|xã|nông thôn|phát triển|lãnh đạo|đảng|nhà nước)",
     lambda t: {
         "type": "mixed",
         "content": _summarize_topic(t, "chính sách và quản lý"),
         "calls": [{"name": "search", "args": {"query": _extract_topic(t)}}]
     }),
    
    # Traffic/transport → check_traffic
    (r"(giao thông|đường|cầu|điểm nghẽn|xe|phương tiện|tuyến)",
     lambda t: {
         "type": "mixed",
         "content": _summarize_topic(t, "giao thông"),
         "calls": [{"name": "check_traffic", "args": {"query": _extract_topic(t)}}]
     }),
    
    # Education → search
    (r"(đào tạo|giáo viên|trường|học sinh|giáo dục|chuyên ngành|văn bằng)",
     lambda t: {
         "type": "mixed",
         "content": _summarize_topic(t, "giáo dục"),
         "calls": [{"name": "search", "args": {"query": _extract_topic(t)}}]
     }),
    
    # Agriculture/land → search
    (r"(đất|nông nghiệp|trồng|cây|ruộng|vườn|canh tác|thu hoạch|giao đất)",
     lambda t: {
         "type": "mixed",
         "content": _summarize_topic(t, "nông nghiệp và đất đai"),
         "calls": [{"name": "search", "args": {"query": _extract_topic(t)}}]
     }),
    
    # Business/economy → search + stock
    (r"(kinh doanh|buôn bán|doanh nghiệp|công ty|thị trường|vốn|đầu tư|quảng cáo)",
     lambda t: {
         "type": "mixed",
         "content": _summarize_topic(t, "kinh doanh"),
         "calls": [{"name": "search", "args": {"query": _extract_topic(t)}}]
     }),
    
    # Healthcare → search
    (r"(bệnh|sức khỏe|bác sĩ|thuốc|y tế|chữa trị|khám|viện)",
     lambda t: {
         "type": "mixed",
         "content": _summarize_topic(t, "y tế và sức khỏe"),
         "calls": [{"name": "search", "args": {"query": _extract_topic(t)}}]
     }),
    
    # Weather related
    (r"(thời tiết|mưa|nắng|bão|lũ|ngập|khí hậu)",
     lambda t: {
         "type": "mixed",
         "content": "Để tôi kiểm tra thông tin thời tiết cho bạn.",
         "calls": [{"name": "get_weather", "args": {"query": _extract_topic(t)}}]
     }),
    
    # Social/community
    (r"(người dân|nhân dân|bà con|cộng đồng|hàng xóm|khu dân cư)",
     lambda t: {
         "type": "mixed",
         "content": _summarize_topic(t, "đời sống cộng đồng"),
         "calls": [{"name": "search", "args": {"query": _extract_topic(t)}}]
     }),
    
    # Housing/construction
    (r"(nhà|xây dựng|căn|khu|công trình|kiến trúc)",
     lambda t: {
         "type": "mixed",
         "content": _summarize_topic(t, "nhà ở và xây dựng"),
         "calls": [{"name": "search", "args": {"query": _extract_topic(t)}}]
     }),
    
    # Tourism
    (r"(du lịch|tham quan|khách|resort|nghỉ dưỡng|danh lam)",
     lambda t: {
         "type": "mixed",
         "content": _summarize_topic(t, "du lịch"),
         "calls": [{"name": "search", "args": {"query": _extract_topic(t)}}]
     }),
]


def _extract_topic(transcript: str) -> str:
    """Extract key phrases from transcript for search query."""
    # Take first 50 chars as topic
    words = transcript.split()[:12]
    return " ".join(words)


def _summarize_topic(transcript: str, topic_name: str) -> str:
    """Generate a contextual response summary."""
    templates = [
        f"Tôi hiểu bạn đang hỏi về {topic_name}. Để tôi tìm thêm thông tin chi tiết.",
        f"Đây là vấn đề liên quan đến {topic_name}. Tôi sẽ tra cứu thêm cho bạn.",
        f"Về {topic_name}, tôi cần tìm thêm dữ liệu để trả lời chính xác.",
        f"Cảm ơn bạn đã hỏi về {topic_name}. Để tôi kiểm tra.",
    ]
    idx = int(hashlib.md5(transcript[:30].encode()).hexdigest(), 16) % len(templates)
    return templates[idx]


def generate_vimd_label(transcript: str) -> dict:
    """Generate label for a ViMD sample based on transcript content."""
    t_lower = transcript.lower()
    
    # Check topic patterns
    for pattern, generator in VIMD_TOPIC_PATTERNS:
        if re.search(pattern, t_lower):
            return generator(transcript)
    
    # Fallback: general text response + search
    # Short transcripts → direct response
    if len(transcript) < 50:
        return {
            "type": "text",
            "content": f"Tôi nghe thấy rồi. Bạn cần tôi giúp gì thêm không?"
        }
    
    # Long transcripts → search for context
    return {
        "type": "mixed",
        "content": "Tôi đã nghe nội dung bạn chia sẻ. Để tôi tìm thêm thông tin liên quan.",
        "calls": [{"name": "search", "args": {"query": _extract_topic(transcript)}}]
    }


# ============================================================
# MAIN: Generate all labels
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["speech_massive", "vimd", "all"], default="all")
    parser.add_argument("--output_dir", default="data/agent_labels")
    parser.add_argument("--vimd_samples", type=int, default=0, help="Max ViMD samples (0=all)")
    args = parser.parse_args()
    
    if args.dataset in ("speech_massive", "all"):
        generate_speech_massive_labels(args.output_dir)
    
    if args.dataset in ("vimd", "all"):
        generate_vimd_labels(args.output_dir, args.vimd_samples)
    
    # Also export tool definitions
    export_tool_defs(args.output_dir)


def generate_speech_massive_labels(output_dir: str):
    """Generate labels for Speech-MASSIVE using intent mapping."""
    from datasets import load_dataset
    
    print("=" * 60)
    print("  Generating Speech-MASSIVE labels")
    print("=" * 60)
    
    labels = []
    intent_counts = {}
    type_counts = {"text": 0, "tool_call": 0, "mixed": 0}
    
    for split in ["train", "validation", "test"]:
        try:
            ds = load_dataset("doof-ferb/Speech-MASSIVE_vie", split=split)
            # Remove audio column — we only need text fields for label generation
            if "audio" in ds.column_names:
                ds = ds.remove_columns(["audio"])
            print(f"  Split '{split}': {len(ds)} samples")
            
            for item in ds:
                intent = item["intent_str"]
                utt = item["utt"]
                label = generate_speech_massive_label(intent, utt)
                
                labels.append({
                    "dataset": "speech_massive",
                    "split": split,
                    "id": item.get("id", ""),
                    "transcript": utt,
                    "intent": intent,
                    "scenario": item["scenario_str"],
                    "output": label,
                })
                
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
                type_counts[label["type"]] += 1
                
        except Exception as e:
            print(f"  [!] Failed to load split '{split}': {e}")
    
    # Save
    out_path = f"{output_dir}/speech_massive_labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    
    print(f"\n  Total: {len(labels)} labels saved to {out_path}")
    print(f"  Types: {type_counts}")
    print(f"  Intents: {len(intent_counts)} unique")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {intent:30} {count:5}")


def generate_vimd_labels(output_dir: str, max_samples: int = 0):
    """Generate labels for ViMD using heuristic topic matching.
    Uses HF rows API to avoid downloading 51GB of audio data."""
    import urllib.request
    
    print("\n" + "=" * 60)
    print("  Generating ViMD labels (via HF API — no audio download)")
    print("=" * 60)
    
    labels = []
    type_counts = {"text": 0, "tool_call": 0, "mixed": 0}
    topic_counts = {}
    
    split_sizes = {"train": 15023, "valid": 1900, "test": 2026}
    batch_size = 100  # HF API max per request
    
    for split, total in split_sizes.items():
        print(f"  Split '{split}': {total} samples")
        fetched = 0
        
        while fetched < total:
            if max_samples > 0 and len([l for l in labels if l.get("split") == split]) >= max_samples:
                break
            
            url = (f"https://datasets-server.huggingface.co/rows?"
                   f"dataset=nguyendv02%2FViMD_Dataset&config=default"
                   f"&split={split}&offset={fetched}&length={batch_size}")
            
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                
                rows = data.get("rows", [])
                if not rows:
                    break
                
                for r in rows:
                    row = r["row"]
                    transcript = row.get("text", "")
                    if not transcript or len(transcript.strip()) < 5:
                        continue
                    
                    label = generate_vimd_label(transcript)
                    
                    # Track which topic matched
                    matched_topic = "general"
                    t_lower = transcript.lower()
                    for pattern, _ in VIMD_TOPIC_PATTERNS:
                        if re.search(pattern, t_lower):
                            matched_topic = pattern[:30]
                            break
                    topic_counts[matched_topic] = topic_counts.get(matched_topic, 0) + 1
                    
                    labels.append({
                        "dataset": "vimd",
                        "split": split,
                        "transcript": transcript[:500],
                        "output": label,
                    })
                    
                    type_counts[label["type"]] += 1
                
                fetched += len(rows)
                if fetched % 500 == 0 or fetched >= total:
                    print(f"    {fetched}/{total} fetched...")
                    
            except Exception as e:
                print(f"    [!] API error at offset {fetched}: {e}")
                break
    
    # Save
    out_path = f"{output_dir}/vimd_labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    
    print(f"\n  Total: {len(labels)} labels saved to {out_path}")
    print(f"  Types: {type_counts}")
    print(f"  Top topics:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {topic:35} {count:5}")


def export_tool_defs(output_dir: str):
    """Export tool definitions for reference."""
    out_path = f"{output_dir}/tool_definitions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(TOOLS, f, ensure_ascii=False, indent=2)
    print(f"\n  Tool definitions saved to {out_path} ({len(TOOLS)} tools)")


if __name__ == "__main__":
    main()
