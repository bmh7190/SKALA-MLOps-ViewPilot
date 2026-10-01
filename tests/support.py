from datetime import datetime, timedelta, timezone


def video(index=0, **changes):
    row = {
        "video_id": f"video_{index}",
        "published_at": (datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(),
        "subscriber_count_at_publish": 15000,
        "category": "gaming",
        "target_views_day7": 12000,
    }
    for day in range(1, 4):
        row.update({f"day{day}_views": 3000 - day * 100,
                    f"day{day}_impressions": 40000 - day * 100,
                    f"day{day}_avg_view_percentage": 45.5})
    row.update(changes)
    return row


def csv_text(rows, target=True):
    import csv
    import io
    from data.features import CSV_COLUMNS, INPUT_COLUMNS

    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=CSV_COLUMNS if target else INPUT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue()
