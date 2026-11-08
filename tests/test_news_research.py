from stock_prediction.news_research import fetch_global_news


def test_global_news_aggregates_languages_deduplicates_and_classifies(monkeypatch):
    calls = []

    def fake_feed(query, hl, gl, ceid):
        calls.append((query, hl, gl, ceid))
        if hl == "en-US":
            return [{
                "title": "Example Corp beats estimates with strong demand",
                "url": "https://example.com/news",
                "source": "Example Finance",
                "time": "2026-11-08 10:00 UTC",
            }]
        if hl == "zh-CN":
            return [{
                "title": "Example Corp beats estimates with strong demand",
                "url": "https://example.com/news-duplicate",
                "source": "Duplicate Source",
                "time": "2026-11-08 10:00 UTC",
            }, {
                "title": "行业监管政策更新",
                "url": "https://example.com/policy",
                "source": "Example China",
                "time": "2026-11-08 09:00 UTC",
            }]
        return []

    monkeypatch.setattr("stock_prediction.news_research._feed", fake_feed)
    monkeypatch.setattr("stock_prediction.news_research.time.sleep", lambda _: None)
    result = fetch_global_news("600000", "Example Corp", "banking", limit=60)

    assert len(calls) >= 4
    assert result["count"] == 2
    assert result["status"] == "success"
    assert result["sentiment_score"] > 0
    assert result["successful_feeds"] == len(calls)
    assert result["headlines"][0]["sentiment"] == "利好倾向"
    assert result["topic_counts"]["政策与监管"] == 1
    assert result["sources"]


def test_global_news_reports_all_feed_failures(monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError("feed timeout")

    monkeypatch.setattr("stock_prediction.news_research._feed", fail)
    monkeypatch.setattr("stock_prediction.news_research.time.sleep", lambda _: None)
    result = fetch_global_news("000001", limit=5)

    assert result["count"] == 0
    assert result["status"] == "failed"
    assert result["sentiment_score"] == 0.0
    assert result["errors"]
