#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全球股票新闻聚合：Google News RSS 多语言检索、去重、分类和透明规则情绪。"""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
import re
import time
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


POSITIVE = ("beat estimates", "profit rises", "revenue growth", "upgrade", "buyback", "wins contract",
            "获批", "业绩增长", "盈利增长", "订单", "回购", "增持", "上调", "创新高", "利好",
            "record profit", "strong demand", "上方修正", "増益", "業績改善")
NEGATIVE = ("misses estimates", "profit warning", "downgrade", "investigation", "lawsuit", "recall",
            "亏损", "业绩下滑", "立案", "处罚", "减持", "下调", "债务", "违约", "利空",
            "weak demand", "export ban", "sanctions", "減益", "下方修正", "不正")
TOPICS = {
    "财报与业绩": ("earnings", "revenue", "profit", "财报", "业绩", "決算"),
    "产品与技术": ("product", "technology", "launch", "芯片", "AI", "产品", "技術"),
    "政策与监管": ("regulation", "policy", "government", "监管", "政策", "制裁", "関税"),
    "贸易与地缘": ("trade", "tariff", "export", "sanction", "贸易", "关税", "地缘"),
    "宏观与利率": ("federal reserve", "interest rate", "inflation", "经济", "美联储", "利率", "インフレ"),
    "诉讼与风险": ("lawsuit", "investigation", "recall", "诉讼", "调查", "风险", "訴訟"),
}


def _text(parent: ET.Element, path: str) -> str:
    node = parent.find(path)
    return unescape(" ".join(node.itertext()).strip()) if node is not None else ""


def _parse_date(value: str) -> str:
    if not value:
        return ""
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OverflowError):
        return value[:80]


def _sentiment(text: str) -> tuple[str, int, int]:
    low = text.casefold()
    pos = sum(low.count(w.casefold()) for w in POSITIVE)
    neg = sum(low.count(w.casefold()) for w in NEGATIVE)
    label = "利好倾向" if pos > neg else ("利空倾向" if neg > pos else "中性/待判断")
    return label, pos, neg


def _feed(query: str, hl: str, gl: str, ceid: str) -> list[dict[str, str]]:
    url = ("https://news.google.com/rss/search?q=" + quote_plus(query) +
           "&hl=" + quote_plus(hl) + "&gl=" + quote_plus(gl) + "&ceid=" + quote_plus(ceid))
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; StockResearchBot/1.0)"})
    with urlopen(req, timeout=18) as response:
        payload = response.read(2_500_000)
    root = ET.fromstring(payload)
    items: list[dict[str, str]] = []
    for item in root.findall("./channel/item"):
        title = _text(item, "title")
        link = _text(item, "link")
        source = _text(item, "source") or "Google News"
        published = _parse_date(_text(item, "pubDate"))
        if title and link:
            items.append({"title": title, "url": link, "source": source, "time": published})
    return items


def fetch_global_news(code: str, company_name: str = "", industry: str = "", limit: int = 60) -> dict[str, Any]:
    """跨中文、英文、日文和欧洲媒体语种检索公司、行业及宏观相关消息。"""
    identity = company_name.strip() or code
    # 每种语言各查公司新闻；行业和宏观查询防止只看到公司公告。
    queries = [
        (f'"{identity}" OR "{code}"', "en-US", "US", "US:en"),
        (f'"{identity}" OR "{code}"', "zh-CN", "CN", "CN:zh-Hans"),
        (f'"{identity}" OR "{code}"', "ja-JP", "JP", "JP:ja"),
    ]
    if industry:
        queries.extend([
            (f'"{industry}" industry AI semiconductor trade policy interest rates', "en-US", "US", "US:en"),
            (f'"{industry}" 行业 政策 产业链', "zh-CN", "CN", "CN:zh-Hans"),
        ])
    queries.extend([
        (f'"{identity}" central bank interest rates trade policy supply chain', "en-US", "GB", "GB:en"),
    ])

    collected: list[dict[str, str]] = []
    errors: list[str] = []
    successful_feeds = 0
    for query, hl, gl, ceid in queries:
        try:
            collected.extend(_feed(query, hl, gl, ceid))
            successful_feeds += 1
        except Exception as exc:
            errors.append(f"{gl}/{hl}: {type(exc).__name__}: {str(exc)[:140]}")
        time.sleep(0.15)

    unique: dict[str, dict[str, str]] = {}
    for item in collected:
        key = re.sub(r"\W+", "", item["title"].casefold())
        if key and key not in unique:
            label, pos, neg = _sentiment(item["title"])
            item["sentiment"] = label
            item["positive_keyword_hits"] = str(pos)
            item["negative_keyword_hits"] = str(neg)
            unique[key] = item

    headlines = list(unique.values())[:max(1, limit)]
    positive = sum(int(x["positive_keyword_hits"]) for x in headlines)
    negative = sum(int(x["negative_keyword_hits"]) for x in headlines)
    total = positive + negative
    topic_counts = {
        topic: sum(any(word.casefold() in x["title"].casefold() for word in words) for x in headlines)
        for topic, words in TOPICS.items()
    }
    impact_notes = []
    for topic, count in topic_counts.items():
        if count:
            impact_notes.append(f"{topic}相关报道 {count} 条；需结合原文、事件时间和公司收入敞口核实，不能仅凭标题推断涨跌。")
    if positive or negative:
        impact_notes.append(f"规则词典识别到正面/负面关键词 {positive}/{negative} 次；该统计不代表新闻因果效应。")
    if not headlines:
        status = "failed" if successful_feeds == 0 else "empty"
    elif errors:
        status = "partial"
    else:
        status = "success"
    return {
        "count": len(headlines),
        "sentiment_score": float((positive - negative) / total) if total else 0.0,
        "positive_keyword_hits": positive,
        "negative_keyword_hits": negative,
        "headlines": headlines,
        "status": status,
        "coverage": "公司/代码相关报道 + 行业链条 + 宏观利率、贸易与政策；Google News RSS 多语言聚合",
        "languages": ["英语", "简体中文", "日语"],
        "successful_feeds": successful_feeds,
        "attempted_feeds": len(queries),
        "sources": sorted({x["source"] for x in headlines}),
        "topic_counts": topic_counts,
        "impact_notes": impact_notes,
        "errors": errors[:8],
        "method": "Google News RSS 多语言聚合；标题级透明关键词规则分类，非 FinBERT/LLM，不作因果判断",
    }
