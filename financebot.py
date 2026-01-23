# 福生无量天尊
from openai import OpenAI
import feedparser
import requests
from newspaper import Article
from datetime import datetime
import time
import pytz
import os


# =========================
# 配置区（可选）
# =========================

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

# 早报/晚报每个来源抓取条数（可用 Actions 变量/环境变量覆盖）
FULL_MAX_PER_SOURCE = int(os.getenv("FULL_MAX_PER_SOURCE", "5"))  # 早报
LITE_MAX_PER_SOURCE = int(os.getenv("LITE_MAX_PER_SOURCE", "3"))  # 晚报

# 抓取正文最大长度（用于LLM输入）
ARTICLE_TEXT_MAX_LEN = int(os.getenv("ARTICLE_TEXT_MAX_LEN", "1500"))

# 轻微延迟，减少反爬风险
REQUEST_SLEEP_SEC = float(os.getenv("REQUEST_SLEEP_SEC", "0.25"))

# AI 研究报告最大长度（避免微信里太长）
REPORT_MAX_CHARS = int(os.getenv("REPORT_MAX_CHARS", "1800"))


# =========================
# 环境变量
# =========================

# 优先使用 DEEPSEEK_API_KEY；兼容你 workflow 里设置的 OPENAI_API_KEY
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("未设置 API Key：请在 Github Actions 中设置 DEEPSEEK_API_KEY（推荐）或 OPENAI_API_KEY。")

base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
model_name = os.getenv("LLM_MODEL", DEFAULT_MODEL)

# Server 酱 keys（支持多个，用英文逗号分隔）
server_chan_keys_env = os.getenv("SERVER_CHAN_KEYS")
if not server_chan_keys_env:
    raise ValueError("环境变量 SERVER_CHAN_KEYS 未设置，请在 Github Actions 中设置此变量！")
server_chan_keys = [k.strip() for k in server_chan_keys_env.split(",") if k.strip()]

openai_client = OpenAI(api_key=api_key, base_url=base_url)


# =========================
# RSS 源
# =========================
rss_feeds = {
    "💲 华尔街见闻": {
        "华尔街见闻": "https://dedicated.wallstreetcn.com/rss.xml",
    },
    "💻 36氪": {
        "36氪": "https://36kr.com/feed",
    },
    "🇨🇳 中国经济": {
        "香港經濟日報": "https://www.hket.com/rss/china",
        "东方财富": "http://rss.eastmoney.com/rss_partener.xml",
        "百度股票焦点": "http://news.baidu.com/n?cmd=1&class=stock&tn=rss&sub=0",
        "中新网": "https://www.chinanews.com.cn/rss/finance.xml",
        "国家统计局-最新发布": "https://www.stats.gov.cn/sj/zxfb/rss.xml",
    },
    "🇺🇸 美国经济": {
        "华尔街日报 - 经济": "https://feeds.content.dowjones.io/public/rss/WSJcomUSBusiness",
        "华尔街日报 - 市场": "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
        "MarketWatch美股": "https://www.marketwatch.com/rss/topstories",
        "ZeroHedge华尔街新闻": "https://feeds.feedburner.com/zerohedge/feed",
        "ETF Trends": "https://www.etftrends.com/feed/",
    },
    "🌍 世界经济": {
        "华尔街日报 - 经济": "https://feeds.content.dowjones.io/public/rss/socialeconomyfeed",
        "BBC全球经济": "http://feeds.bbci.co.uk/news/business/rss.xml",
    },
}


# =========================
# 时间/模式
# =========================

def now_cn():
    return datetime.now(pytz.timezone("Asia/Shanghai"))


def today_str_cn():
    return now_cn().strftime("%Y-%m-%d")


def get_run_mode():
    """
    早报 full：上午触发（08:50）
    晚报 lite：下午/晚上触发（19:30）
    你用的是定时任务，所以用小时判断足够稳定。
    """
    return "full" if now_cn().hour < 12 else "lite"


# =========================
# 抓取/解析
# =========================

def fetch_article_text(url, max_len=ARTICLE_TEXT_MAX_LEN):
    """抓取文章正文（仅 full 模式使用）"""
    try:
        print(f"📰 正在爬取文章内容: {url}")
        article = Article(url)
        article.download()
        article.parse()
        text = (article.text or "").strip()
        if not text:
            print(f"⚠️ 文章内容为空: {url}")
            return ""
        return text[:max_len]
    except Exception as e:
        print(f"❌ 文章爬取失败: {url}，错误: {e}")
        return ""


def fetch_feed_with_headers(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    return feedparser.parse(url, request_headers=headers)


def fetch_feed_with_retry(url, retries=3, delay=5):
    for i in range(retries):
        try:
            feed = fetch_feed_with_headers(url)
            if feed and hasattr(feed, "entries") and len(feed.entries) > 0:
                return feed
        except Exception as e:
            print(f"⚠️ 第 {i+1} 次请求 {url} 失败: {e}")
        time.sleep(delay)
    print(f"❌ 跳过 {url}, 尝试 {retries} 次后仍失败。")
    return None


def fetch_rss_articles(rss_feeds, mode="full", max_articles=5):
    """
    返回：
      - news_data：用于展示（标题+链接）
      - analysis_text：用于LLM分析（仅full模式会填充）
      - stats：简单统计（用于头部看板）
    """
    news_data = {}
    analysis_text = ""
    stats = {
        "sources_ok": 0,
        "sources_fail": 0,
        "items_kept": 0,
        "items_total_seen": 0,
        "body_ok": 0,   # full 才有意义
    }

    for category, sources in rss_feeds.items():
        category_content = ""
        for source, url in sources.items():
            print(f"📡 正在获取 {source} 的 RSS 源: {url}")
            feed = fetch_feed_with_retry(url)
            if not feed:
                stats["sources_fail"] += 1
                print(f"⚠️ 无法获取 {source} 的 RSS 数据")
                continue

            stats["sources_ok"] += 1
            print(f"✅ {source} RSS 获取成功，共 {len(feed.entries)} 条新闻")

            articles = []
            # 只取前 max_articles 条
            entries = feed.entries[:max_articles]
            stats["items_total_seen"] += len(entries)

            for entry in entries:
                title = entry.get("title", "无标题")
                link = entry.get("link", "") or entry.get("guid", "")
                if not link:
                    print(f"⚠️ {source} 的新闻 '{title}' 没有链接，跳过")
                    continue

                # full 模式：抓正文用于研究报告
                if mode == "full":
                    article_text = fetch_article_text(link)
                    if article_text:
                        stats["body_ok"] += 1
                        analysis_text += f"\n{article_text}\n\n"

                print(f"🔹 {source} - {title} 获取成功")
                stats["items_kept"] += 1
                articles.append(f"- [{title}]({link})")
                time.sleep(REQUEST_SLEEP_SEC)

            if articles:
                category_content += f"#### {source}\n" + "\n".join(articles) + "\n\n"

        if category_content.strip():
            news_data[category] = category_content

    return news_data, analysis_text, stats


# =========================
# LLM 总结（仅 full）
# =========================

def summarize(text):
    if not text.strip():
        return "（未抓取到足够正文内容，已自动降级：仅推送新闻速览。）"

    try:
        completion = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": """
你是一名专业的财经新闻分析师。请根据以下新闻正文，输出一份“可直接阅读的投研晨报”，要求：
- 结构固定为四段（用标题标注）：【热点看板】【宏观与政策】【行业与主题轮动】【风险提示与结论】
- 【热点看板】用要点列出：1天热点TOP3、3天走强且此前两周平淡的主题TOP3（如无涨幅数据，结合新闻热度与情绪推断）
- 每个热点给出：催化剂、复盘（近3个月关键逻辑/阶段变化）、展望（短炒/可持续）
- 全文控制在 1500 字以内，逻辑清晰、面向专业投资者。
""".strip(),
                },
                {"role": "user", "content": text},
            ],
        )
        result = completion.choices[0].message.content.strip()
        return result[:REPORT_MAX_CHARS]

    except Exception as e:
        # 关键：LLM失败也不让 workflow 直接挂掉
        print(f"❌ AI 总结失败：{repr(e)}")
        return "（AI 总结失败：可能是余额不足/限流/网络问题，已自动降级：仅推送新闻速览。）"


# =========================
# 推送
# =========================

def send_to_wechat(title, content):
    for key in server_chan_keys:
        url = f"https://sctapi.ftqq.com/{key}.send"
        data = {"title": title, "desp": content}
        try:
            response = requests.post(url, data=data, timeout=10)
            if response.ok:
                print(f"✅ 推送成功: {key}")
            else:
                print(f"❌ 推送失败: {key}, 响应：{response.text}")
        except Exception as e:
            print(f"❌ 推送异常: {key}, 错误：{e}")


# =========================
# 文本排版（样式优化核心）
# =========================

def fmt_header(today_str, mode, stats):
    mode_name = "☀️ 早报｜研究报告" if mode == "full" else "🌆 晚报｜盘后快报"
    lines = []
    lines.append(f"# {mode_name}")
    lines.append("")
    lines.append(f"**日期**：{today_str}（北京时间）")
    lines.append(f"**数据**：RSS 源 {stats['sources_ok']} 成功 / {stats['sources_fail']} 失败；条目 {stats['items_kept']}（扫描 {stats['items_total_seen']}）")
    if mode == "full":
        lines.append(f"**正文抓取**：成功 {stats['body_ok']} 条（用于研究分析）")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def fmt_news_section(news_data):
    lines = []
    lines.append("## 📰 新闻速览（按分类）")
    lines.append("")
    # 分类之间加分割线，阅读更舒服
    for category, content in news_data.items():
        if content.strip():
            lines.append(f"### {category}")
            lines.append("")
            lines.append(content.strip())
            lines.append("---")
            lines.append("")
    return "\n".join(lines).strip()


def fmt_full_report(summary):
    lines = []
    lines.append("## 🧠 今日研究报告（宏观 + 行业）")
    lines.append("")
    lines.append("> 建议先看这一部分：热点、逻辑、展望都在这里。")
    lines.append("")
    lines.append(summary.strip())
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def fmt_lite_focus_template():
    # 晚报不调用模型：给一个固定的“盘后重点模板”，你可以手动在微信里补一句
    return "\n".join([
        "## ✅ 盘后重点（建议只看这部分）",
        "",
        "- 1）",
        "- 2）",
        "- 3）",
        "",
        "> 注：晚报为快报模式（不抓正文、不调用模型），更快更省钱。",
        "",
        "---",
        ""
    ])


# =========================
# 主程序
# =========================

if __name__ == "__main__":
    today_str = today_str_cn()
    mode = get_run_mode()

    if mode == "full":
        print("☀️ 早报模式：抓正文 + 调模型，生成研究报告")
        max_per_source = FULL_MAX_PER_SOURCE

        news_data, analysis_text, stats = fetch_rss_articles(
            rss_feeds, mode="full", max_articles=max_per_source
        )

        summary = summarize(analysis_text)

        # 如果研究报告失败，会返回“已降级”提示，这时也照样推送新闻速览
        content = ""
        content += fmt_header(today_str, mode, stats)
        content += fmt_full_report(summary)
        content += fmt_news_section(news_data)

        push_title = f"☀️ {today_str} 早报｜财经研究报告"

    else:
        print("🌆 晚报模式：不抓正文、不调模型，仅推送盘后快报")
        max_per_source = LITE_MAX_PER_SOURCE

        news_data, _, stats = fetch_rss_articles(
            rss_feeds, mode="lite", max_articles=max_per_source
        )

        content = ""
        content += fmt_header(today_str, mode, stats)
        content += fmt_lite_focus_template()
        content += fmt_news_section(news_data)

        push_title = f"🌆 {today_str} 晚报｜盘后快报"

    send_to_wechat(title=push_title, content=content)
