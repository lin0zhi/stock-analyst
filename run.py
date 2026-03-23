import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from openai import OpenAI
import json

# --- 配置 ---

# 你想关注的投资组合 (请在运行时或配置中修改)
# 主要持仓集中在 A 股 (上交所 .SS, 深交所 .SZ)
MY_PORTFOLIO = [
    "002624.SZ", # 完美世界
    "600183.SS", # 生益科技
    "601318.SS", # 中国平安
]

MARKET_INDICES = {
    "Nasdaq 100": "^NDX",
    "S&P 500": "^GSPC",
    "Gold (GC=F)": "GC=F",
    "VIX": "^VIX",
    "SSE Composite": "000001.SS"
}

# --- 1. 获取市场指数数据 ---
def get_market_data():
    print("正在获取全球关键指数数据...")
    data_summary = []
    
    tickers = list(MARKET_INDICES.values())
    try:
        # 批量下载数据 (过去5天以计算涨跌幅)
        df = yf.download(tickers, period="5d", progress=False)['Close']
        
        # 整理数据
        # 注意：yfinance 返回的 DataFrame 列可能是 MultiIndex，也可能是单层
        latest_date = df.index[-1].strftime('%Y-%m-%d')
        
        for name, symbol in MARKET_INDICES.items():
            if symbol in df.columns:
                series = df[symbol].dropna()
                if len(series) >= 2:
                    today_close = series.iloc[-1].item() # .item() 转换为 float
                    prev_close = series.iloc[-2].item()
                    change = today_close - prev_close
                    pct_change = (change / prev_close) * 100
                    
                    data_summary.append(f"- {name}: {today_close:.2f} (变动: {pct_change:+.2f}%)")
                else:
                    data_summary.append(f"- {name}: 数据不足")
            else:
                 data_summary.append(f"- {name}: 获取失败")
                 
    except Exception as e:
        print(f"获取指数数据时出错: {e}")
        return "无法获取实时指数数据。"

    return "\n".join(data_summary)

# --- 2. 获取新闻 ---
def get_latest_news(client):
    news_items = []
    
    # 1. 宏观/综合金融新闻
    print("正在使用千问大模型联网获取宏观财经新闻...")
    try:
        macro_prompt = "请联网搜索并列出过去24小时全球及A股市场最重要的10条财经新闻，简要总结每条新闻的核心内容。请确保新闻的时效性。"
        response = client.chat.completions.create(
            model="qwen-plus", # 使用通义千问模型
            messages=[
                {"role": "system", "content": "你是一名具有联网搜索能力的财经助手。请确保提供的新闻是过去24小时内发生的真实新闻。"},
                {"role": "user", "content": macro_prompt}
            ],
            extra_body={"enable_search": True} # 开启联网搜索
        )
        macro_news = response.choices[0].message.content
        news_items.append(f"### 宏观财经新闻\n{macro_news}\n")
    except Exception as e:
        print(f"获取宏观新闻出错: {e}")
        news_items.append("无法获取宏观新闻。")

    # 2. 个股新闻 (针对 MY_PORTFOLIO)
    print(f"正在逐个获取 {len(MY_PORTFOLIO)} 只持仓股票的相关新闻...")
    for stock_code in MY_PORTFOLIO:
        try:
            # 提取股票代码或名称
            stock_prompt = f"请联网搜索 {stock_code} (及相关上市公司) 近期（过去3天）最重要的新闻，简要总结利好或利空消息。如果没有重大新闻，请简要说明。"
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一名具有联网搜索能力的个股分析助手。"},
                    {"role": "user", "content": stock_prompt}
                ],
                extra_body={"enable_search": True} # 开启联网搜索
            )
            stock_news = response.choices[0].message.content
            news_items.append(f"### {stock_code} 个股新闻\n{stock_news}\n")
        except Exception as e:
            print(f"获取 {stock_code} 新闻出错: {e}")
            news_items.append(f"### {stock_code} 新闻\n获取失败: {str(e)}\n")
    
    return "\n".join(news_items)

# --- 3. AI 分析 ---
def analyze_market(client, market_data_str, news_str):
    print("正在请求千问大模型进行综合分析...")
    
    prompt = f"""
    你是一名资深的 A 股与全球市场分析师。请根据以下提供的【市场数据】和【最新新闻】，对我的主要 A 股持仓进行深度复盘与展望。

    ### 1. 实时市场数据 (全球指数 & A股)
    {market_data_str}

    ### 2. 过去24小时重要新闻 (宏观 & 个股)
    {news_str}

    ### 3. 我的持仓组合 (A股为主)
    {", ".join(MY_PORTFOLIO)}

    ### 任务要求：
    1. 【宏观大势】：结合上证指数、纳指、黄金及 VIX 恐慌指数，研判当前 A 股的市场情绪（偏多/偏空/震荡）。
    2. 【指数分析】：
       - 请根据实时市场数据 (全球指数 & A股)及过去24小时重要新闻 (宏观）分析研判纳达克100指数、标普500指数、黄金ETF、日经指数的走势。
    3. 【持仓个股点评】：
       - 请重点分析上述持仓中，今日由于新闻或板块轮动**受影响最大**的 2-3 只股票。
       - 结合搜索到的个股新闻，解读其利好/利空性质。
    3. 【操作策略】：
       - 针对 A 股持仓给出明确建议（加仓/减仓/观望/止损）。
       - 如果市场风险极高，请明确提示。
    4. 【机会挖掘】：(可选) 基于今日新闻，A 股市场中是否有其他值得关注的热点板块？

    请输出为清晰的 Markdown 格式，语气专业、客观，重点突出。
    """

    try:
        response = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system", "content": "你是一位专业的金融分析师。"},
                {"role": "user", "content": prompt}
            ],
            parameters={
                "temperature": 0.1,  # 降低随机性，提高准确性
                # 如果平台支持，可开启推理增强参数
                # "enable_reasoning": True
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"AI 分析失败: {e}")
        return "AI 分析服务暂时不可用。"

# --- 4. 发送邮件 ---
def send_email_to_all(content, receivers):
    host = os.getenv("EMAIL_HOST")
    port = os.getenv("EMAIL_PORT")
    user = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")
    
    if not all([host, port, user, password, receivers]):
        print("邮件配置不完整，跳过发送。")
        return

    # 拆分收件人 (支持逗号分隔)
    receiver_list = [r.strip() for r in receivers.split(',')]

    print(f"准备发送邮件给 {len(receiver_list)} 位联系人...")
    
    # 建立 SMTP 连接
    try:
        server = smtplib.SMTP_SSL(host, int(port))
        server.login(user, password)
    except Exception as e:
        print(f"SMTP 连接失败: {e}")
        return

    # 遍历发送（或者群发）
    for receiver in receiver_list:
        # 使用 mixed 类型兼容正文和附件
        msg = MIMEMultipart('mixed')
        
        # 修正 From 字段格式： "AI Analyst <user@example.com>"
        sender_header = user
        if '@' in user and '<' not in user:
            sender_header = f"DeepSeek Stock Analyst <{user}>"
        
        msg['From'] = sender_header
        msg['To'] = receiver
        msg['Subject'] = Header(f"📊 [A股日报] 市场复盘与持仓建议 - {datetime.now().strftime('%m-%d')}", 'utf-8')
        
        # 1. 邮件正文 (关键修正: 使用 text/plain，确保内容直接显示)
        text_part = MIMEText(content, 'plain', 'utf-8')
        msg.attach(text_part)

        # 2. 邮件附件 (可选: 如果需要将报告存为附件，使用 .txt 或 .md)
        # 这里我们将同一份报告作为 .txt 附件发送，方便存档
        attachment = MIMEText(content, 'plain', 'utf-8')
        filename = f"stock_analysis_{datetime.now().strftime('%Y%m%d')}.txt"
        attachment.add_header('Content-Disposition', 'attachment', filename=filename)
        msg.attach(attachment)
        
        try:
            server.sendmail(user, [receiver], msg.as_string())
            print(f"已发送 -> {receiver}")
        except Exception as e:
            print(f"发送给 {receiver} 失败: {e}")

    server.quit()
    print("所有邮件处理完毕。")

# --- 主程序 ---
if __name__ == "__main__":
    # 配置 DashScope
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("AI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    if not api_key:
        print("请设置 DASHSCOPE_API_KEY 环境变量。")
        exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 1. 获取数据
    market_info = get_market_data()
    # 传入 client 供大模型调用
    news_info = get_latest_news(client)

    # 2. AI 分析
    report = analyze_market(client, market_info, news_info)
    
    # 3. 打印结果 (用于日志)
    print("\n" + "="*30)
    print(report)
    print("="*30 + "\n")

    # 4. 发送邮件
    receivers_env = os.getenv("EMAIL_RECEIVERS") # 格式: "user1@a.com, user2@b.com"
    if receivers_env:
        send_email_to_all(report, receivers_env)
    else:
        print("未配置 EMAIL_RECEIVERS，未发送邮件。")
