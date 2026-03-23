import os
import json
from openai import OpenAI

# --- 配置 (保持与 run.py 一致) ---
MY_PORTFOLIO = [
    "002624.SZ", # 完美世界
    "600183.SS", # 生益科技
    "601318.SS", # 中国平安
]

# --- 从 run.py 复制的新闻获取逻辑 (适配 QwenSearch) ---
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

# --- 主程序 ---
if __name__ == "__main__":
    # 配置 DashScope (阿里云)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("AI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    if not api_key:
        print("错误: 必须设置 AI_API_KEY 环境变量才能运行此测试。")
        print("请使用 DashScope API Key (sk-xxxx)")
        exit(1)

    print(f"正在连接 AI 服务: {base_url}")
    print(f"使用模型: qwen-plus (开启搜索增强)")
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        print("开始测试新闻抓取逻辑 (Qwen 联网模式)...")
        results = get_latest_news(client)
        
        print("\n" + "="*50)
        print(" 抓取结果汇总 ".center(50, "="))
        print("="*50)
        print(results)
        print("="*50)
        
    except Exception as e:
        print(f"测试过程中发生未捕获异常: {e}")
