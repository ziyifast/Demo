import os
import datetime


def save_report(content: str) -> str:
    """
    将渗透测试报告保存到本地文件
    
    参数:
        content: 报告内容
        
    返回:
        保存结果信息
    """
    os.makedirs("reports", exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"reports/pen_test_report_{timestamp}.md"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    return f"报告已成功保存到: {report_file}"