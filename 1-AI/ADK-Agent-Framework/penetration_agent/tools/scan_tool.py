import requests
import re
from typing import Dict, List


def port_scan(target: str) -> str:
    """
    模拟端口扫描（真实环境中可使用nmap）
    
    参数:
        target: 目标地址（URL或IP）
    
    返回:
        扫描结果字符串
    """
    ports = [80, 443, 8080]
    result = f"端口扫描结果 - {target}\n"
    result += "=" * 50 + "\n"
    result += f"{'端口':<8} {'状态':<10} {'服务':<20}\n"
    result += "-" * 50 + "\n"
    
    for port in ports:
        try:
            if port == 443:
                url = f"https://{target.replace('http://', '').replace('https://', '').rstrip('/')}:{port}"
            else:
                url = f"http://{target.replace('http://', '').replace('https://', '').rstrip('/')}:{port}"
            
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                server = response.headers.get('Server', 'Unknown')
                result += f"{port:<8} {'开放':<10} {server:<20}\n"
            else:
                result += f"{port:<8} {'开放':<10} {'HTTP响应:'+str(response.status_code):<20}\n"
        except:
            result += f"{port:<8} {'关闭':<10} {'连接失败':<20}\n"
    
    return result


def detect_sqli(target: str) -> str:
    """
    检测SQL注入漏洞
    
    参数:
        target: 目标地址（URL）
    
    返回:
        漏洞检测结果
    """
    result = ""
    payloads = ["'", "' OR 1=1--", "' UNION SELECT 1,2,3--"]
    
    for payload in payloads:
        try:
            test_url = target + "?id=1" + payload
            response = requests.get(test_url, timeout=5)
            
            if response.status_code == 500:
                result += f"✓ 发现SQL注入漏洞！Payload: {payload}\n"
                result += f"   URL: {test_url}\n"
                result += f"   状态码: {response.status_code}\n"
                error_patterns = ["MySQL", "SQL syntax", "database error", "Warning:", "error in your SQL syntax"]
                for pattern in error_patterns:
                    if pattern.lower() in response.text.lower():
                        result += f"   检测到数据库错误信息: {pattern}\n"
                return result
            elif "mysql" in response.text.lower() or "error" in response.text.lower():
                result += f"✓ 可能存在SQL注入漏洞！Payload: {payload}\n"
                result += f"   URL: {test_url}\n"
                result += f"   状态码: {response.status_code}\n"
                return result
        except Exception as e:
            result += f"✗ 测试失败: {str(e)}\n"
    
    if not result:
        result = "未检测到SQL注入漏洞\n"
    
    return result


def directory_enumeration(target: str) -> str:
    """
    目录枚举，发现隐藏的敏感路径
    
    参数:
        target: 目标地址（URL）
    
    返回:
        目录枚举结果字符串
    """
    directories = [
        "/admin", "/api", "/backup", "/.git", "/robots.txt", 
        "/sitemap.xml", "/upload", "/config", "/private", "/test",
        "/login", "/logout", "/user", "/index.php", "/.htaccess"
    ]
    
    result = f"目录枚举结果 - {target}\n"
    result += "=" * 60 + "\n"
    result += f"{'路径':<25} {'状态码':<10} {'大小':<10}\n"
    result += "-" * 60 + "\n"
    
    for dir_path in directories:
        try:
            url = target.rstrip('/') + dir_path
            response = requests.get(url, timeout=3)
            result += f"{dir_path:<25} {response.status_code:<10} {len(response.text):<10}\n"
        except:
            result += f"{dir_path:<25} {'超时':<10} {'-':<10}\n"
    
    return result


def detect_xss(target: str) -> str:
    """
    检测XSS漏洞
    
    参数:
        target: 目标地址（URL）
    
    返回:
        漏洞检测结果
    """
    result = ""
    xss_payloads = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert(1)>",
        "<svg/onload=alert(1)>"
    ]
    
    for payload in xss_payloads:
        try:
            test_url = target + "?id=" + payload
            response = requests.get(test_url, timeout=5)
            
            if payload in response.text:
                result += f"✓ 发现XSS漏洞！Payload: {payload[:30]}...\n"
                result += f"   URL: {test_url}\n"
                return result
        except:
            pass
    
    if not result:
        result = "未检测到XSS漏洞\n"
    
    return result


def detect_server_info(target: str) -> str:
    """
    检测服务器信息泄露
    
    参数:
        target: 目标地址（URL）
    
    返回:
        服务器信息
    """
    result = f"服务器信息检测 - {target}\n"
    result += "=" * 60 + "\n"
    
    try:
        response = requests.get(target, timeout=5)
        
        headers_to_check = ['Server', 'X-Powered-By', 'X-AspNet-Version', 'X-Cache']
        for header in headers_to_check:
            if header in response.headers:
                result += f"{header}: {response.headers[header]}\n"
        
        if 'Server' in response.headers:
            server = response.headers['Server']
            if 'Apache' in server or 'Nginx' in server or 'IIS' in server:
                result += f"\n警告: 服务器信息暴露可能被用于针对性攻击\n"
        
        result += f"\n页面标题: {re.search(r'<title>(.*?)</title>', response.text, re.IGNORECASE).group(1) if re.search(r'<title>(.*?)</title>', response.text, re.IGNORECASE) else '未找到'}\n"
        result += f"页面大小: {len(response.text)} 字节\n"
        
    except Exception as e:
        result += f"检测失败: {str(e)}\n"
    
    return result


def run_comprehensive_scan(target: str) -> str:
    """
    执行综合扫描，包含端口扫描、漏洞检测和信息收集
    
    参数:
        target: 目标地址（URL）
    
    返回:
        综合扫描结果字符串
    """
    result = f"{'='*70}\n"
    result += f"综合安全扫描报告 - {target}\n"
    result += f"{'='*70}\n\n"
    
    result += "[阶段1] 服务器信息收集\n"
    result += "-" * 40 + "\n"
    result += detect_server_info(target) + "\n"
    
    result += "[阶段2] 端口扫描\n"
    result += "-" * 40 + "\n"
    result += port_scan(target) + "\n"
    
    result += "[阶段3] SQL注入检测\n"
    result += "-" * 40 + "\n"
    result += detect_sqli(target) + "\n"
    
    result += "[阶段4] XSS检测\n"
    result += "-" * 40 + "\n"
    result += detect_xss(target) + "\n"
    
    result += "[阶段5] 目录枚举\n"
    result += "-" * 40 + "\n"
    result += directory_enumeration(target) + "\n"
    
    result += f"{'='*70}\n"
    result += "扫描完成\n"
    result += f"{'='*70}\n"
    
    return result