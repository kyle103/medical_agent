#!/usr/bin/env python3
"""
前端Demo服务器
提供医疗智能助手前端界面
"""

import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import threading
import time

class FrontendHandler(SimpleHTTPRequestHandler):
    """自定义HTTP处理器，支持SPA路由"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(os.path.dirname(__file__), 'frontend'), **kwargs)
    
    def do_GET(self):
        # 处理前端路由，所有路径都返回index.html
        if self.path.startswith('/api/'):
            # API请求转发到后端服务器
            self.proxy_to_backend('GET')
        else:
            # 静态文件服务
            if self.path == '/' or not os.path.exists(os.path.join(self.directory, self.path[1:])):
                self.path = '/index.html'
            super().do_GET()
    
    def do_POST(self):
        """处理POST请求，转发到后端API"""
        if self.path.startswith('/api/'):
            self.proxy_to_backend('POST')
        else:
            self.send_error(404, "File not found")
    
    def proxy_to_backend(self, method='GET'):
        """将API请求代理到后端服务器"""
        import urllib.request
        import json
        
        try:
            backend_url = f"http://localhost:8000{self.path}"
            
            # 读取请求体（如果是POST）
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else None
            
            # 创建请求
            if method == 'POST':
                req = urllib.request.Request(backend_url, data=post_data, method='POST')
                req.add_header('Content-Type', 'application/json')
            else:
                req = urllib.request.Request(backend_url, method='GET')
            
            # 复制请求头
            for header, value in self.headers.items():
                if header.lower() not in ['host', 'content-length']:
                    req.add_header(header, value)
            
            # 发送请求
            with urllib.request.urlopen(req) as response:
                self.send_response(response.getcode())
                
                # 复制响应头
                for header, value in response.headers.items():
                    if header.lower() not in ['content-length', 'transfer-encoding']:
                        self.send_header(header, value)
                
                self.end_headers()
                
                # 复制响应体，直接写入原始数据
                response_data = response.read()
                self.wfile.write(response_data)
                
        except Exception as e:
            self.send_error(502, f"后端服务不可用: {str(e)}")
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        print(f"[前端服务器] {format % args}")

def start_frontend_server(port=3000):
    """启动前端服务器"""
    server = HTTPServer(('localhost', port), FrontendHandler)
    
    print(f"🚀 前端Demo服务器启动成功!")
    print(f"📱 访问地址: http://localhost:{port}")
    print(f"🔗 后端API: http://localhost:8000")
    print("按 Ctrl+C 停止服务器")
    
    # 自动打开浏览器
    def open_browser():
        time.sleep(1)
        webbrowser.open(f'http://localhost:{port}')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    finally:
        server.server_close()

if __name__ == "__main__":
    start_frontend_server()