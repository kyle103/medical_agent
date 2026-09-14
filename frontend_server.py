#!/usr/bin/env python3
"""
前端Demo服务器
提供医疗智能助手前端界面
"""

import os
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

# Windows 控制台默认 GBK，emoji/中文 print 会 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

class FrontendHandler(SimpleHTTPRequestHandler):
    """自定义HTTP处理器，支持SPA路由"""

    # 单个连接的空闲超时（秒）。浏览器会为同一来源开"预连接"空 socket（Edge/Chrome
    # 都会），这种连接只完成 TCP 握手、不发任何请求字节。没有超时的话，处理线程会
    # 永久阻塞在 rfile.readline() 上——配合单线程 HTTPServer 就是整站假死：
    # 页面卡在加载中，而 :8000 后端日志一片正常（请求根本没发出去）。
    # 超时后被 handle_one_request 捕获 → 关掉这个连接，线程立刻回收。
    timeout = 15

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(os.path.dirname(__file__), 'frontend'), **kwargs)

    def end_headers(self):
        """给所有静态响应加 `Cache-Control: no-cache`。

        `SimpleHTTPRequestHandler` 只发 `Last-Modified`，**不发 `Cache-Control`/`ETag`**，
        浏览器于是走"启发式缓存"（新鲜期约为文档年龄的 10%）—— 改完静态文件后，已经打开的
        标签页可能长时间不重新拉，看起来就是"改了代码页面没变"。

        `no-cache` = 允许缓存，但每次使用前必须带 `If-Modified-Since` 回源校验（不是
        `no-store`）。文件没变则 304 零传输，变了立刻拿到新的 —— 开发期就该这样。
        """
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_GET(self):
        # 处理前端路由，所有路径都返回index.html
        #
        # ⚠️ 判断"文件是否存在"前**必须先剥掉 query string**：`?v=123` 这种缓存击穿参数会让
        # `os.path.exists('frontend/app.css?v=123')` 为假 → 回落到 index.html → 浏览器拿到
        # 一份 HTML 当样式表用，**整份 CSS 静默失效**（不报错、不加载、页面裸奔）。
        # 基类 `translate_path` 自己会剥 query，这里只需把"存不存在"的判断对齐即可。
        path_only = self.path.split('?', 1)[0].split('#', 1)[0]
        if path_only.startswith('/api/'):
            # API请求直接返回404，让前端直接请求后端
            self.send_error(404, "API endpoint not found")
        else:
            # 静态文件服务
            if path_only == '/' or not os.path.exists(os.path.join(self.directory, path_only[1:])):
                self.path = '/index.html'
            super().do_GET()
    
    def do_POST(self):
        """处理POST请求，直接返回404，让前端直接请求后端"""
        if self.path.startswith('/api/'):
            self.send_error(404, "API endpoint not found")
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
            try:
                with urllib.request.urlopen(req) as response:
                    self.send_response(response.getcode())
                    
                    # 复制响应头
                    for header, value in response.headers.items():
                        if header.lower() not in ['content-length', 'transfer-encoding', 'connection']:
                            self.send_header(header, value)
                    
                    self.end_headers()
                    
                    # 复制响应体，直接写入原始数据
                    response_data = response.read()
                    self.wfile.write(response_data)
            except urllib.error.HTTPError as e:
                # 处理HTTP错误，包括401
                try:
                    self.send_response(e.code)
                    
                    # 复制响应头
                    for header, value in e.headers.items():
                        if header.lower() not in ['content-length', 'transfer-encoding', 'connection']:
                            self.send_header(header, value)
                    
                    self.end_headers()
                    
                    # 复制响应体
                    response_data = e.read()
                    self.wfile.write(response_data)
                except Exception as inner_e:
                    # 处理内部错误，确保至少返回正确的状态码
                    self.send_response(e.code)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    error_data = json.dumps({"detail": "未授权"}).encode('utf-8')
                    self.wfile.write(error_data)
                
        except Exception as e:
            # 处理Unicode编码问题
            error_message = "后端服务不可用"
            self.send_error(502, error_message)
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        print(f"[前端服务器] {format % args}")

def start_frontend_server(port=3000):
    """启动前端服务器（浏览器由一键启动脚本负责打开，这里不做）"""
    # 绑定 0.0.0.0 避免 localhost 解析到 IPv6 ::1 导致 127.0.0.1 访问不到。
    # 必须是 ThreadingHTTPServer：单线程 HTTPServer 只要有一个连接不放手（浏览器的
    # 预连接空 socket、或者一个没读完响应的标签页），唯一的处理线程就永久卡在
    # rfile.readline() 上，之后所有静态请求全部挂起 → 页面加载不出来、后端却一条
    # 请求都收不到（用户看到的就是"登录成功但前端完全没法用"）。
    server = ThreadingHTTPServer(('0.0.0.0', port), FrontendHandler)
    server.daemon_threads = True

    print(f"前端Demo服务器启动成功!")
    print(f"访问地址: http://localhost:{port}")
    print(f"后端API: http://localhost:8000")
    print("按 Ctrl+C 停止服务器")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("服务器已停止")
    finally:
        server.server_close()

if __name__ == "__main__":
    start_frontend_server()