"""
智能短视频分析 —— 启动入口（前后端分离）

- 后端 API：backend/server.py（上传/状态/结果/纠错/导出，JSON 接口）
- 前端页面：frontend/index.html（浏览器调用 API）

启动: python web_ui.py --port 8800
然后浏览器打开 http://127.0.0.1:8800
"""

from backend.server import main

if __name__ == "__main__":
    main()
