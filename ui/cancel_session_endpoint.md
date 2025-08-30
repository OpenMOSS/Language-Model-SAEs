# 后端会话管理端点

为了实现真正的快速切换，需要在后端添加以下端点：

## 1. 取消会话端点

```python
@app.post("/cancel_session")
async def cancel_session(request: dict):
    session_id = request.get("session_id")
    if session_id:
        # 停止该会话的所有正在进行的Stockfish推理
        cancel_stockfish_analysis_by_session(session_id)
        return {"status": "success", "message": f"Session {session_id} cancelled"}
    return {"status": "error", "message": "No session_id provided"}
```

## 2. 修改现有的 /analyze/stockfish 端点

```python
@app.post("/analyze/stockfish")
async def analyze_stockfish(request: dict):
    fen = request.get("fen")
    session_id = request.get("session_id")
    
    # 检查会话是否已被取消
    if session_id and is_session_cancelled(session_id):
        return {"status": "cancelled", "message": "Session was cancelled"}
    
    # 在推理过程中定期检查会话状态
    result = await stockfish_analysis_with_session_check(fen, session_id)
    return result
```

这样可以实现：
1. 前端切换feature时立即调用 /cancel_session
2. 后端停止旧会话的所有推理
3. 新会话的推理逐个完成并返回结果 