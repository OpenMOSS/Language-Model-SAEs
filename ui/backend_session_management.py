# 后端会话管理实现示例
import asyncio
import threading
from typing import Dict, Set
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()

# 全局会话管理
active_sessions: Set[str] = set()
session_tasks: Dict[str, asyncio.Task] = {}
session_locks: Dict[str, threading.Lock] = {}

class StockfishAnalyzer:
    """Stockfish分析器，支持取消功能"""
    
    def __init__(self):
        self.process = None
        self.cancelled = False
    
    async def analyze_position_with_cancellation(self, fen: str, session_id: str) -> dict:
        """可取消的位置分析"""
        try:
            # 检查会话是否已被取消
            if session_id not in active_sessions:
                return {"status": "cancelled", "message": "Session was cancelled before analysis"}
            
            # 模拟Stockfish分析过程（实际实现中这里会启动Stockfish进程）
            print(f"开始分析 FEN: {fen[:20]}... (会话: {session_id})")
            
            # 分段分析，每段之间检查取消状态
            for i in range(10):  # 模拟10个分析步骤
                await asyncio.sleep(0.5)  # 每步0.5秒
                
                # 定期检查会话是否被取消
                if session_id not in active_sessions:
                    print(f"分析被取消 (步骤 {i+1}/10): {fen[:20]}... (会话: {session_id})")
                    return {"status": "cancelled", "message": f"Analysis cancelled at step {i+1}"}
            
            # 模拟分析结果
            result = {
                "status": "success",
                "best_move": "e2e4",  # 示例走法
                "ponder": "e7e5",
                "fen": fen,
                "evaluation": 0.2
            }
            
            print(f"分析完成: {fen[:20]}... (会话: {session_id}) -> {result['best_move']}")
            return result
            
        except asyncio.CancelledError:
            print(f"分析任务被取消: {fen[:20]}... (会话: {session_id})")
            return {"status": "cancelled", "message": "Analysis task was cancelled"}
        except Exception as e:
            print(f"分析出错: {fen[:20]}... (会话: {session_id}) -> {str(e)}")
            return {"status": "error", "error": str(e)}

stockfish_analyzer = StockfishAnalyzer()

@app.post("/analyze/stockfish")
async def analyze_stockfish(request: dict):
    """Stockfish分析端点"""
    fen = request.get("fen")
    session_id = request.get("session_id")
    
    if not fen:
        raise HTTPException(status_code=400, detail="FEN is required")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    
    # 检查会话是否有效
    if session_id not in active_sessions:
        print(f"会话无效或已取消: {session_id}")
        return {"status": "cancelled", "message": "Session is not active"}
    
    # 创建分析任务
    task = asyncio.create_task(
        stockfish_analyzer.analyze_position_with_cancellation(fen, session_id)
    )
    
    # 存储任务引用（可用于强制取消）
    if session_id not in session_tasks:
        session_tasks[session_id] = []
    session_tasks[session_id].append(task)
    
    try:
        result = await task
        return result
    finally:
        # 清理任务引用
        if session_id in session_tasks:
            session_tasks[session_id].remove(task)

@app.post("/cancel_session")
async def cancel_session(request: dict):
    """取消会话端点"""
    session_id = request.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    
    print(f"🛑 收到取消会话请求: {session_id}")
    
    # 从活跃会话中移除
    if session_id in active_sessions:
        active_sessions.remove(session_id)
        print(f"✅ 会话已从活跃列表移除: {session_id}")
    
    # 取消该会话的所有正在进行的任务
    if session_id in session_tasks:
        tasks = session_tasks[session_id]
        cancelled_count = 0
        
        for task in tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1
        
        # 清理任务列表
        del session_tasks[session_id]
        print(f"✅ 已取消 {cancelled_count} 个正在进行的分析任务")
    
    return {
        "status": "success", 
        "message": f"Session {session_id} cancelled successfully"
    }

@app.post("/start_session")
async def start_session(request: dict):
    """开始新会话"""
    session_id = request.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    
    # 添加到活跃会话
    active_sessions.add(session_id)
    session_tasks[session_id] = []
    
    print(f"✨ 新会话已启动: {session_id}")
    
    return {
        "status": "success",
        "message": f"Session {session_id} started"
    }

@app.get("/sessions")
async def get_active_sessions():
    """获取活跃会话列表"""
    return {
        "active_sessions": list(active_sessions),
        "task_counts": {sid: len(tasks) for sid, tasks in session_tasks.items()}
    }

if __name__ == "__main__":
    print("🚀 启动后端服务器...")
    print("📍 端点:")
    print("  POST /start_session - 开始新会话")
    print("  POST /analyze/stockfish - 分析位置")
    print("  POST /cancel_session - 取消会话")
    print("  GET /sessions - 查看活跃会话")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 