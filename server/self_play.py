import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from transformer_lens import HookedTransformer
import sys
from pathlib import Path
import chess

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from lm_saes.circuit.leela_board import LeelaBoard
    LEELA_BOARD_AVAILABLE = True
except ImportError:
    print("WARNING: LeelaBoard not found, using fallback board logic")
    LeelaBoard = None
    LEELA_BOARD_AVAILABLE = False


class ChessSelfPlay:
    """国际象棋自对弈引擎"""
    
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.device = next(model.parameters()).device
        
        # 初始化LeelaBoard（如果可用）
        if LEELA_BOARD_AVAILABLE and LeelaBoard is not None:
            self.lboard = LeelaBoard.from_fen(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                history_synthesis=True
            )
        else:
            self.lboard = None
        
        # 缓存当前局面的模型输出
        self.current_fen = None
        self.cached_outputs = None
    
    def fen_to_tensor(self, fen: str) -> str:
        """将FEN字符串转换为模型输入张量"""
        # 这里需要根据你的模型输入格式来实现
        # 假设模型直接接受FEN字符串
        return fen
    
    def run_model_inference(self, fen: str) -> tuple:
        """运行模型推理并缓存结果"""
        # 如果FEN没有变化，直接返回缓存的结果
        if self.current_fen == fen and self.cached_outputs is not None:
            return self.cached_outputs
        
        # 运行新的推理
        with torch.no_grad():
            outputs, _ = self.model.run_with_cache(fen, prepend_bos=False)
        
        # 缓存结果
        self.current_fen = fen
        self.cached_outputs = outputs
        
        return outputs
    
    def get_model_evaluation(self, fen: str) -> Tuple[float, float, float]:
        """获取模型对当前局面的评估 (WDL: Win, Draw, Loss) - 统一从白方视角"""
        try:
            # 使用缓存的模型推理结果
            outputs = self.run_model_inference(fen)
            
            # 模型输出是一个列表，包含三个元素：
            # outputs[0]: logits, shape [1, 1858]
            # outputs[1]: WDL, shape [1, 3] - [当前行棋方胜率, 和棋率, 当前行棋方败率]
            # outputs[2]: 其他输出, shape [1, 1]
            
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                wdl_tensor = outputs[1]  # 获取WDL输出
                if wdl_tensor.shape == torch.Size([1, 3]):
                    # WDL已经是概率分布，不需要softmax
                    current_player_win = wdl_tensor[0][0].item()  # 当前行棋方胜率
                    draw_prob = wdl_tensor[0][1].item()  # 和棋率
                    current_player_loss = wdl_tensor[0][2].item()  # 当前行棋方败率
                    
                    # 判断当前行棋方是谁
                    board = chess.Board(fen)
                    is_white_to_move = board.turn == chess.WHITE
                    
                    # 转换为白方视角的WDL
                    if is_white_to_move:
                        # 白方行棋，直接使用模型输出
                        white_win = current_player_win
                        white_loss = current_player_loss
                    else:
                        # 黑方行棋，需要翻转胜败率
                        white_win = current_player_loss  # 黑方败率 = 白方胜率
                        white_loss = current_player_win  # 黑方胜率 = 白方败率
                    
                    return white_win, draw_prob, white_loss
                else:
                    print(f"WDL输出形状不正确: {wdl_tensor.shape}, 期望 [1, 3]")
                    return 0.5, 0.2, 0.3
            else:
                print(f"模型输出格式不正确，期望包含至少2个元素的列表，实际得到: {type(outputs)}")
                return 0.5, 0.2, 0.3
                
        except Exception as e:
            print(f"模型评估失败: {e}")
            return 0.5, 0.2, 0.3
    
    def get_legal_moves(self, fen: str) -> List[str]:
        """获取当前局面的合法移动"""
        try:
            board = chess.Board(fen)
            return [move.uci() for move in board.legal_moves]
        except Exception as e:
            print(f"获取合法移动失败: {e}")
            return []
    
    def get_move_probabilities(self, fen: str, legal_moves: List[str]) -> Dict[str, float]:
        """获取每个合法移动的概率"""
        try:
            # 使用缓存的模型推理结果
            outputs = self.run_model_inference(fen)
            
            # 模型输出是一个列表，包含三个元素：
            # outputs[0]: logits, shape [1, 1858] - 移动概率logits
            # outputs[1]: WDL, shape [1, 3]
            # outputs[2]: 其他输出, shape [1, 1]
            
            move_probs = {}
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 1:
                logits = outputs[0]  # 获取logits输出
                if logits.shape == torch.Size([1, 1858]):
                    # 对logits进行softmax得到概率分布
                    probs = torch.softmax(logits[0], dim=-1)
                    
                    if self.lboard and LEELA_BOARD_AVAILABLE:
                        # 使用LeelaBoard获取移动索引
                        for move in legal_moves:
                            try:
                                move_idx = self.lboard.uci2idx(move)
                                if move_idx is not None and move_idx < probs.shape[0]:
                                    move_probs[move] = probs[move_idx].item()
                            except Exception as move_error:
                                print(f"获取移动 {move} 的索引失败: {move_error}")
                                continue
                    else:
                        # 回退方法：均匀分布
                        uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
                        for move in legal_moves:
                            move_probs[move] = uniform_prob
                else:
                    print(f"Logits输出形状不正确: {logits.shape}, 期望 [1, 1858]")
                    # 回退到均匀分布
                    uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
                    for move in legal_moves:
                        move_probs[move] = uniform_prob
            else:
                print(f"模型输出格式不正确，期望包含至少1个元素的列表，实际得到: {type(outputs)}")
                # 回退到均匀分布
                uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
                for move in legal_moves:
                    move_probs[move] = uniform_prob
            
            # 归一化概率
            total_prob = sum(move_probs.values())
            if total_prob > 0:
                for move in move_probs:
                    move_probs[move] /= total_prob
            
            return move_probs
            
        except Exception as e:
            print(f"获取移动概率失败: {e}")
            # 回退到均匀分布
            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
            return {move: uniform_prob for move in legal_moves}
    
    def select_move(self, move_probs: Dict[str, float], temperature: float = 1.0) -> str:
        """根据概率分布选择移动"""
        if not move_probs:
            return ""
        
        moves = list(move_probs.keys())
        probs = list(move_probs.values())
        
        # 始终选择概率最大的移动（贪婪选择）
        return moves[np.argmax(probs)]
    
    def make_move(self, fen: str, move: str) -> str:
        """执行移动并返回新的FEN"""
        try:
            board = chess.Board(fen)
            move_obj = chess.Move.from_uci(move)
            
            if move_obj in board.legal_moves:
                board.push(move_obj)
                return board.fen()
            else:
                print(f"非法移动: {move}")
                return fen
        except Exception as e:
            print(f"执行移动失败: {e}")
            return fen
    
    def play_game(self, initial_fen: str, max_moves: int = 10, temperature: float = 1.0) -> Dict[str, Any]:
        """进行自对弈"""
        game_data = {
            'positions': [],
            'moves': [],
            'wdl_history': [],
            'move_probabilities': [],
            'current_player': 'white'
        }
        
        current_fen = initial_fen
        
        for move_num in range(max_moves):
            # 获取当前局面的合法移动
            legal_moves = self.get_legal_moves(current_fen)
            
            if not legal_moves:
                print(f"游戏结束于第{move_num}步")
                break
            
            # 获取模型评估
            win_prob, draw_prob, loss_prob = self.get_model_evaluation(current_fen)
            
            # 获取移动概率
            move_probs = self.get_move_probabilities(current_fen, legal_moves)
            
            # 选择移动
            selected_move = self.select_move(move_probs, temperature)
            
            # 记录当前状态
            game_data['positions'].append(current_fen)
            game_data['wdl_history'].append({
                'win': win_prob,
                'draw': draw_prob,
                'loss': loss_prob,
                'move_number': move_num + 1
            })
            game_data['move_probabilities'].append(move_probs)
            
            if selected_move:
                game_data['moves'].append(selected_move)
                current_fen = self.make_move(current_fen, selected_move)
                
                # 切换玩家
                game_data['current_player'] = 'black' if game_data['current_player'] == 'white' else 'white'
            else:
                print(f"第{move_num + 1}步无法选择移动")
                break
        
        # 添加最终位置
        game_data['positions'].append(current_fen)
        
        return game_data
    
    def analyze_position_sequence(self, positions: List[str]) -> List[Dict[str, Any]]:
        """分析位置序列，获取每个位置的详细评估"""
        analysis = []
        
        for i, fen in enumerate(positions):
            win_prob, draw_prob, loss_prob = self.get_model_evaluation(fen)
            legal_moves = self.get_legal_moves(fen)
            move_probs = self.get_move_probabilities(fen, legal_moves)
            
            analysis.append({
                'position_index': i,
                'fen': fen,
                'wdl': {
                    'win': win_prob,
                    'draw': draw_prob,
                    'loss': loss_prob
                },
                'legal_moves_count': len(legal_moves),
                'top_moves': sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            })
        
        return analysis


# 全局自对弈实例
_self_play_engine = None

def get_self_play_engine(model: HookedTransformer) -> ChessSelfPlay:
    """获取或创建自对弈引擎实例"""
    global _self_play_engine
    if _self_play_engine is None:
        _self_play_engine = ChessSelfPlay(model)
    return _self_play_engine

def run_self_play(initial_fen: str, max_moves: int = 10, temperature: float = 1.0, model: Optional[HookedTransformer] = None) -> Dict[str, Any]:
    """运行自对弈的公共接口"""
    if model is None:
        raise ValueError("Model is required for self-play")
    
    engine = get_self_play_engine(model)
    return engine.play_game(initial_fen, max_moves, temperature)


def analyze_game_positions(positions: List[str], model: Optional[HookedTransformer] = None) -> List[Dict[str, Any]]:
    """分析游戏位置的公共接口"""
    if model is None:
        raise ValueError("Model is required for position analysis")
    
    engine = get_self_play_engine(model)
    return engine.analyze_position_sequence(positions)