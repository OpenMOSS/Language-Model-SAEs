import torch
from typing import Dict, Any, List, Tuple, Optional
from transformer_lens import HookedTransformer
import sys
from pathlib import Path
import chess

# add project root to Python path
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
    """chess self-play engine"""
    
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.device = next(model.parameters()).device
        
        # initialize LeelaBoard (if available)
        if LEELA_BOARD_AVAILABLE and LeelaBoard is not None:
            self.lboard = LeelaBoard.from_fen(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                history_synthesis=True
            )
        else:
            self.lboard = None
        
        # cache the model output of the current position
        self.current_fen = None
        self.cached_outputs = None
    
    def fen_to_tensor(self, fen: str) -> str:
        """convert FEN string to model input tensor"""
        # here we need to implement the conversion based on the model input format
        # assume the model directly accepts the FEN string
        return fen
    
    def run_model_inference(self, fen: str) -> tuple:
        """run model inference and cache the result"""
        # if the FEN has not changed, return the cached result
        if self.current_fen == fen and self.cached_outputs is not None:
            print(f"using cached model output (FEN: {fen})")
            return self.cached_outputs
        
        print(f"running new model inference (FEN: {fen})")
        # run new inference
        with torch.no_grad():
            outputs, _ = self.model.run_with_cache(fen, prepend_bos=False)
        
        # print model output information for debugging
        if isinstance(outputs, (list, tuple)):
            print(f"model output type: {type(outputs)}, length: {len(outputs)}")
            for i, output in enumerate(outputs):
                if hasattr(output, 'shape'):
                    print(f"  outputs[{i}] shape: {output.shape}")
                else:
                    print(f"  outputs[{i}] type: {type(output)}")
        else:
            print(f"model output type: {type(outputs)}")
        
        # cache the result
        self.current_fen = fen
        self.cached_outputs = outputs
        
        return outputs
    
    def get_model_evaluation(self, fen: str) -> Tuple[float, float, float]:
        """get the model evaluation of the current position (WDL: Win, Draw, Loss) - directly return the current player win rate"""
        # use the cached model inference result
        outputs = self.run_model_inference(fen)
        
        # the model output is a list, containing three elements:
        # outputs[0]: logits, shape [1, 1858]
        # outputs[1]: WDL, shape [1, 3] - [current player win rate, draw rate, current player loss rate]
        # outputs[2]: other output, shape [1, 1]
        
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            wdl_tensor = outputs[1]  # get the WDL output
            if wdl_tensor.shape == torch.Size([1, 3]):
                # WDL is already a probability distribution, no need to softmax
                current_player_win = wdl_tensor[0][0].item()  # current player win rate
                draw_prob = wdl_tensor[0][1].item()  # draw rate
                current_player_loss = wdl_tensor[0][2].item()  # current player loss rate
                
                # directly return the current player win rate information, without flipping
                return current_player_win, draw_prob, current_player_loss
            else:
                raise RuntimeError(f"WDL output shape incorrect: {wdl_tensor.shape}, expected [1, 3]")
        else:
            raise RuntimeError(
                f"Model output format incorrect, expected list or tuple containing at least 2 elements, got: {type(outputs)}"
            )
    
    def get_legal_moves(self, fen: str) -> List[str]:
        """get the legal moves of the current position"""
        try:
            board = chess.Board(fen)
            return [move.uci() for move in board.legal_moves]
        except Exception as e:
            print(f"get legal moves failed: {e}")
            return []
    
    def get_move_probabilities(self, fen: str, legal_moves: List[str]) -> Dict[str, float]:
        """get the probabilities of each legal move - according to the correct logic in the notebook"""
        try:
            # use the cached model inference result
            outputs = self.run_model_inference(fen)
            
            # the model output is a list, containing three elements:
            # outputs[0]: logits, shape [1, 1858] - move probability logits
            # outputs[1]: WDL, shape [1, 3]
            # outputs[2]: other output, shape [1, 1]
            
            move_probs = {}
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 1:
                policy_logits = outputs[0][0]  # get (1858,) from (1, 1858)
                
                if policy_logits.shape == torch.Size([1858,]):
                    print(f"policy output shape: {policy_logits.shape}")
                    print(f"score range: [{policy_logits.min():.3f}, {policy_logits.max():.3f}]")
                    
                    if self.lboard and LEELA_BOARD_AVAILABLE:
                        # important: update LeelaBoard state to match current FEN
                        try:
                            # create a new LeelaBoard instance to match current FEN
                            temp_lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
                            print(f"current FEN: {fen}")
                            print(f"LeelaBoard player: {'white' if temp_lboard.turn else 'black'}")
                            
                            # sort all indices by score from high to low
                            sorted_indices = torch.argsort(policy_logits, descending=True)
                            
                            # find the probabilities of all legal moves
                            legal_uci_set = set(legal_moves)
                            print(f"legal moves count: {len(legal_moves)}")
                            
                            # iterate over all indices, find legal moves and record their logits
                            found_moves = []
                            for idx in sorted_indices:
                                try:
                                    uci = temp_lboard.idx2uci(idx.item())
                                    if uci in legal_uci_set:
                                        # record the logit of this legal move
                                        move_probs[uci] = policy_logits[idx].item()
                                        found_moves.append((uci, idx.item(), policy_logits[idx].item()))
                                        print(f"Move {uci} -> idx {idx.item()}, logit {policy_logits[idx].item():.4f}")
                                        
                                        # only show the detailed information of the top 5 best moves
                                        if len(found_moves) >= 5:
                                            break
                                except Exception as move_error:
                                    # skip invalid indices
                                    continue
                            
                            print(f"found {len(found_moves)} legal moves")
                            
                            # if legal moves are found, calculate the softmax probabilities
                            if move_probs:
                                # get the logits of all legal moves
                                legal_logits = []
                                valid_moves = []
                                
                                for move in legal_moves:
                                    try:
                                        move_idx = temp_lboard.uci2idx(move)
                                        if move_idx is not None:
                                            legal_logits.append(policy_logits[move_idx].item())
                                            valid_moves.append(move)
                                    except Exception:
                                        continue
                                
                                if legal_logits:
                                    legal_logits_tensor = torch.tensor(legal_logits)
                                    # calculate the softmax probabilities
                                    legal_probs = torch.softmax(legal_logits_tensor, dim=-1)
                                    
                                    # update the probability dictionary
                                    move_probs = {}
                                    for i, move in enumerate(valid_moves):
                                        move_probs[move] = legal_probs[i].item()
                                else:
                                    print("cannot get the logits of any legal moves")
                                    uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
                                    for move in legal_moves:
                                        move_probs[move] = uniform_prob
                            else:
                                print("no legal moves found, using uniform distribution")
                                uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
                                for move in legal_moves:
                                    move_probs[move] = uniform_prob
                                    
                        except Exception as lboard_error:
                            print(f"LeelaBoard processing failed: {lboard_error}")
                            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
                            for move in legal_moves:
                                move_probs[move] = uniform_prob
                    else:
                        print("LeelaBoard unavailable, using uniform distribution")
                        # fallback method: uniform distribution
                        uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
                        for move in legal_moves:
                            move_probs[move] = uniform_prob
                else:
                    print(f"Policy logits output shape incorrect: {policy_logits.shape}, expected [1858]")
                    # fallback method: uniform distribution
                    uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
                    for move in legal_moves:
                        move_probs[move] = uniform_prob
            else:
                print(f"model output format incorrect, expected list or tuple containing at least 1 element, got: {type(outputs)}")
                # fallback method: uniform distribution
                uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
                for move in legal_moves:
                    move_probs[move] = uniform_prob
            
            if move_probs:
                sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
                print(f"Top 5 moves with probabilities: {sorted_moves[:5]}")
            
            return move_probs
            
        except Exception as e:
            print(f"get move probabilities failed: {e}")
            # fallback method: uniform distribution
            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
            return {move: uniform_prob for move in legal_moves}
    
    def select_move(self, move_probs: Dict[str, float], temperature: float = 1.0) -> str:
        if not move_probs:
            return ""
        
        # sort by probability from high to low
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_moves:
            return ""
        
        # select the move with the highest probability
        selected_move, selected_prob = sorted_moves[0]
        
        print(f"Selected move: {selected_move} (prob: {selected_prob:.4f})")
        print(f"Top 5 moves: {sorted_moves[:5]}")
        
        return selected_move
    
    def make_move(self, fen: str, move: str) -> str:
        """make move and return new FEN"""
        try:
            board = chess.Board(fen)
            move_obj = chess.Move.from_uci(move)
            
            if move_obj in board.legal_moves:
                board.push(move_obj)
                return board.fen()
            else:
                print(f"invalid move: {move}")
                return fen
        except Exception as e:
            print(f"make move failed: {e}")
            return fen
    
    def play_game(self, initial_fen: str, max_moves: int = 10, temperature: float = 1.0) -> Dict[str, Any]:
        """play game"""
        game_data = {
            'positions': [],
            'moves': [],
            'wdl_history': [],
            'move_probabilities': [],
            'current_player': 'white'
        }
        
        current_fen = initial_fen
        print(f"initial FEN: {current_fen}")    
        
        for move_num in range(max_moves):
            print(f"\n--- step {move_num + 1} ---")
            print(f"current FEN: {current_fen}")
            
            # get legal moves of the current position
            legal_moves = self.get_legal_moves(current_fen)
            print(f"legal moves count: {len(legal_moves)}")
            print(f"legal moves: {legal_moves[:10]}...")  # only show the first 10 moves
            
            if not legal_moves:
                print(f"game ended at step {move_num}")
                break
            
            # get model evaluation
            win_prob, draw_prob, loss_prob = self.get_model_evaluation(current_fen)
            print(f"model evaluation: win rate={win_prob:.3f}, draw rate={draw_prob:.3f}, loss rate={loss_prob:.3f}")
            
            # get move probabilities
            move_probs = self.get_move_probabilities(current_fen, legal_moves)
            
            # select move
            selected_move = self.select_move(move_probs, temperature)
            
            # record current state
            game_data['positions'].append(current_fen)
            game_data['wdl_history'].append({
                'win': win_prob,
                'draw': draw_prob,
                'loss': loss_prob,
                'move_number': move_num + 1
            })
            game_data['move_probabilities'].append(move_probs)
            
            if selected_move:
                print(f"make move: {selected_move}")
                game_data['moves'].append(selected_move)
                current_fen = self.make_move(current_fen, selected_move)
                print(f"after move FEN: {current_fen}")
                
                # switch player
                game_data['current_player'] = 'black' if game_data['current_player'] == 'white' else 'white'
                print(f"current player: {game_data['current_player']}")
            else:
                print(f"step {move_num + 1} cannot select move")
                break
        
        # add final position
        game_data['positions'].append(current_fen)
        print("\n=== self-play ended ===")
        print(f"final FEN: {current_fen}")
        print(f"total steps: {len(game_data['moves'])}")
        
        return game_data
    
    def analyze_position_sequence(self, positions: List[str]) -> List[Dict[str, Any]]:
        """analyze position sequence, get detailed evaluation for each position"""
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


# global self-play engine instance
_self_play_engine = None

def get_self_play_engine(model: HookedTransformer) -> ChessSelfPlay:
    """get or create self-play engine instance"""
    global _self_play_engine
    if _self_play_engine is None:
        _self_play_engine = ChessSelfPlay(model)
    return _self_play_engine

def run_self_play(initial_fen: str, max_moves: int = 10, temperature: float = 1.0, model: Optional[HookedTransformer] = None) -> Dict[str, Any]:
    """run self-play public interface"""
    if model is None:
        raise ValueError("Model is required for self-play")
    
    engine = get_self_play_engine(model)
    return engine.play_game(initial_fen, max_moves, temperature)


def analyze_game_positions(positions: List[str], model: Optional[HookedTransformer] = None) -> List[Dict[str, Any]]:
    """analyze game positions public interface"""
    if model is None:
        raise ValueError("Model is required for position analysis")
    
    engine = get_self_play_engine(model)
    return engine.analyze_position_sequence(positions)