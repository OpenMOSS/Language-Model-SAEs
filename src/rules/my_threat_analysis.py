import chess

__all__ = ["analyze_threats_by_lesser_pieces", "get_threat_details"]


def analyze_threats_by_lesser_pieces(fen: str) -> dict:
    """分析棋子受到较小棋子的威胁情况。
    
    根据Stockfish的威胁分析逻辑：
    - threatByLesser[KNIGHT] = threatByLesser[BISHOP] = 被对方兵攻击
    - threatByLesser[ROOK] = 被对方马或象攻击
    - threatByLesser[QUEEN] = 被对方车攻击 或 被对方马/象攻击
    
    Args:
        fen: 合法的 FEN 字符串
        
    Returns:
        dict: 包含各种棋子威胁分析的结果
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"无效的 FEN: {fen}") from exc
    
    own_color = board.turn
    opponent_color = not own_color
    
    result = {
        "knight_threats": [],
        "bishop_threats": [],
        "rook_threats": [],
        "queen_threats": [],
        "summary": {
            "knight_under_threat": False,
            "bishop_under_threat": False,
            "rook_under_threat": False,
            "queen_under_threat": False,
            "total_threats": 0
        }
    }
    
    # 分析马受到兵威胁
    knight_squares = board.pieces(chess.KNIGHT, own_color)
    for knight_sq in knight_squares:
        pawn_attackers = board.attackers(opponent_color, knight_sq) & board.pieces(chess.PAWN, opponent_color)
        if pawn_attackers:
            knight_square_name = chess.square_name(knight_sq)
            attacker_squares = [chess.square_name(sq) for sq in pawn_attackers]
            result["knight_threats"].append({
                "piece_square": knight_square_name,
                "threat_type": "pawn_attack",
                "attacker_squares": attacker_squares,
                "attacker_pieces": ["pawn"] * len(attacker_squares)
            })
    
    # 分析象受到兵威胁
    bishop_squares = board.pieces(chess.BISHOP, own_color)
    for bishop_sq in bishop_squares:
        pawn_attackers = board.attackers(opponent_color, bishop_sq) & board.pieces(chess.PAWN, opponent_color)
        if pawn_attackers:
            bishop_square_name = chess.square_name(bishop_sq)
            attacker_squares = [chess.square_name(sq) for sq in pawn_attackers]
            result["bishop_threats"].append({
                "piece_square": bishop_square_name,
                "threat_type": "pawn_attack",
                "attacker_squares": attacker_squares,
                "attacker_pieces": ["pawn"] * len(attacker_squares)
            })
    
    # 分析车受到马或象威胁
    rook_squares = board.pieces(chess.ROOK, own_color)
    for rook_sq in rook_squares:
        knight_attackers = board.attackers(opponent_color, rook_sq) & board.pieces(chess.KNIGHT, opponent_color)
        bishop_attackers = board.attackers(opponent_color, rook_sq) & board.pieces(chess.BISHOP, opponent_color)
        
        if knight_attackers or bishop_attackers:
            rook_square_name = chess.square_name(rook_sq)
            threat_details = {
                "piece_square": rook_square_name,
                "threat_type": "minor_piece_attack",
                "attacker_squares": [],
                "attacker_pieces": []
            }
            
            if knight_attackers:
                knight_squares_list = [chess.square_name(sq) for sq in knight_attackers]
                threat_details["attacker_squares"].extend(knight_squares_list)
                threat_details["attacker_pieces"].extend(["knight"] * len(knight_squares_list))
            
            if bishop_attackers:
                bishop_squares_list = [chess.square_name(sq) for sq in bishop_attackers]
                threat_details["attacker_squares"].extend(bishop_squares_list)
                threat_details["attacker_pieces"].extend(["bishop"] * len(bishop_squares_list))
            
            result["rook_threats"].append(threat_details)
    
    # 分析后受到车威胁或马/象威胁
    queen_squares = board.pieces(chess.QUEEN, own_color)
    for queen_sq in queen_squares:
        rook_attackers = board.attackers(opponent_color, queen_sq) & board.pieces(chess.ROOK, opponent_color)
        knight_attackers = board.attackers(opponent_color, queen_sq) & board.pieces(chess.KNIGHT, opponent_color)
        bishop_attackers = board.attackers(opponent_color, queen_sq) & board.pieces(chess.BISHOP, opponent_color)
        
        if rook_attackers or knight_attackers or bishop_attackers:
            queen_square_name = chess.square_name(queen_sq)
            threat_details = {
                "piece_square": queen_square_name,
                "threat_type": "major_piece_attack",
                "attacker_squares": [],
                "attacker_pieces": []
            }
            
            if rook_attackers:
                rook_squares_list = [chess.square_name(sq) for sq in rook_attackers]
                threat_details["attacker_squares"].extend(rook_squares_list)
                threat_details["attacker_pieces"].extend(["rook"] * len(rook_squares_list))
            
            if knight_attackers:
                knight_squares_list = [chess.square_name(sq) for sq in knight_attackers]
                threat_details["attacker_squares"].extend(knight_squares_list)
                threat_details["attacker_pieces"].extend(["knight"] * len(knight_squares_list))
            
            if bishop_attackers:
                bishop_squares_list = [chess.square_name(sq) for sq in bishop_attackers]
                threat_details["attacker_squares"].extend(bishop_squares_list)
                threat_details["attacker_pieces"].extend(["bishop"] * len(bishop_squares_list))
            
            result["queen_threats"].append(threat_details)
    
    # 更新摘要信息
    result["summary"]["knight_under_threat"] = len(result["knight_threats"]) > 0
    result["summary"]["bishop_under_threat"] = len(result["bishop_threats"]) > 0
    result["summary"]["rook_under_threat"] = len(result["rook_threats"]) > 0
    result["summary"]["queen_under_threat"] = len(result["queen_threats"]) > 0
    result["summary"]["total_threats"] = (
        len(result["knight_threats"]) + 
        len(result["bishop_threats"]) + 
        len(result["rook_threats"]) + 
        len(result["queen_threats"])
    )
    
    return result


def get_threat_details(fen: str) -> dict:
    """获取详细的威胁分析信息，包括威胁的严重程度评估。
    
    Args:
        fen: FEN 字符串
        
    Returns:
        dict: 详细的威胁分析结果
    """
    basic_analysis = analyze_threats_by_lesser_pieces(fen)
    
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"无效的 FEN: {fen}") from exc
    
    own_color = board.turn
    opponent_color = not own_color
    
    # 添加威胁严重程度评估
    for piece_type, threats in [
        ("knight", basic_analysis["knight_threats"]),
        ("bishop", basic_analysis["bishop_threats"]),
        ("rook", basic_analysis["rook_threats"]),
        ("queen", basic_analysis["queen_threats"])
    ]:
        for threat in threats:
            piece_square = chess.parse_square(threat["piece_square"])
            
            # 评估威胁严重程度
            threat["severity"] = "low"
            threat["can_defend"] = False
            threat["defender_squares"] = []
            
            # 检查是否有己方棋子可以保护
            defenders = board.attackers(own_color, piece_square)
            if defenders:
                threat["can_defend"] = True
                threat["defender_squares"] = [chess.square_name(sq) for sq in defenders]
                
                # 根据攻击者和防守者的数量评估严重程度
                attacker_count = len(threat["attacker_squares"])
                defender_count = len(threat["defender_squares"])
                
                if attacker_count > defender_count:
                    threat["severity"] = "high"
                elif attacker_count == defender_count:
                    threat["severity"] = "medium"
                else:
                    threat["severity"] = "low"
            
            # 检查是否可以通过移动来避免威胁
            threat["can_escape"] = False
            threat["escape_squares"] = []
            
            # 获取该棋子的所有合法移动
            for move in board.legal_moves:
                if move.from_square == piece_square:
                    # 临时执行移动检查是否安全
                    board.push(move)
                    new_square = move.to_square
                    attackers_after_move = board.attackers(opponent_color, new_square)
                    board.pop()
                    
                    if not attackers_after_move:
                        threat["can_escape"] = True
                        threat["escape_squares"].append(chess.square_name(new_square))
    
    return basic_analysis


def get_threat_summary(fen: str) -> dict:
    """获取威胁分析的简要摘要。
    
    Args:
        fen: FEN 字符串
        
    Returns:
        dict: 威胁分析摘要
    """
    analysis = analyze_threats_by_lesser_pieces(fen)
    
    summary = {
        "has_threats": analysis["summary"]["total_threats"] > 0,
        "threat_count": analysis["summary"]["total_threats"],
        "threatened_pieces": [],
        "most_critical_threat": None
    }
    
    # 收集所有受威胁的棋子
    for piece_type, threats in [
        ("knight", analysis["knight_threats"]),
        ("bishop", analysis["bishop_threats"]),
        ("rook", analysis["rook_threats"]),
        ("queen", analysis["queen_threats"])
    ]:
        for threat in threats:
            summary["threatened_pieces"].append({
                "piece_type": piece_type,
                "square": threat["piece_square"],
                "threat_type": threat["threat_type"],
                "attacker_count": len(threat["attacker_squares"])
            })
    
    # 找出最严重的威胁（后 > 车 > 象/马）
    piece_values = {"queen": 4, "rook": 3, "bishop": 2, "knight": 2}
    max_value = 0
    
    for piece in summary["threatened_pieces"]:
        value = piece_values.get(piece["piece_type"], 0)
        if value > max_value:
            max_value = value
            summary["most_critical_threat"] = piece
    
    return summary 