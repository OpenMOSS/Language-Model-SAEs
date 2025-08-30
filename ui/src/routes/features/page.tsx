import { AppNavbar } from "@/components/app/navbar";
import { SectionNavigator } from "@/components/app/section-navigator";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FeatureSchema } from "@/types/feature";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import React, { useEffect, useState, useMemo, useCallback, Suspense, lazy, useRef } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { useAsyncFn, useMount, useDebounce } from "react-use";
import { z } from "zod";


const FeatureCard = lazy(() => import("@/components/feature/feature-card").then(module => ({ default: module.FeatureCard })));

// 全局计数器确保唯一ID
let boardCounter = 0;

// 全局分析状态管理
class AnalysisStateManager {
  private analysisStates = new Map<string, {
    stockfishAnalysis: any;
    isLoading: boolean;
    analysisStarted: boolean;
    analysisCompleted: boolean;
  }>();

  getAnalysisState(key: string) {
    return this.analysisStates.get(key) || {
      stockfishAnalysis: null,
      isLoading: false,
      analysisStarted: false,
      analysisCompleted: false
    };
  }

  setAnalysisState(key: string, state: {
    stockfishAnalysis: any;
    isLoading: boolean;
    analysisStarted: boolean;
    analysisCompleted: boolean;
  }) {
    this.analysisStates.set(key, state);
  }

  clear() {
    this.analysisStates.clear();
  }
}

// 全局分析状态管理器实例
const globalAnalysisStateManager = new AnalysisStateManager();

// 全局统计管理
interface RuleStatistics {
  totalBoards: number;
  analyzedBoards: number;
  // 己方棋子被抓
  is_rook_under_attack: number;
  is_knight_under_attack: number;
  is_bishop_under_attack: number;
  is_queen_under_attack: number;
  // 可以攻击对方棋子
  is_can_capture_rook: number;
  is_can_capture_knight: number;
  is_can_capture_bishop: number;
  is_can_capture_queen: number;
  // 王的状态
  is_king_in_check: number;
  is_checkmate: number;
  is_stalemate: number;
  // 模型推理移动棋子类型统计
  model_move_pawn: number;
  model_move_knight: number;
  model_move_bishop: number;
  model_move_rook: number;
  model_move_queen: number;
  model_move_king: number;
  // 棋盘位置热力图统计
  position_a1: number; position_a2: number; position_a3: number; position_a4: number;
  position_a5: number; position_a6: number; position_a7: number; position_a8: number;
  position_b1: number; position_b2: number; position_b3: number; position_b4: number;
  position_b5: number; position_b6: number; position_b7: number; position_b8: number;
  position_c1: number; position_c2: number; position_c3: number; position_c4: number;
  position_c5: number; position_c6: number; position_c7: number; position_c8: number;
  position_d1: number; position_d2: number; position_d3: number; position_d4: number;
  position_d5: number; position_d6: number; position_d7: number; position_d8: number;
  position_e1: number; position_e2: number; position_e3: number; position_e4: number;
  position_e5: number; position_e6: number; position_e7: number; position_e8: number;
  position_f1: number; position_f2: number; position_f3: number; position_f4: number;
  position_f5: number; position_f6: number; position_f7: number; position_f8: number;
  position_g1: number; position_g2: number; position_g3: number; position_g4: number;
  position_g5: number; position_g6: number; position_g7: number; position_g8: number;
  position_h1: number; position_h2: number; position_h3: number; position_h4: number;
  position_h5: number; position_h6: number; position_h7: number; position_h8: number;
  // 终点位置热力图统计
  to_position_a1: number; to_position_a2: number; to_position_a3: number; to_position_a4: number;
  to_position_a5: number; to_position_a6: number; to_position_a7: number; to_position_a8: number;
  to_position_b1: number; to_position_b2: number; to_position_b3: number; to_position_b4: number;
  to_position_b5: number; to_position_b6: number; to_position_b7: number; to_position_b8: number;
  to_position_c1: number; to_position_c2: number; to_position_c3: number; to_position_c4: number;
  to_position_c5: number; to_position_c6: number; to_position_c7: number; to_position_c8: number;
  to_position_d1: number; to_position_d2: number; to_position_d3: number; to_position_d4: number;
  to_position_d5: number; to_position_d6: number; to_position_d7: number; to_position_d8: number;
  to_position_e1: number; to_position_e2: number; to_position_e3: number; to_position_e4: number;
  to_position_e5: number; to_position_e6: number; to_position_e7: number; to_position_e8: number;
  to_position_f1: number; to_position_f2: number; to_position_f3: number; to_position_f4: number;
  to_position_f5: number; to_position_f6: number; to_position_f7: number; to_position_f8: number;
  to_position_g1: number; to_position_g2: number; to_position_g3: number; to_position_g4: number;
  to_position_g5: number; to_position_g6: number; to_position_g7: number; to_position_g8: number;
  to_position_h1: number; to_position_h2: number; to_position_h3: number; to_position_h4: number;
  to_position_h5: number; to_position_h6: number; to_position_h7: number; to_position_h8: number;
  // 走法类型分类统计
  move_pawn_push: number;          // 兵推进
  move_pawn_capture: number;       // 兵吃子
  move_piece_development: number;  // 子力出动
  move_piece_capture: number;      // 子力吃子
  move_castling: number;           // 王车易位
  move_king_move: number;          // 王移动
  move_check: number;              // 将军
  move_en_passant: number;         // 吃过路兵
  move_promotion: number;          // 兵升变
  // 游戏阶段分析
  phase_opening: number;           // 开局 (1-10步)
  phase_middlegame: number;        // 中局 (11-30步)
  phase_endgame: number;           // 残局 (31步以后)
  phase_early_opening: number;     // 早期开局 (1-5步)
  phase_late_opening: number;      // 晚期开局 (6-10步)
  phase_early_middlegame: number;  // 早期中局 (11-20步)
  phase_late_middlegame: number;   // 晚期中局 (21-30步)
  phase_early_endgame: number;     // 早期残局 (31-40步)
  phase_late_endgame: number;      // 晚期残局 (40步以后)
  // 激活位置棋子类型统计 - 己方棋子
  activated_own_pawn: number;      // 激活位置的己方兵
  activated_own_knight: number;    // 激活位置的己方马
  activated_own_bishop: number;    // 激活位置的己方象
  activated_own_rook: number;      // 激活位置的己方车
  activated_own_queen: number;     // 激活位置的己方后
  activated_own_king: number;      // 激活位置的己方王
  // 激活位置棋子类型统计 - 对方棋子
  activated_opp_pawn: number;      // 激活位置的对方兵
  activated_opp_knight: number;    // 激活位置的对方马
  activated_opp_bishop: number;    // 激活位置的对方象
  activated_opp_rook: number;      // 激活位置的对方车
  activated_opp_queen: number;     // 激活位置的对方后
  activated_opp_king: number;      // 激活位置的对方王
  // 激活位置空格
  activated_piece_empty: number;   // 激活位置的空格
}

interface MaterialStatistics {
  majorAdvantage: number;      // 大优 (差值>=5)
  moderateAdvantage: number;   // 中优 (差值3-4.9)
  minorAdvantage: number;      // 小优 (差值1-2.9)
  balanced: number;            // 均势 (差值0-0.9)
  minorDisadvantage: number;   // 小劣 (差值-1至-2.9)
  moderateDisadvantage: number; // 中劣 (差值-3至-4.9)
  majorDisadvantage: number;   // 大劣 (差值<=-5)
  averageActivePlayerMaterial: number;  // 行棋方平均物质
  averageOpponentMaterial: number;      // 对手平均物质
  averageMaterialDifference: number;    // 平均物质差值
}

interface WDLStatistics {
  decisive: number;           // 必胜局面 (胜率>=95%)
  majorAdvantage: number;     // 大优 (胜率80-94%)
  moderateAdvantage: number;  // 中优 (胜率65-79%)
  minorAdvantage: number;     // 小优 (胜率55-64%)
  balanced: number;           // 均势 (胜率45-54%)
  minorDisadvantage: number;  // 小劣 (胜率35-44%)
  moderateDisadvantage: number; // 中劣 (胜率20-34%)
  majorDisadvantage: number;  // 大劣 (胜率5-19%)
  hopeless: number;          // 必败局面 (胜率<=4%)
  averageWinProb: number;    // 行棋方平均胜率
  averageDrawProb: number;   // 平均和棋率
  averageLossProb: number;   // 行棋方平均负率
  matePositions: number;     // 强制将死局面
}

interface ComprehensiveStatistics extends RuleStatistics {
  material: MaterialStatistics;
  wdl: WDLStatistics;
}

// 创建统计管理器
class StatisticsManager {
  private statistics: ComprehensiveStatistics = {
    totalBoards: 0,
    analyzedBoards: 0,
    is_rook_under_attack: 0,
    is_knight_under_attack: 0,
    is_bishop_under_attack: 0,
    is_queen_under_attack: 0,
    is_can_capture_rook: 0,
    is_can_capture_knight: 0,
    is_can_capture_bishop: 0,
    is_can_capture_queen: 0,
    is_king_in_check: 0,
    is_checkmate: 0,
    is_stalemate: 0,
    // 模型推理移动棋子类型统计
    model_move_pawn: 0,
    model_move_knight: 0,
    model_move_bishop: 0,
    model_move_rook: 0,
    model_move_queen: 0,
    model_move_king: 0,
    // 棋盘位置热力图统计 - 初始化所有64个位置
    position_a1: 0, position_a2: 0, position_a3: 0, position_a4: 0,
    position_a5: 0, position_a6: 0, position_a7: 0, position_a8: 0,
    position_b1: 0, position_b2: 0, position_b3: 0, position_b4: 0,
    position_b5: 0, position_b6: 0, position_b7: 0, position_b8: 0,
    position_c1: 0, position_c2: 0, position_c3: 0, position_c4: 0,
    position_c5: 0, position_c6: 0, position_c7: 0, position_c8: 0,
    position_d1: 0, position_d2: 0, position_d3: 0, position_d4: 0,
    position_d5: 0, position_d6: 0, position_d7: 0, position_d8: 0,
    position_e1: 0, position_e2: 0, position_e3: 0, position_e4: 0,
    position_e5: 0, position_e6: 0, position_e7: 0, position_e8: 0,
    position_f1: 0, position_f2: 0, position_f3: 0, position_f4: 0,
    position_f5: 0, position_f6: 0, position_f7: 0, position_f8: 0,
    position_g1: 0, position_g2: 0, position_g3: 0, position_g4: 0,
    position_g5: 0, position_g6: 0, position_g7: 0, position_g8: 0,
    position_h1: 0, position_h2: 0, position_h3: 0, position_h4: 0,
    position_h5: 0, position_h6: 0, position_h7: 0, position_h8: 0,
    // 终点位置热力图统计 - 初始化所有64个位置
    to_position_a1: 0, to_position_a2: 0, to_position_a3: 0, to_position_a4: 0,
    to_position_a5: 0, to_position_a6: 0, to_position_a7: 0, to_position_a8: 0,
    to_position_b1: 0, to_position_b2: 0, to_position_b3: 0, to_position_b4: 0,
    to_position_b5: 0, to_position_b6: 0, to_position_b7: 0, to_position_b8: 0,
    to_position_c1: 0, to_position_c2: 0, to_position_c3: 0, to_position_c4: 0,
    to_position_c5: 0, to_position_c6: 0, to_position_c7: 0, to_position_c8: 0,
    to_position_d1: 0, to_position_d2: 0, to_position_d3: 0, to_position_d4: 0,
    to_position_d5: 0, to_position_d6: 0, to_position_d7: 0, to_position_d8: 0,
    to_position_e1: 0, to_position_e2: 0, to_position_e3: 0, to_position_e4: 0,
    to_position_e5: 0, to_position_e6: 0, to_position_e7: 0, to_position_e8: 0,
    to_position_f1: 0, to_position_f2: 0, to_position_f3: 0, to_position_f4: 0,
    to_position_f5: 0, to_position_f6: 0, to_position_f7: 0, to_position_f8: 0,
    to_position_g1: 0, to_position_g2: 0, to_position_g3: 0, to_position_g4: 0,
    to_position_g5: 0, to_position_g6: 0, to_position_g7: 0, to_position_g8: 0,
    to_position_h1: 0, to_position_h2: 0, to_position_h3: 0, to_position_h4: 0,
    to_position_h5: 0, to_position_h6: 0, to_position_h7: 0, to_position_h8: 0,
    // 走法类型分类统计
    move_pawn_push: 0,
    move_pawn_capture: 0,
    move_piece_development: 0,
    move_piece_capture: 0,
    move_castling: 0,
    move_king_move: 0,
    move_check: 0,
    move_en_passant: 0,
    move_promotion: 0,
    // 游戏阶段分析
    phase_opening: 0,
    phase_middlegame: 0,
    phase_endgame: 0,
    phase_early_opening: 0,
    phase_late_opening: 0,
    phase_early_middlegame: 0,
    phase_late_middlegame: 0,
    phase_early_endgame: 0,
    phase_late_endgame: 0,
    // 激活位置棋子类型统计 - 己方棋子
    activated_own_pawn: 0,
    activated_own_knight: 0,
    activated_own_bishop: 0,
    activated_own_rook: 0,
    activated_own_queen: 0,
    activated_own_king: 0,
    // 激活位置棋子类型统计 - 对方棋子
    activated_opp_pawn: 0,
    activated_opp_knight: 0,
    activated_opp_bishop: 0,
    activated_opp_rook: 0,
    activated_opp_queen: 0,
    activated_opp_king: 0,
    // 激活位置空格
    activated_piece_empty: 0,
    material: {
      majorAdvantage: 0,
      moderateAdvantage: 0,
      minorAdvantage: 0,
      balanced: 0,
      minorDisadvantage: 0,
      moderateDisadvantage: 0,
      majorDisadvantage: 0,
      averageActivePlayerMaterial: 0,
      averageOpponentMaterial: 0,
      averageMaterialDifference: 0
    },
    wdl: {
      decisive: 0,
      majorAdvantage: 0,
      moderateAdvantage: 0,
      minorAdvantage: 0,
      balanced: 0,
      minorDisadvantage: 0,
      moderateDisadvantage: 0,
      majorDisadvantage: 0,
      hopeless: 0,
      averageWinProb: 0,
      averageDrawProb: 0,
      averageLossProb: 0,
      matePositions: 0
    }
  };
  private listeners: ((stats: ComprehensiveStatistics) => void)[] = [];
  
  // 用于计算平均值的累积数据
  private materialAccumulator = {
    totalWhiteMaterial: 0,
    totalBlackMaterial: 0,
    totalMaterialDifference: 0,
    count: 0
  };
  
  private wdlAccumulator = {
    totalWinProb: 0,
    totalDrawProb: 0,
    totalLossProb: 0,
    count: 0
  };
  private maxBoardsLimit: number = 0;

  reset(maxBoards: number = 0) {
    this.maxBoardsLimit = maxBoards;
    // 只有在maxBoards为0时才重置统计数据
    if (maxBoards === 0) {
    this.statistics = {
      totalBoards: 0,
      analyzedBoards: 0,
      is_rook_under_attack: 0,
      is_knight_under_attack: 0,
      is_bishop_under_attack: 0,
      is_queen_under_attack: 0,
      is_can_capture_rook: 0,
      is_can_capture_knight: 0,
      is_can_capture_bishop: 0,
      is_can_capture_queen: 0,
      is_king_in_check: 0,
      is_checkmate: 0,
      is_stalemate: 0,
      // 模型推理移动棋子类型统计
      model_move_pawn: 0,
      model_move_knight: 0,
      model_move_bishop: 0,
      model_move_rook: 0,
      model_move_queen: 0,
      model_move_king: 0,
      
      // 棋盘位置热力图统计 - 初始化所有64个位置
      position_a1: 0, position_a2: 0, position_a3: 0, position_a4: 0,
      position_a5: 0, position_a6: 0, position_a7: 0, position_a8: 0,
      position_b1: 0, position_b2: 0, position_b3: 0, position_b4: 0,
      position_b5: 0, position_b6: 0, position_b7: 0, position_b8: 0,
      position_c1: 0, position_c2: 0, position_c3: 0, position_c4: 0,
      position_c5: 0, position_c6: 0, position_c7: 0, position_c8: 0,
      position_d1: 0, position_d2: 0, position_d3: 0, position_d4: 0,
      position_d5: 0, position_d6: 0, position_d7: 0, position_d8: 0,
      position_e1: 0, position_e2: 0, position_e3: 0, position_e4: 0,
      position_e5: 0, position_e6: 0, position_e7: 0, position_e8: 0,
      position_f1: 0, position_f2: 0, position_f3: 0, position_f4: 0,
      position_f5: 0, position_f6: 0, position_f7: 0, position_f8: 0,
      position_g1: 0, position_g2: 0, position_g3: 0, position_g4: 0,
      position_g5: 0, position_g6: 0, position_g7: 0, position_g8: 0,
      position_h1: 0, position_h2: 0, position_h3: 0, position_h4: 0,
      position_h5: 0, position_h6: 0, position_h7: 0, position_h8: 0,
      // 终点位置热力图统计 - 初始化所有64个位置
      to_position_a1: 0, to_position_a2: 0, to_position_a3: 0, to_position_a4: 0,
      to_position_a5: 0, to_position_a6: 0, to_position_a7: 0, to_position_a8: 0,
      to_position_b1: 0, to_position_b2: 0, to_position_b3: 0, to_position_b4: 0,
      to_position_b5: 0, to_position_b6: 0, to_position_b7: 0, to_position_b8: 0,
      to_position_c1: 0, to_position_c2: 0, to_position_c3: 0, to_position_c4: 0,
      to_position_c5: 0, to_position_c6: 0, to_position_c7: 0, to_position_c8: 0,
      to_position_d1: 0, to_position_d2: 0, to_position_d3: 0, to_position_d4: 0,
      to_position_d5: 0, to_position_d6: 0, to_position_d7: 0, to_position_d8: 0,
      to_position_e1: 0, to_position_e2: 0, to_position_e3: 0, to_position_e4: 0,
      to_position_e5: 0, to_position_e6: 0, to_position_e7: 0, to_position_e8: 0,
      to_position_f1: 0, to_position_f2: 0, to_position_f3: 0, to_position_f4: 0,
      to_position_f5: 0, to_position_f6: 0, to_position_f7: 0, to_position_f8: 0,
      to_position_g1: 0, to_position_g2: 0, to_position_g3: 0, to_position_g4: 0,
      to_position_g5: 0, to_position_g6: 0, to_position_g7: 0, to_position_g8: 0,
      to_position_h1: 0, to_position_h2: 0, to_position_h3: 0, to_position_h4: 0,
      to_position_h5: 0, to_position_h6: 0, to_position_h7: 0, to_position_h8: 0,
      // 走法类型分类统计
      move_pawn_push: 0,
      move_pawn_capture: 0,
      move_piece_development: 0,
      move_piece_capture: 0,
      move_castling: 0,
      move_king_move: 0,
      move_check: 0,
      move_en_passant: 0,
      move_promotion: 0,
      // 游戏阶段分析
      phase_opening: 0,
      phase_middlegame: 0,
      phase_endgame: 0,
      phase_early_opening: 0,
      phase_late_opening: 0,
      phase_early_middlegame: 0,
      phase_late_middlegame: 0,
      phase_early_endgame: 0,
      phase_late_endgame: 0,
      // 激活位置棋子类型统计 - 己方棋子
      activated_own_pawn: 0,
      activated_own_knight: 0,
      activated_own_bishop: 0,
      activated_own_rook: 0,
      activated_own_queen: 0,
      activated_own_king: 0,
      // 激活位置棋子类型统计 - 对方棋子
      activated_opp_pawn: 0,
      activated_opp_knight: 0,
      activated_opp_bishop: 0,
      activated_opp_rook: 0,
      activated_opp_queen: 0,
      activated_opp_king: 0,
      // 激活位置空格
      activated_piece_empty: 0,
      material: {
        majorAdvantage: 0,
        moderateAdvantage: 0,
        minorAdvantage: 0,
        balanced: 0,
        minorDisadvantage: 0,
        moderateDisadvantage: 0,
        majorDisadvantage: 0,
        averageActivePlayerMaterial: 0,
        averageOpponentMaterial: 0,
        averageMaterialDifference: 0
      },
      wdl: {
        decisive: 0,
        majorAdvantage: 0,
        moderateAdvantage: 0,
        minorAdvantage: 0,
        balanced: 0,
        minorDisadvantage: 0,
        moderateDisadvantage: 0,
        majorDisadvantage: 0,
        hopeless: 0,
        averageWinProb: 0,
        averageDrawProb: 0,
        averageLossProb: 0,
        matePositions: 0
      }
    };
    
    // 重置累积器
    this.materialAccumulator = { totalWhiteMaterial: 0, totalBlackMaterial: 0, totalMaterialDifference: 0, count: 0 };
    this.wdlAccumulator = { totalWinProb: 0, totalDrawProb: 0, totalLossProb: 0, count: 0 };
    
    console.log(`统计重置，最大棋盘限制: ${maxBoards}`);
    this.notifyListeners();
    }
  }

  setTotalBoards(count: number) {
    this.statistics.totalBoards = count;
    console.log(`设置总棋盘数: ${count}`);
    this.notifyListeners();
  }

  incrementTotal() {
    // 这个方法保留是为了兼容性，但现在主要使用 setTotalBoards
    this.statistics.totalBoards++;
    this.notifyListeners();
  }

  updateAnalysis(analysisData: any) {
    this.statistics.analyzedBoards++;
    
    // 更新规则统计
    const rules = analysisData.rules;
    // 己方棋子被抓
    if (rules?.is_rook_under_attack) this.statistics.is_rook_under_attack++;
    if (rules?.is_knight_under_attack) this.statistics.is_knight_under_attack++;
    if (rules?.is_bishop_under_attack) this.statistics.is_bishop_under_attack++;
    if (rules?.is_queen_under_attack) this.statistics.is_queen_under_attack++;
    // 可以攻击对方棋子
    if (rules?.is_can_capture_rook) this.statistics.is_can_capture_rook++;
    if (rules?.is_can_capture_knight) this.statistics.is_can_capture_knight++;
    if (rules?.is_can_capture_bishop) this.statistics.is_can_capture_bishop++;
    if (rules?.is_can_capture_queen) this.statistics.is_can_capture_queen++;
    // 王的状态
    if (rules?.is_king_in_check) this.statistics.is_king_in_check++;
    if (rules?.is_checkmate) this.statistics.is_checkmate++;
    if (rules?.is_stalemate) this.statistics.is_stalemate++;
    
    // 模型推理移动棋子类型统计
    const modelAnalysis = analysisData.model;
    if (modelAnalysis && !modelAnalysis.error && modelAnalysis.best_move) {
      const bestMove = modelAnalysis.best_move;
      if (bestMove && bestMove.length >= 4) {
        const fromSquare = bestMove.substring(0, 2);
        const toSquare = bestMove.substring(2, 4);
        
        // 从FEN解析棋盘状态
        const fen = analysisData.fen;
        if (fen) {
          const fenParts = fen.split(' ');
          const boardFen = fenParts[0];
          const activeColor = fenParts[1];
          
          // 解析棋盘
          const board: string[][] = Array(8).fill(null).map(() => Array(8).fill(''));
          const rows = boardFen.split('/');
          
          if (rows.length === 8) {
            for (let i = 0; i < 8; i++) {
              let col = 0;
              for (const char of rows[i]) {
                if (/\d/.test(char)) {
                  col += parseInt(char);
                } else {
                  if (col < 8) {
                    board[i][col] = char;
                    col++;
                  }
                }
              }
            }
            
            // 获取起始位置的棋子
            const fromFile = fromSquare.charCodeAt(0) - 'a'.charCodeAt(0);
            const fromRank = 8 - parseInt(fromSquare[1]);
            
            if (fromRank >= 0 && fromRank < 8 && fromFile >= 0 && fromFile < 8) {
              const piece = board[fromRank][fromFile];
              if (piece) {
                // 根据棋子类型统计
                const pieceLower = piece.toLowerCase();
                switch (pieceLower) {
                  case 'p':
                    this.statistics.model_move_pawn++;
                    break;
                  case 'n':
                    this.statistics.model_move_knight++;
                    break;
                  case 'b':
                    this.statistics.model_move_bishop++;
                    break;
                  case 'r':
                    this.statistics.model_move_rook++;
                    break;
                  case 'q':
                    this.statistics.model_move_queen++;
                    break;
                  case 'k':
                    this.statistics.model_move_king++;
                    break;
                }
              }
            }
          }
        }
      }
    }
    
    // 棋盘位置热力图统计
    if (modelAnalysis && !modelAnalysis.error && modelAnalysis.best_move) {
      const bestMove = modelAnalysis.best_move;
      if (bestMove && bestMove.length >= 4) {
        const fromSquare = bestMove.substring(0, 2);
        const toSquare = bestMove.substring(2, 4);
        
        // 从FEN解析行棋方
        const fen = analysisData.fen;
        let activeColor = 'w'; // 默认为白方
        if (fen) {
          const fenParts = fen.split(' ');
          activeColor = fenParts[1] || 'w';
        }
        
        // 统计起点位置（相对于行棋方）
        let normalizedFromSquare = fromSquare;
        if (activeColor === 'b') {
          // 黑方行棋时，只进行上下翻转（行翻转）
          const rank = parseInt(fromSquare[1]);
          const flippedRank = 9 - rank; // 翻转行
          normalizedFromSquare = `${fromSquare[0]}${flippedRank}`;
        }
        
        const positionKey = `position_${normalizedFromSquare}`;
        if (this.statistics.hasOwnProperty(positionKey)) {
          (this.statistics as any)[positionKey]++;
        }
        
        // 统计终点位置（相对于行棋方）
        let normalizedToSquare = toSquare;
        if (activeColor === 'b') {
          // 黑方行棋时，只进行上下翻转（行翻转）
          const rank = parseInt(toSquare[1]);
          const flippedRank = 9 - rank; // 翻转行
          normalizedToSquare = `${toSquare[0]}${flippedRank}`;
        }
        
        const toPositionKey = `to_position_${normalizedToSquare}`;
        if (this.statistics.hasOwnProperty(toPositionKey)) {
          (this.statistics as any)[toPositionKey]++;
        }
      }
    }
    
    // 走法类型分类统计
    if (modelAnalysis && !modelAnalysis.error && modelAnalysis.best_move) {
      const bestMove = modelAnalysis.best_move;
      if (bestMove && bestMove.length >= 4) {
        const fromSquare = bestMove.substring(0, 2);
        const toSquare = bestMove.substring(2, 4);
        
        // 从FEN解析棋盘状态
        const fen = analysisData.fen;
        if (fen) {
          const fenParts = fen.split(' ');
          const boardFen = fenParts[0];
          const activeColor = fenParts[1];
          
          // 解析棋盘
          const board: string[][] = Array(8).fill(null).map(() => Array(8).fill(''));
          const rows = boardFen.split('/');
          
          if (rows.length === 8) {
            for (let i = 0; i < 8; i++) {
              let col = 0;
              for (const char of rows[i]) {
                if (/\d/.test(char)) {
                  col += parseInt(char);
                } else {
                  if (col < 8) {
                    board[i][col] = char;
                    col++;
                  }
                }
              }
            }
            
            // 获取起始位置的棋子
            const fromFile = fromSquare.charCodeAt(0) - 'a'.charCodeAt(0);
            const fromRank = 8 - parseInt(fromSquare[1]);
            
            if (fromRank >= 0 && fromRank < 8 && fromFile >= 0 && fromFile < 8) {
              const piece = board[fromRank][fromFile];
              if (piece) {
                const pieceLower = piece.toLowerCase();
                const isWhite = piece >= 'A' && piece <= 'Z';
                
                // 分析走法类型
                if (pieceLower === 'p') {
                  // 兵走法
                  const fromRankNum = parseInt(fromSquare[1]);
                  const toRankNum = parseInt(toSquare[1]);
                  const rankDiff = isWhite ? toRankNum - fromRankNum : fromRankNum - toRankNum;
                  
                  if (rankDiff > 0) {
                    this.statistics.move_pawn_push++;
                  } else {
                    this.statistics.move_pawn_capture++;
                  }
                  
                  // 检查吃过路兵
                  // 1. 必须是斜向移动（对角线）
                  // 2. 目标位置必须是FEN中指定的en passant square
                  const toFile = toSquare.charCodeAt(0) - 'a'.charCodeAt(0);
                  const isDiagonalMove = Math.abs(fromFile - toFile) === 1 && Math.abs(fromRankNum - toRankNum) === 1;
                  
                  if (isDiagonalMove) {
                    // 检查FEN中的en passant square
                    const fenParts = fen.split(' ');
                    const enPassantSquare = fenParts[3]; // 第4部分是en passant square
                    
                    if (enPassantSquare !== '-' && enPassantSquare === toSquare) {
                      // 验证目标位置是空的（被吃的兵已经被移除）
                      const toRank = 8 - toRankNum;
                      if (toRank >= 0 && toRank < 8 && toFile >= 0 && toFile < 8) {
                        const targetPiece = board[toRank][toFile];
                        if (!targetPiece || targetPiece === '') {
                    this.statistics.move_en_passant++;
                        }
                      }
                    }
                  }
                  
                  // 检查兵升变
                  if ((isWhite && toRankNum === 8) || (!isWhite && toRankNum === 1)) {
                    this.statistics.move_promotion++;
                  }
                } else {
                  // 子力走法
                  if (pieceLower === 'k') {
                    this.statistics.move_king_move++;
                    // 检查王车易位
                    if (Math.abs(fromFile - (toSquare.charCodeAt(0) - 'a'.charCodeAt(0))) === 2) {
                      this.statistics.move_castling++;
                    }
                  } else {
                    this.statistics.move_piece_development++;
                  }
                  
                  // 检查吃子
                  const toFile = toSquare.charCodeAt(0) - 'a'.charCodeAt(0);
                  const toRank = 8 - parseInt(toSquare[1]);
                  if (toRank >= 0 && toRank < 8 && toFile >= 0 && toFile < 8) {
                    const targetPiece = board[toRank][toFile];
                    if (targetPiece && targetPiece !== '') {
                      this.statistics.move_piece_capture++;
                    }
                  }
                }
                
                // 检查将军
                if (modelAnalysis.is_check) {
                  this.statistics.move_check++;
                }
              }
            }
          }
        }
      }
    }
    
    // 游戏阶段分析
    const fen = analysisData.fen;
    if (fen) {
      const fenParts = fen.split(' ');
      const fullmove = parseInt(fenParts[5]) || 1;
      
      if (fullmove <= 10) {
        this.statistics.phase_opening++;
        if (fullmove <= 5) {
          this.statistics.phase_early_opening++;
        } else {
          this.statistics.phase_late_opening++;
        }
      } else if (fullmove <= 30) {
        this.statistics.phase_middlegame++;
        if (fullmove <= 20) {
          this.statistics.phase_early_middlegame++;
        } else {
          this.statistics.phase_late_middlegame++;
        }
      } else {
        this.statistics.phase_endgame++;
        if (fullmove <= 40) {
          this.statistics.phase_early_endgame++;
        } else {
          this.statistics.phase_late_endgame++;
        }
      }
    }
    
    // 更新物质力量统计
    const material = analysisData.material;
    const activeColor = analysisData.fen?.split(' ')[1]; // 从FEN获取当前行棋方
    
    if (material && !material.error && activeColor) {
      const advantage = material.material_advantage;
      
      // 计算行棋方相对于对手的物质差值
      let materialDifference = 0;
      if (activeColor === 'w') {
        // 白方行棋：白方物质 - 黑方物质
        materialDifference = (material.white_material || 0) - (material.black_material || 0);
        // 累积数据（白方=行棋方，黑方=对手）
        this.materialAccumulator.totalWhiteMaterial += material.white_material || 0;
        this.materialAccumulator.totalBlackMaterial += material.black_material || 0;
      } else {
        // 黑方行棋：黑方物质 - 白方物质
        materialDifference = (material.black_material || 0) - (material.white_material || 0);
        // 累积数据（黑方=行棋方，白方=对手）
        this.materialAccumulator.totalWhiteMaterial += material.black_material || 0;
        this.materialAccumulator.totalBlackMaterial += material.white_material || 0;
      }
      
      // 根据物质差值分类
      if (materialDifference >= 5) this.statistics.material.majorAdvantage++;
      else if (materialDifference >= 3) this.statistics.material.moderateAdvantage++;
      else if (materialDifference >= 1) this.statistics.material.minorAdvantage++;
      else if (materialDifference >= -0.9) this.statistics.material.balanced++;
      else if (materialDifference >= -2.9) this.statistics.material.minorDisadvantage++;
      else if (materialDifference >= -4.9) this.statistics.material.moderateDisadvantage++;
      else this.statistics.material.majorDisadvantage++;
      
      this.materialAccumulator.totalMaterialDifference += Math.abs(material.material_difference || 0);
      this.materialAccumulator.count++;
      
      // 计算平均值（totalWhiteMaterial现在代表行棋方，totalBlackMaterial代表对手）
      this.statistics.material.averageActivePlayerMaterial = this.materialAccumulator.totalWhiteMaterial / this.materialAccumulator.count;
      this.statistics.material.averageOpponentMaterial = this.materialAccumulator.totalBlackMaterial / this.materialAccumulator.count;
      this.statistics.material.averageMaterialDifference = this.materialAccumulator.totalMaterialDifference / this.materialAccumulator.count;
    }
    
    // 更新WDL统计
    const wdl = analysisData.wdl;
    if (wdl && !wdl.error) {
      const winProb = wdl.win_probability || 0;
      const drawProb = wdl.draw_probability || 0;
      const lossProb = wdl.loss_probability || 0;
      
      // 使用净胜率差值 (胜率 - 败率) 来分类局面类型
      const netAdvantage = (winProb - lossProb) * 100; // 转换为百分比
      if (netAdvantage >= 80) this.statistics.wdl.decisive++;
      else if (netAdvantage >= 50) this.statistics.wdl.majorAdvantage++;
      else if (netAdvantage >= 25) this.statistics.wdl.moderateAdvantage++;
      else if (netAdvantage >= 10) this.statistics.wdl.minorAdvantage++;
      else if (netAdvantage >= -10) this.statistics.wdl.balanced++;
      else if (netAdvantage >= -25) this.statistics.wdl.minorDisadvantage++;
      else if (netAdvantage >= -50) this.statistics.wdl.moderateDisadvantage++;
      else if (netAdvantage >= -80) this.statistics.wdl.majorDisadvantage++;
      else this.statistics.wdl.hopeless++;
      
      // 将死局面
      if (wdl.evaluation_type === 'mate') this.statistics.wdl.matePositions++;
      
      // 累积WDL数据
      this.wdlAccumulator.totalWinProb += winProb;
      this.wdlAccumulator.totalDrawProb += drawProb;
      this.wdlAccumulator.totalLossProb += lossProb;
      this.wdlAccumulator.count++;
      
      // 计算平均值
      this.statistics.wdl.averageWinProb = this.wdlAccumulator.totalWinProb / this.wdlAccumulator.count;
      this.statistics.wdl.averageDrawProb = this.wdlAccumulator.totalDrawProb / this.wdlAccumulator.count;
      this.statistics.wdl.averageLossProb = this.wdlAccumulator.totalLossProb / this.wdlAccumulator.count;
    }
    
    this.notifyListeners();
  }

  // 统计激活位置的棋子类型
  updateActivationStatistics(fen: string, activations: number[]) {
    if (!fen || !activations || activations.length !== 64) {
      return;
    }

    // 解析FEN字符串获取棋盘状态和行棋方
    const fenParts = fen.split(' ');
    const boardFen = fenParts[0];
    const activeColor = fenParts[1] || 'w'; // 当前行棋方
    
    // 构建棋盘状态数组
    const board: string[][] = Array(8).fill(null).map(() => Array(8).fill(''));
    const rows = boardFen.split('/');
    
    if (rows.length === 8) {
      for (let i = 0; i < 8; i++) {
        let col = 0;
        for (const char of rows[i]) {
          if (/\d/.test(char)) {
            col += parseInt(char);
          } else {
            if (col < 8) {
              board[i][col] = char;
              col++;
            }
          }
        }
      }
      
      // 检查每个位置的激活值和对应的棋子
      for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
          // 激活值数组索引（从左下角开始，按行优先）
          const activationIndex = (7 - i) * 8 + j;
          const activation = activations[activationIndex];
          
          // 只统计激活值不为0的位置
          if (activation !== 0) {
            const piece = board[i][j];
            if (piece === '') {
              // 空格
              this.statistics.activated_piece_empty++;
            } else {
              // 有棋子，判断是己方还是对方，然后根据类型统计
              const isWhitePiece = piece >= 'A' && piece <= 'Z';
              const isOwnPiece = (activeColor === 'w' && isWhitePiece) || (activeColor === 'b' && !isWhitePiece);
              const pieceLower = piece.toLowerCase();
              
              if (isOwnPiece) {
                // 己方棋子
                switch (pieceLower) {
                  case 'p':
                    this.statistics.activated_own_pawn++;
                    break;
                  case 'n':
                    this.statistics.activated_own_knight++;
                    break;
                  case 'b':
                    this.statistics.activated_own_bishop++;
                    break;
                  case 'r':
                    this.statistics.activated_own_rook++;
                    break;
                  case 'q':
                    this.statistics.activated_own_queen++;
                    break;
                  case 'k':
                    this.statistics.activated_own_king++;
                    break;
                }
              } else {
                // 对方棋子
                switch (pieceLower) {
                  case 'p':
                    this.statistics.activated_opp_pawn++;
                    break;
                  case 'n':
                    this.statistics.activated_opp_knight++;
                    break;
                  case 'b':
                    this.statistics.activated_opp_bishop++;
                    break;
                  case 'r':
                    this.statistics.activated_opp_rook++;
                    break;
                  case 'q':
                    this.statistics.activated_opp_queen++;
                    break;
                  case 'k':
                    this.statistics.activated_opp_king++;
                    break;
                }
              }
            }
          }
        }
      }
    }
    
    this.notifyListeners();
  }

  getStatistics(): ComprehensiveStatistics {
    return { ...this.statistics };
  }

  subscribe(listener: (stats: ComprehensiveStatistics) => void) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener(this.getStatistics()));
  }
}

const globalStatsManager = new StatisticsManager();

// 规则统计卡片组件
const RuleStatisticsCard = ({ maxActivationTimes = 0 }: { maxActivationTimes?: number }) => {
  const [statistics, setStatistics] = useState<ComprehensiveStatistics>(() => globalStatsManager.getStatistics());

  useEffect(() => {
    const unsubscribe = globalStatsManager.subscribe(setStatistics);
    return unsubscribe;
  }, []);

  const rules = [
    // 己方棋子被抓
    { key: 'is_rook_under_attack', label: '车被抓', icon: '♜', color: '#e53e3e' },
    { key: 'is_knight_under_attack', label: '马被抓', icon: '♞', color: '#dd6b20' },
    { key: 'is_bishop_under_attack', label: '象被抓', icon: '♝', color: '#d69e2e' },
    { key: 'is_queen_under_attack', label: '后被抓', icon: '♛', color: '#9f7aea' },
    // 可以攻击对方棋子
    { key: 'is_can_capture_rook', label: '可吃车', icon: '♖', color: '#38a169' },
    { key: 'is_can_capture_knight', label: '可吃马', icon: '♘', color: '#38a169' },
    { key: 'is_can_capture_bishop', label: '可吃象', icon: '♗', color: '#38a169' },
    { key: 'is_can_capture_queen', label: '可吃后', icon: '♕', color: '#38a169' },
    // 王的状态
    { key: 'is_king_in_check', label: '将军', icon: '♚', color: '#e53e3e' },
    { key: 'is_checkmate', label: '将死', icon: '✗', color: '#c53030' },
    { key: 'is_stalemate', label: '逼和', icon: '⚫', color: '#718096' }
  ] as const;

  const modelMoves = [
    { key: 'model_move_pawn', label: '移动兵', icon: '♟', color: '#6b7280' },
    { key: 'model_move_knight', label: '移动马', icon: '♞', color: '#059669' },
    { key: 'model_move_bishop', label: '移动象', icon: '♝', color: '#0891b2' },
    { key: 'model_move_rook', label: '移动车', icon: '♜', color: '#dc2626' },
    { key: 'model_move_queen', label: '移动后', icon: '♛', color: '#9333ea' },
    { key: 'model_move_king', label: '移动王', icon: '♚', color: '#f59e0b' }
  ] as const;

  const moveTypes = [
    { key: 'move_pawn_push', label: '兵推进', icon: '⬆️', color: '#059669' },
    { key: 'move_pawn_capture', label: '兵吃子', icon: '↗️', color: '#dc2626' },
    { key: 'move_piece_development', label: '子力出动', icon: '🚀', color: '#0891b2' },
    { key: 'move_piece_capture', label: '子力吃子', icon: '⚔️', color: '#dc2626' },
    { key: 'move_castling', label: '王车易位', icon: '🏰', color: '#9333ea' },
    { key: 'move_king_move', label: '王移动', icon: '👑', color: '#f59e0b' },
    { key: 'move_check', label: '将军', icon: '⚡', color: '#e53e3e' },
    { key: 'move_en_passant', label: '吃过路兵', icon: '🔄', color: '#059669' },
    { key: 'move_promotion', label: '兵升变', icon: '⭐', color: '#9333ea' }
  ] as const;

  const gamePhases = [
    { key: 'phase_early_opening', label: '早期开局', icon: '🌱', color: '#059669', desc: '1-5步' },
    { key: 'phase_late_opening', label: '晚期开局', icon: '🌿', color: '#16a34a', desc: '6-10步' },
    { key: 'phase_early_middlegame', label: '早期中局', icon: '🌳', color: '#0891b2', desc: '11-20步' },
    { key: 'phase_late_middlegame', label: '晚期中局', icon: '🌲', color: '#0d9488', desc: '21-30步' },
    { key: 'phase_early_endgame', label: '早期残局', icon: '🍂', color: '#f59e0b', desc: '31-40步' },
    { key: 'phase_late_endgame', label: '晚期残局', icon: '🍁', color: '#dc2626', desc: '40+步' }
  ] as const;

  const getPercentage = (count: number) => {
    if (statistics.analyzedBoards === 0) return 0;
    return Math.round((count / statistics.analyzedBoards) * 100);
  };

    return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      {/* 标题和进度 */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="text-base font-bold text-gray-800">🧩 规则统计</div>
          <div className="text-xs text-gray-500">
            ({statistics.analyzedBoards}/{statistics.totalBoards} 已分析，限制:{maxActivationTimes})
          </div>
        </div>
        {statistics.totalBoards > 0 && maxActivationTimes > 0 && (
          <div className="flex items-center gap-2">
            <div className="w-20 bg-gray-200 rounded-full h-1">
              <div 
                className="h-1 bg-green-500 rounded-full transition-all duration-300"
                style={{ 
                  width: `${(statistics.analyzedBoards / statistics.totalBoards) * 100}%`
                }}
              />
            </div>
            <span className="text-xs text-green-600 font-medium">
              {Math.round((statistics.analyzedBoards / statistics.totalBoards) * 100)}%
            </span>
          </div>
        )}
      </div>
      
      {statistics.totalBoards === 0 ? (
        <div className="text-gray-500 text-sm text-center py-2">等待棋盘数据...</div>
             ) : statistics.analyzedBoards === 0 ? (
         <div className="flex items-center justify-center gap-2 text-yellow-600 text-sm py-2">
           <div className="animate-spin">🔄</div>
           <span>分析中... ({statistics.totalBoards}个棋盘，限制:{maxActivationTimes})</span>
         </div>
            ) : (
        <div className="space-y-4">
          {/* 棋类规则统计 */}
          <div>
            <h4 className="text-sm font-bold text-gray-700 mb-2">🎯 棋类规则</h4>
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 xl:grid-cols-11 gap-2">
              {rules.map(rule => {
                const count = statistics[rule.key as keyof RuleStatistics] as number;
                const percentage = getPercentage(count);
                
                return (
                  <div key={rule.key} className="flex flex-col items-center p-2 bg-gray-50 rounded-md min-h-20">
                    <div className="flex items-center justify-center gap-1 mb-1 h-6">
                      <span style={{ color: rule.color, fontSize: '14px' }}>{rule.icon}</span>
                      <span className="text-xs font-medium text-gray-700 text-center leading-tight">
                        {rule.label}
                      </span>
                    </div>
                    <div className="flex flex-col items-center flex-1 justify-center">
                      <div className="text-lg font-bold" style={{ color: rule.color }}>
                        {percentage}%
                      </div>
                      <div className="text-xs text-gray-500">
                        {count}/{statistics.analyzedBoards}
                      </div>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
                      <div 
                        className="h-1 rounded-full transition-all duration-300"
                        style={{ 
                          width: `${percentage}%`,
                          backgroundColor: rule.color
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 模型推理移动棋子类型统计 */}
          <div>
            <h4 className="text-sm font-bold text-gray-700 mb-2">🤖 模型推理移动棋子类型</h4>
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
              {modelMoves.map(move => {
                const count = statistics[move.key as keyof RuleStatistics] as number;
                const percentage = getPercentage(count);
                
                return (
                  <div key={move.key} className="flex flex-col items-center p-2 bg-gray-50 rounded-md min-h-20">
                    <div className="flex items-center justify-center gap-1 mb-1 h-6">
                      <span style={{ color: move.color, fontSize: '14px' }}>{move.icon}</span>
                      <span className="text-xs font-medium text-gray-700 text-center leading-tight">
                        {move.label}
                      </span>
                    </div>
                    <div className="flex flex-col items-center flex-1 justify-center">
                      <div className="text-lg font-bold" style={{ color: move.color }}>
                        {percentage}%
                      </div>
                      <div className="text-xs text-gray-500">
                        {count}/{statistics.analyzedBoards}
                      </div>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
                      <div 
                        className="h-1 rounded-full transition-all duration-300"
                        style={{ 
                          width: `${percentage}%`,
                          backgroundColor: move.color
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 棋盘位置热力图 */}
          <div className="flex gap-4 justify-center">
          <div>
              <h4 className="text-sm font-bold text-gray-700 mb-2">🗺️ 起始位置</h4>
              <div className="bg-gray-100 p-2 rounded-lg">
                <div className="grid grid-cols-8 gap-0.5 max-w-[160px] mx-auto">
                {Array.from({ length: 64 }, (_, i) => {
                  const rank = Math.floor(i / 8);
                  const file = i % 8;
                  const square = `${String.fromCharCode(97 + file)}${8 - rank}`;
                  const positionKey = `position_${square}`;
                  const count = statistics[positionKey as keyof RuleStatistics] as number || 0;
                  const percentage = getPercentage(count);
                  const intensity = Math.min(percentage / 20, 1); // 最大20%为最热
                  
                  return (
                    <div
                      key={square}
                        className={`w-5 h-5 flex items-center justify-center text-[8px] font-bold border border-gray-300 ${
                        (rank + file) % 2 === 0 ? 'bg-white' : 'bg-gray-200'
                      }`}
                      style={{
                        backgroundColor: count > 0 ? 
                          `rgba(255, 0, 0, ${intensity * 0.7})` : 
                          ((rank + file) % 2 === 0 ? '#ffffff' : '#e5e7eb')
                      }}
                      title={`${square}: ${count}次 (${percentage}%)`}
                    >
                      {count > 0 ? count : ''}
                    </div>
                  );
                })}
              </div>
                <div className="mt-1 text-[10px] text-gray-600 text-center">
                  红色：起始位置
              </div>
            </div>
            </div>

            <div>
              <h4 className="text-sm font-bold text-gray-700 mb-2">🎯 目标位置</h4>
              <div className="bg-gray-100 p-2 rounded-lg">
                <div className="grid grid-cols-8 gap-0.5 max-w-[160px] mx-auto">
                  {Array.from({ length: 64 }, (_, i) => {
                    const rank = Math.floor(i / 8);
                    const file = i % 8;
                    const square = `${String.fromCharCode(97 + file)}${8 - rank}`;
                    const positionKey = `to_position_${square}`;
                    const count = statistics[positionKey as keyof RuleStatistics] as number || 0;
                    const percentage = getPercentage(count);
                    const intensity = Math.min(percentage / 20, 1); // 最大20%为最热
                    
                    return (
                      <div
                        key={square}
                        className={`w-5 h-5 flex items-center justify-center text-[8px] font-bold border border-gray-300 ${
                          (rank + file) % 2 === 0 ? 'bg-white' : 'bg-gray-200'
                        }`}
                        style={{
                          backgroundColor: count > 0 ? 
                            `rgba(0, 128, 255, ${intensity * 0.7})` : 
                            ((rank + file) % 2 === 0 ? '#ffffff' : '#e5e7eb')
                        }}
                        title={`${square}: ${count}次 (${percentage}%)`}
                      >
                        {count > 0 ? count : ''}
                      </div>
                    );
                  })}
                </div>
                <div className="mt-1 text-[10px] text-gray-600 text-center">
                  蓝色：目标位置
                </div>
              </div>
            </div>
          </div>
          <div className="text-xs text-gray-600 text-center mt-2">
            (相对于行棋方视角，黑方行棋时上下对称)
          </div>

          {/* 走法类型分类统计 */}
          <div>
            <h4 className="text-sm font-bold text-gray-700 mb-2">🎯 走法类型分类</h4>
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-9 gap-2">
              {moveTypes.map(move => {
                const count = statistics[move.key as keyof RuleStatistics] as number;
                const percentage = getPercentage(count);
                
                return (
                  <div key={move.key} className="flex flex-col items-center p-2 bg-gray-50 rounded-md min-h-20">
                    <div className="flex items-center justify-center gap-1 mb-1 h-6">
                      <span style={{ fontSize: '12px' }}>{move.icon}</span>
                      <span className="text-xs font-medium text-gray-700 text-center leading-tight">
                        {move.label}
                      </span>
                    </div>
                    <div className="flex flex-col items-center flex-1 justify-center">
                      <div className="text-lg font-bold" style={{ color: move.color }}>
                        {percentage}%
                      </div>
                      <div className="text-xs text-gray-500">
                        {count}/{statistics.analyzedBoards}
                      </div>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
                      <div 
                        className="h-1 rounded-full transition-all duration-300"
                        style={{ 
                          width: `${percentage}%`,
                          backgroundColor: move.color
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 游戏阶段分析 */}
          <div>
            <h4 className="text-sm font-bold text-gray-700 mb-2">⏰ 游戏阶段分析</h4>
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
              {gamePhases.map(phase => {
                const count = statistics[phase.key as keyof RuleStatistics] as number;
                const percentage = getPercentage(count);
                
                return (
                  <div key={phase.key} className="flex flex-col items-center p-2 bg-gray-50 rounded-md min-h-20">
                    <div className="flex items-center justify-center gap-1 mb-1 h-6">
                      <span style={{ fontSize: '12px' }}>{phase.icon}</span>
                      <span className="text-xs font-medium text-gray-700 text-center leading-tight">
                        {phase.label}
                      </span>
                    </div>
                    <div className="flex flex-col items-center flex-1 justify-center">
                      <div className="text-lg font-bold" style={{ color: phase.color }}>
                        {percentage}%
                      </div>
                      <div className="text-xs text-gray-500">
                        {count}/{statistics.analyzedBoards}
                      </div>
                      <div className="text-xs text-gray-400">{phase.desc}</div>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
                      <div 
                        className="h-1 rounded-full transition-all duration-300"
                        style={{ 
                          width: `${percentage}%`,
                          backgroundColor: phase.color
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 物质力量统计 */}
          <div>
            <h4 className="text-sm font-bold text-gray-700 mb-2">💰 物质力量分布</h4>
            <div className="grid grid-cols-4 lg:grid-cols-7 gap-1">
              {[
                { key: 'majorAdvantage', label: '大优', color: '#1B5E20', icon: '🟢', desc: '≥+5' },
                { key: 'moderateAdvantage', label: '中优', color: '#388E3C', icon: '🟢', desc: '+3~+4.9' },
                { key: 'minorAdvantage', label: '小优', color: '#66BB6A', icon: '🟢', desc: '+1~+2.9' },
                { key: 'balanced', label: '均势', color: '#FF9800', icon: '⚖️', desc: '±0~0.9' },
                { key: 'minorDisadvantage', label: '小劣', color: '#EF5350', icon: '🔴', desc: '-1~-2.9' },
                { key: 'moderateDisadvantage', label: '中劣', color: '#D32F2F', icon: '🔴', desc: '-3~-4.9' },
                { key: 'majorDisadvantage', label: '大劣', color: '#B71C1C', icon: '🔴', desc: '≤-5' }
              ].map(item => {
                const count = statistics.material[item.key as keyof MaterialStatistics] as number;
                const percentage = getPercentage(count);
                
                return (
                  <div key={item.key} className="flex flex-col items-center p-1 bg-gray-50 rounded-md">
                    <div className="flex items-center gap-1 mb-1">
                      <span style={{ fontSize: '12px' }}>{item.icon}</span>
                      <span className="text-xs font-medium text-gray-700">{item.label}</span>
                    </div>
                    <div className="text-sm font-bold" style={{ color: item.color }}>
                      {percentage}%
                    </div>
                    <div className="text-xs text-gray-500">{count}</div>
                    <div className="text-xs text-gray-400">{item.desc}</div>
                    <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
                      <div 
                        className="h-1 rounded-full transition-all duration-300"
                        style={{ 
                          width: `${percentage}%`,
                          backgroundColor: item.color
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mt-2 text-xs text-gray-600 text-center">
              平均物质: 行棋方{statistics.material.averageActivePlayerMaterial.toFixed(1)} vs 对手{statistics.material.averageOpponentMaterial.toFixed(1)} 
              (差值: {statistics.material.averageMaterialDifference.toFixed(1)})
            </div>
          </div>

                    {/* WDL胜负概率统计 */}
          <div>
            <h4 className="text-sm font-bold text-gray-700 mb-2">📊 胜负概率分布 (净胜率差值 = 胜率 - 败率)</h4>
            <div className="grid grid-cols-3 lg:grid-cols-5 xl:grid-cols-9 gap-1">
              {[
                { key: 'decisive', label: '决定性优势', color: '#1B5E20', icon: '💚', desc: '≥+80%' },
                { key: 'majorAdvantage', label: '大优', color: '#2E7D32', icon: '🟢', desc: '+50~79%' },
                { key: 'moderateAdvantage', label: '中优', color: '#388E3C', icon: '🟢', desc: '+25~49%' },
                { key: 'minorAdvantage', label: '小优', color: '#66BB6A', icon: '🟢', desc: '+10~24%' },
                { key: 'balanced', label: '均势', color: '#FF9800', icon: '🟡', desc: '±10%内' },
                { key: 'minorDisadvantage', label: '小劣', color: '#EF5350', icon: '🔴', desc: '-10~-24%' },
                { key: 'moderateDisadvantage', label: '中劣', color: '#D32F2F', icon: '🔴', desc: '-25~-49%' },
                { key: 'majorDisadvantage', label: '大劣', color: '#B71C1C', icon: '🔴', desc: '-50~-79%' },
                { key: 'hopeless', label: '决定性劣势', color: '#4A148C', icon: '💜', desc: '≤-80%' }
              ].map(item => {
                const count = statistics.wdl[item.key as keyof WDLStatistics] as number;
                const percentage = getPercentage(count);
                
                return (
                  <div key={item.key} className="flex flex-col items-center p-1 bg-gray-50 rounded-md">
                    <div className="flex items-center gap-1 mb-1">
                      <span style={{ fontSize: '12px' }}>{item.icon}</span>
                      <span className="text-xs font-medium text-gray-700">{item.label}</span>
                    </div>
                    <div className="text-sm font-bold" style={{ color: item.color }}>
                      {percentage}%
                    </div>
                    <div className="text-xs text-gray-500">{count}</div>
                    <div className="text-xs text-gray-400">{item.desc}</div>
                    <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
                      <div 
                        className="h-1 rounded-full transition-all duration-300"
                        style={{ 
                          width: `${percentage}%`,
                          backgroundColor: item.color
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mt-2 text-xs text-gray-600 text-center">
              平均WDL: 胜{(statistics.wdl.averageWinProb * 100).toFixed(1)}% | 
              和{(statistics.wdl.averageDrawProb * 100).toFixed(1)}% | 
              负{(statistics.wdl.averageLossProb * 100).toFixed(1)}%
              {statistics.wdl.matePositions > 0 && ` | 将死局面: ${statistics.wdl.matePositions}`}
            </div>
          </div>

          {/* 激活位置棋子类型统计 */}
          <div>
            <h4 className="text-sm font-bold text-gray-700 mb-2">🎯 激活位置棋子分布</h4>
            <div className="grid grid-cols-4 lg:grid-cols-7 xl:grid-cols-13 gap-1">
              {(() => {
                // 计算总激活位置数
                const totalActivated = statistics.activated_own_pawn + statistics.activated_own_knight + 
                                      statistics.activated_own_bishop + statistics.activated_own_rook + 
                                      statistics.activated_own_queen + statistics.activated_own_king +
                                      statistics.activated_opp_pawn + statistics.activated_opp_knight + 
                                      statistics.activated_opp_bishop + statistics.activated_opp_rook + 
                                      statistics.activated_opp_queen + statistics.activated_opp_king +
                                      statistics.activated_piece_empty;

                // 所有棋子类型的配置
                const allPieces = [
                  // 己方棋子（蓝色系）
                  { key: 'activated_own_pawn', label: '己方兵', icon: '♙', color: '#3B82F6', bgColor: 'bg-blue-50' },
                  { key: 'activated_own_knight', label: '己方马', icon: '♘', color: '#1E40AF', bgColor: 'bg-blue-50' },
                  { key: 'activated_own_bishop', label: '己方象', icon: '♗', color: '#1D4ED8', bgColor: 'bg-blue-50' },
                  { key: 'activated_own_rook', label: '己方车', icon: '♖', color: '#2563EB', bgColor: 'bg-blue-50' },
                  { key: 'activated_own_queen', label: '己方后', icon: '♕', color: '#3B82F6', bgColor: 'bg-blue-50' },
                  { key: 'activated_own_king', label: '己方王', icon: '♔', color: '#1E3A8A', bgColor: 'bg-blue-50' },
                  // 对方棋子（红色系）
                  { key: 'activated_opp_pawn', label: '对方兵', icon: '♟', color: '#DC2626', bgColor: 'bg-red-50' },
                  { key: 'activated_opp_knight', label: '对方马', icon: '♞', color: '#B91C1C', bgColor: 'bg-red-50' },
                  { key: 'activated_opp_bishop', label: '对方象', icon: '♝', color: '#991B1B', bgColor: 'bg-red-50' },
                  { key: 'activated_opp_rook', label: '对方车', icon: '♜', color: '#7F1D1D', bgColor: 'bg-red-50' },
                  { key: 'activated_opp_queen', label: '对方后', icon: '♛', color: '#6B1D1D', bgColor: 'bg-red-50' },
                  { key: 'activated_opp_king', label: '对方王', icon: '♚', color: '#450A0A', bgColor: 'bg-red-50' },
                  // 空格
                  { key: 'activated_piece_empty', label: '空格', icon: '⬜', color: '#6B7280', bgColor: 'bg-gray-50' }
                ];

                return allPieces.map(item => {
                  const count = statistics[item.key as keyof RuleStatistics] as number;
                  const percentage = totalActivated > 0 ? ((count / totalActivated) * 100).toFixed(1) : '0';
                  
                  return (
                    <div key={item.key} className={`flex flex-col items-center p-1 ${item.bgColor} rounded-md`}>
                      <div className="flex items-center gap-1 mb-1">
                        <span style={{ fontSize: '12px' }}>{item.icon}</span>
                        <span className="text-xs font-medium text-gray-700">{item.label}</span>
                      </div>
                      <div className="text-sm font-bold" style={{ color: item.color }}>
                        {percentage}%
                      </div>
                      <div className="text-xs text-gray-500">{count}</div>
                      <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
                        <div 
                          className="h-1 rounded-full transition-all duration-300"
                          style={{ 
                            width: `${percentage}%`,
                            backgroundColor: item.color
                          }}
                        />
                      </div>
                    </div>
                  );
                });
              })()}
            </div>
            <div className="mt-2 text-xs text-gray-600 text-center">
              总激活位置: {statistics.activated_own_pawn + statistics.activated_own_knight + 
                         statistics.activated_own_bishop + statistics.activated_own_rook + 
                         statistics.activated_own_queen + statistics.activated_own_king +
                         statistics.activated_opp_pawn + statistics.activated_opp_knight + 
                         statistics.activated_opp_bishop + statistics.activated_opp_rook + 
                         statistics.activated_opp_queen + statistics.activated_opp_king +
                         statistics.activated_piece_empty} (100%)
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// 支持渐进式加载的棋盘分析组件


// Self-play 棋盘组件
// 统一的分析状态管理
interface UnifiedAnalysisState {
  stockfishAnalysis: any;
  selfplayData: any;
  isStockfishLoading: boolean;
  isSelfplayLoading: boolean;
  stockfishCompleted: boolean;
  selfplayCompleted: boolean;
  currentStep: number;
  currentMode: 'analysis' | 'selfplay';
  taskId?: string;  // 用于任务管理
}

class UnifiedAnalysisManager {
  private states = new Map<string, UnifiedAnalysisState>();
  
  getState(key: string): UnifiedAnalysisState {
    if (!this.states.has(key)) {
      this.states.set(key, {
        stockfishAnalysis: null,
        selfplayData: null,
        isStockfishLoading: false,
        isSelfplayLoading: false,
        stockfishCompleted: false,
        selfplayCompleted: false,
        currentStep: 0,
        currentMode: 'analysis'
      });
    }
    return this.states.get(key)!;
  }
  
  setState(key: string, updates: Partial<UnifiedAnalysisState>) {
    const current = this.getState(key);
    this.states.set(key, { ...current, ...updates });
  }
  
  clear() {
    this.states.clear();
  }
}

const globalUnifiedAnalysisManager = new UnifiedAnalysisManager();

// 统一的棋盘分析组件
const UnifiedChessBoard = ({ fen, activations, sampleIndex, analysisName, contextId, delayMs = 0, autoAnalyze = true, includeInStats = false, globalAnalysisCollapsed = false }: {
  fen: string;
  activations?: number[];
  sampleIndex?: number;
  analysisName?: string;
  contextId?: number;
  delayMs?: number;
  autoAnalyze?: boolean;
  includeInStats?: boolean;
  globalAnalysisCollapsed?: boolean;
}) => {
  const analysisKey = `${fen}_${sampleIndex}_${contextId}`;
  
  // 从统一管理器获取状态
  const initialState = globalUnifiedAnalysisManager.getState(analysisKey);
  
  const [state, setState] = useState<UnifiedAnalysisState>(initialState);
  const [isAnalysisCollapsed, setIsAnalysisCollapsed] = useState<boolean>(globalAnalysisCollapsed);
  
  // 更新状态的辅助函数
  const updateState = (updates: Partial<UnifiedAnalysisState>) => {
    const newState = { ...state, ...updates };
    setState(newState);
    globalUnifiedAnalysisManager.setState(analysisKey, newState);
  };

  // 执行Stockfish分析
  const performStockfishAnalysis = async () => {
    if (state.stockfishCompleted && state.stockfishAnalysis) {
      console.log(`💰 使用已缓存的Stockfish分析结果: ${fen.substring(0, 20)}...`);
      return;
    }

    updateState({ isStockfishLoading: true });
    
    try {
      console.log(`📡 发送Stockfish请求: ${fen.substring(0, 20)}...`);
      
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/analyze/stockfish`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const result = await response.json();
      
      console.log(`✅ Stockfish分析完成: ${fen.substring(0, 20)}...`, {
        bestMove: result.best_move || result.bestMove,
        status: result.status
      });
      
      const normalizedResult = {
        ...result,
        bestMove: result.best_move || result.bestMove,
        ponder: result.ponder,
        status: result.status || 'success',
        error: result.error,
        fen: result.fen || fen,
        isCheck: result.is_check || result.isCheck,
        rules: result.rules,
        material: result.material,
        wdl: result.wdl
      };
      
      updateState({ 
        stockfishAnalysis: normalizedResult,
        isStockfishLoading: false,
        stockfishCompleted: true
      });
      
      // 上报统计数据
      if (includeInStats && normalizedResult.status === 'success') {
        globalStatsManager.updateAnalysis(normalizedResult);
        
        if (activations) {
          globalStatsManager.updateActivationStatistics(fen, activations);
        }
      }
      
    } catch (error: any) {
      console.error('Stockfish分析失败:', error);
      updateState({ isStockfishLoading: false });
    }
  };

  // 取消相关任务的辅助函数
  const cancelRelatedTasks = async (pattern: string) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/tasks/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pattern }),
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log(`⚡ 已取消相关任务: ${result.cancelled_count} 个`);
        return result.cancelled_count;
      }
    } catch (error) {
      console.warn('取消任务失败:', error);
    }
    return 0;
  };

  // 执行Self-play分析（带任务管理）
  const performSelfplayAnalysis = async (isHighPriority = false) => {
    if (state.selfplayCompleted && state.selfplayData && !isHighPriority) {
      console.log(`💰 使用已缓存的Self-play分析结果: ${fen.substring(0, 20)}...`);
      return;
    }

    // 如果是高优先级请求（如分支推演），先取消相关任务
    if (isHighPriority) {
      await cancelRelatedTasks(fen.substring(0, 20));
    }

    updateState({ isSelfplayLoading: true });
    
    try {
      console.log(`🎮 发送Self-play请求 (${isHighPriority ? '高优先级' : '普通'}): ${fen.substring(0, 20)}...`);
      
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/analyze/selfplay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          fen,
          max_moves: 10,
          temperature: 1.0,
          priority: isHighPriority ? 10 : 1
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const result = await response.json();
      
      console.log(`📥 Self-play API响应:`, result);
      console.log(`📥 Self-play数据:`, result.selfplay);
      
      // 检查是否有错误
      if (result.status === 'error' || result.selfplay?.error) {
        console.error(`❌ Self-play分析失败: ${result.selfplay?.error || '未知错误'}`);
        updateState({ isSelfplayLoading: false });
        return;
      }
      
      // 检查是否被取消
      if (result.selfplay?.cancelled) {
        console.log(`⚡ Self-play任务被取消: ${fen.substring(0, 20)}...`);
        updateState({ isSelfplayLoading: false });
        return;
      }
      
      console.log(`✅ Self-play分析完成: ${fen.substring(0, 20)}...`);
      
      updateState({ 
        selfplayData: result.selfplay,
        isSelfplayLoading: false,
        selfplayCompleted: true,
        currentStep: 0,
        taskId: result.selfplay?.task_id
      });
      
    } catch (error: any) {
      console.error('Self-play分析失败:', error);
      updateState({ isSelfplayLoading: false });
    }
  };

  // 切换模式（仅显示，不触发新请求）
  const switchMode = (mode: 'analysis' | 'selfplay') => {
    console.log(`🔄 切换显示模式: ${mode}`);
    updateState({ currentMode: mode });
    
    // 不再在切换时触发新的分析请求
    // 所有分析都在后台自动进行
  };

  // 处理分支推演（高优先级）
  const handleBranchSelfplay = async (selectedMove: string) => {
    console.log(`🎯 开始高优先级分支推演: ${selectedMove}`);
    
    // 取消当前的低优先级任务
    if (state.taskId) {
      await cancelRelatedTasks(fen.substring(0, 20));
    }
    
    // 执行高优先级分支推演
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/analyze/selfplay/branch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          initial_fen: fen,
          game_history: state.selfplayData?.game_history || [],
          branch_step: state.currentStep,
          selected_move: selectedMove,
          max_moves: 10,
          temperature: 1.0,
          priority: 10  // 高优先级
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const result = await response.json();
      
      if (result.status === 'success' && result.branch) {
        // 更新为分支结果
        updateState({ 
          selfplayData: result.branch,
          currentStep: state.currentStep,
          taskId: result.branch?.task_id
        });
        console.log(`✅ 分支推演完成: 从第${state.currentStep}步重新推演`);
      }
      
    } catch (error: any) {
      console.error('分支推演失败:', error);
    }
  };



  // 自动分析逻辑：同时启动推演和分析
  useEffect(() => {
    if (!autoAnalyze) return;
    
    const timer = setTimeout(() => {
      console.log(`🚀 启动统一分析 (延迟${delayMs}ms): ${fen.substring(0, 30)}...`);
      
      // 并行启动两种分析
      const promises = [];
      
      if (!state.stockfishCompleted) {
        promises.push(performStockfishAnalysis());
      }
      
      if (!state.selfplayCompleted) {
        promises.push(performSelfplayAnalysis(false)); // 后台推演，非高优先级
      }
      
      // 并行执行，不等待结果
      Promise.allSettled(promises).then(() => {
        console.log(`✅ 统一分析完成: ${fen.substring(0, 30)}...`);
      });
    }, delayMs);
    
    return () => clearTimeout(timer);
  }, [autoAnalyze, delayMs, fen, state.stockfishCompleted, state.selfplayCompleted]);

  // 渲染模式选择器（显示后台进度）
  const renderModeSelector = () => (
    <div className="flex space-x-2 mb-4">
      <button
        onClick={() => switchMode('analysis')}
        className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
          state.currentMode === 'analysis'
            ? 'bg-blue-500 text-white'
            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
        }`}
      >
        静态分析
        {state.isStockfishLoading && <span className="ml-2 animate-spin">⏳</span>}
        {!state.isStockfishLoading && state.stockfishCompleted && <span className="ml-2">✅</span>}
        {!state.isStockfishLoading && !state.stockfishCompleted && <span className="ml-2">⭕</span>}
      </button>
      <button
        onClick={() => switchMode('selfplay')}
        className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
          state.currentMode === 'selfplay'
            ? 'bg-green-500 text-white'
            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
        }`}
      >
        推演分析
        {state.isSelfplayLoading && <span className="ml-2 animate-spin">⏳</span>}
        {!state.isSelfplayLoading && state.selfplayCompleted && <span className="ml-2">✅</span>}
        {!state.isSelfplayLoading && !state.selfplayCompleted && <span className="ml-2">⭕</span>}
      </button>
    </div>
  );

  const renderContent = () => {
    if (state.currentMode === 'analysis') {
      // 渲染静态分析内容
      return (
        <div>
                      {!state.stockfishAnalysis && (
              <div className="flex items-center justify-center p-4">
                <div className="text-blue-600">
                  {state.isStockfishLoading ? (
                    <>🔍 正在后台分析中...（您可以切换到推演模式查看其他结果）</>
                  ) : (
                    <>⏰ 静态分析将在后台自动开始...</>
                  )}
                </div>
              </div>
            )}
          
          {state.stockfishAnalysis ? (
            <div className="mt-4">
              {generateFENChessBoard(fen, activations, sampleIndex, analysisName, contextId, state.stockfishAnalysis, isAnalysisCollapsed)}
            </div>
          ) : (
            <div className="mt-4">
              <div className="text-gray-500 text-center p-8">
                等待分析结果... 
                {state.isStockfishLoading && <span className="ml-2">分析进行中</span>}
              </div>
            </div>
          )}
        </div>
      );
    } else {
      // 渲染推演分析内容  
      return (
        <div>
          {!state.selfplayData && (
            <div className="flex items-center justify-center p-4">
              <div className="text-green-600">
                {state.isSelfplayLoading ? (
                  <>🎮 正在后台推演中...（您可以切换到分析模式查看其他结果）</>
                ) : (
                  <>⏰ 推演分析将在后台自动开始...</>
                )}
              </div>
            </div>
          )}
          
          {state.selfplayData ? (
            <div className="mt-4">
              <div className="flex justify-between items-center mb-4">
                <h4 className="text-md font-semibold">
                  推演结果 ({state.selfplayData.total_moves || 0} 步)
                  {state.isSelfplayLoading && <span className="ml-2 text-sm text-gray-500">（后台继续推演中）</span>}
                </h4>
                <div className="flex items-center space-x-4">
                  {state.isSelfplayLoading && state.taskId && (
                    <button
                      onClick={() => cancelRelatedTasks(fen.substring(0, 20))}
                      className="px-3 py-1 bg-red-500 text-white text-sm rounded hover:bg-red-600"
                    >
                      取消推演
                    </button>
                  )}
                  <input
                    type="range"
                    min="0"
                    max={(state.selfplayData.game_history?.length || 0)}
                    value={state.currentStep}
                    onChange={(e) => updateState({ currentStep: parseInt(e.target.value) })}
                    className="w-32"
                  />
                  <span className="text-sm text-gray-600">第 {state.currentStep} 步</span>
                </div>
              </div>
              
              {/* 显示候选移动（用于分支推演） */}
              {state.currentStep > 0 && state.selfplayData.game_history?.[state.currentStep - 1]?.top_moves && (
                <div className="mb-4">
                  <h5 className="text-sm font-medium mb-2">候选移动（点击分支推演）:</h5>
                  <div className="flex flex-wrap gap-2">
                    {state.selfplayData.game_history[state.currentStep - 1].top_moves.slice(0, 5).map((move: any, idx: number) => (
                      <button
                        key={idx}
                        onClick={() => handleBranchSelfplay(move.uci)}
                        className={`px-2 py-1 text-xs rounded border ${
                          move.uci === state.selfplayData.game_history[state.currentStep - 1].move
                            ? 'bg-blue-100 border-blue-500 text-blue-700'
                            : 'bg-gray-100 border-gray-300 text-gray-700 hover:bg-gray-200'
                        }`}
                        disabled={state.isSelfplayLoading}
                      >
                        {move.san} ({move.probability?.toFixed(1)}%)
                      </button>
                    ))}
                  </div>
                </div>
              )}
              
              {generateFENChessBoard(
                state.currentStep === 0 ? fen : state.selfplayData.game_history?.[state.currentStep - 1]?.fen_after || fen,
                activations,
                sampleIndex,
                analysisName,
                contextId,
                null,
                isAnalysisCollapsed,
                undefined,
                state.currentStep > 0 ? {
                  lastMove: state.selfplayData.game_history?.[state.currentStep - 1]?.move,
                  nextMove: state.selfplayData.game_history?.[state.currentStep]?.move
                } : undefined
              )}
            </div>
          ) : (
            <div className="mt-4">
              <div className="text-gray-500 text-center p-8">
                等待推演结果... 
                {state.isSelfplayLoading && <span className="ml-2">推演进行中</span>}
              </div>
            </div>
          )}
        </div>
      );
    }
  };

  return (
    <div className="border rounded-lg p-4 bg-white shadow-sm">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">棋局分析</h3>
        <button
          onClick={() => setIsAnalysisCollapsed(!isAnalysisCollapsed)}
          className="text-gray-500 hover:text-gray-700"
        >
          {isAnalysisCollapsed ? '📖' : '📕'}
        </button>
      </div>
      
      {!isAnalysisCollapsed && (
        <>
          {renderModeSelector()}
          {renderContent()}
        </>
      )}
    </div>
  );
};

const SelfPlayChessBoard = ({ fen, activations, sampleIndex, analysisName, contextId, delayMs = 0, autoAnalyze = true, includeInStats = false, globalAnalysisCollapsed = false }: {
  fen: string;
  activations?: number[];
  sampleIndex?: number;
  analysisName?: string;
  contextId?: number;
  delayMs?: number;
  autoAnalyze?: boolean;
  includeInStats?: boolean;
  globalAnalysisCollapsed?: boolean;
}) => {
  const [selfplayData, setSelfplayData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isBranching, setIsBranching] = useState<boolean>(false);

  // 启动self-play分析
  const startSelfplayAnalysis = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/analyze/selfplay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          fen,
          max_moves: 10,
          temperature: 1.0
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const result = await response.json();
      setSelfplayData(result.selfplay);
      setCurrentStep(0);
    } catch (error: any) {
      console.error('Self-play分析失败:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // 处理分支推演
  const handleBranchSelfplay = async (branchStep: number, selectedMove: string) => {
    if (!selfplayData) return;
    
    setIsBranching(true);
    try {
      console.log(`🎯 开始分支推演: 第${branchStep}步选择走法 ${selectedMove}`);
      
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/analyze/selfplay/branch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          initial_fen: fen,
          game_history: selfplayData.game_history,
          branch_step: branchStep,
          selected_move: selectedMove,
          max_moves: 10,
          temperature: 1.0
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const result = await response.json();
      
      if (result.status === 'success' && result.branch) {
        // 更新self-play数据为分支结果
        setSelfplayData(result.branch);
        // 保持当前步骤在分支点
        setCurrentStep(branchStep);
        console.log(`✅ 分支推演成功: 从第${branchStep}步重新推演`);
        console.log(`   - 选择走法: ${result.branch.selected_move}`);
        console.log(`   - 新增步数: ${result.branch.branch_info?.new_moves_count || 0}`);
        console.log(`   - 总步数: ${result.branch.total_moves}`);
        console.log(`   - 剩余可用步数: ${result.branch.branch_info?.remaining_moves_used || 0}`);
      } else {
        throw new Error(result.branch?.error || '分支推演失败');
      }
    } catch (error: any) {
      console.error('分支推演失败:', error);
      alert(`分支推演失败: ${error.message}`);
    } finally {
      setIsBranching(false);
    }
  };

  useEffect(() => {
    if (autoAnalyze && !selfplayData && !isLoading) {
      const timer = setTimeout(() => {
        startSelfplayAnalysis();
      }, delayMs);
      return () => clearTimeout(timer);
    }
  }, [fen, autoAnalyze, delayMs]);

  // 处理滑动轴变化
  const handleStepChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentStep(parseInt(event.target.value));
  };



  // 获取当前显示的FEN
  const getCurrentFen = () => {
    if (!selfplayData?.game_history || currentStep === 0) {
      return fen; // 初始局面
    }
    const step = selfplayData.game_history[currentStep - 1];
    return step?.fen_after || fen;
  };

  // 获取当前步骤信息
  const getCurrentStepInfo = () => {
    if (!selfplayData?.game_history || currentStep === 0) {
      return { player: 'initial', move: '初始局面', probability: 0 };
    }
    return selfplayData.game_history[currentStep - 1];
  };

  // 获取当前步骤的WDL信息
  const getCurrentWDL = () => {
    if (!selfplayData?.wdl_history) return null;
    
    if (currentStep === 0) {
      // 初始局面的WDL
      return selfplayData.wdl_history[0]?.wdl;
    } else if (currentStep <= selfplayData.wdl_history.length - 1) {
      // 移动后的WDL
      return selfplayData.wdl_history[currentStep]?.wdl;
    }
    
    return null;
  };

  // 格式化WDL显示
  const formatWDL = (wdl: any) => {
    if (!wdl) return "无数据";
    
    if (wdl.error) {
      return `错误: ${wdl.error}`;
    }
    
    return `白胜${wdl.white_win_prob}% | 和${wdl.draw_prob}% | 黑胜${wdl.black_win_prob}%`;
  };

  // 生成WDL变化的小图表
  const generateWDLChart = () => {
    if (!selfplayData?.wdl_history || selfplayData.wdl_history.length === 0) {
      return null;
    }

    const chartData = selfplayData.wdl_history.map((entry: any, index: number) => ({
      step: index,
      white: entry.wdl?.white_win_prob || 0,
      draw: entry.wdl?.draw_prob || 0,
      black: entry.wdl?.black_win_prob || 0
    }));

    const maxSteps = chartData.length;
    const chartWidth = Math.max(200, maxSteps * 20);
    const chartHeight = 80;

    return (
      <div className="mt-2">
        <div className="text-xs font-semibold mb-1">WDL变化趋势:</div>
        <div className="bg-gray-50 p-2 rounded" style={{ overflowX: 'auto' }}>
          <svg width={chartWidth} height={chartHeight} className="border border-gray-200">
            {/* 背景网格 */}
            {[0, 25, 50, 75, 100].map(y => (
              <line 
                key={y} 
                x1="0" 
                y1={chartHeight - (y / 100) * chartHeight} 
                x2={chartWidth} 
                y2={chartHeight - (y / 100) * chartHeight} 
                stroke="#e0e0e0" 
                strokeWidth="0.5"
              />
            ))}
            
            {/* WDL数据线 */}
            {chartData.length > 1 && (
              <>
                                 {/* 白方胜率线 (蓝色) */}
                 <polyline
                   fill="none"
                   stroke="#2563eb"
                   strokeWidth="2"
                   points={chartData.map((d: any, i: number) => 
                     `${(i / (maxSteps - 1)) * chartWidth},${chartHeight - (d.white / 100) * chartHeight}`
                   ).join(' ')}
                 />
                 
                 {/* 和棋概率线 (橙色) */}
                 <polyline
                   fill="none"
                   stroke="#ea580c"
                   strokeWidth="2"
                   points={chartData.map((d: any, i: number) => 
                     `${(i / (maxSteps - 1)) * chartWidth},${chartHeight - (d.draw / 100) * chartHeight}`
                   ).join(' ')}
                 />
                 
                 {/* 黑方胜率线 (红色) */}
                 <polyline
                   fill="none"
                   stroke="#dc2626"
                   strokeWidth="2"
                   points={chartData.map((d: any, i: number) => 
                     `${(i / (maxSteps - 1)) * chartWidth},${chartHeight - (d.black / 100) * chartHeight}`
                   ).join(' ')}
                 />
              </>
            )}
            
            {/* 当前步骤指示器 */}
            {currentStep < chartData.length && (
              <line
                x1={(currentStep / (maxSteps - 1)) * chartWidth}
                y1="0"
                x2={(currentStep / (maxSteps - 1)) * chartWidth}
                y2={chartHeight}
                stroke="#666"
                strokeWidth="2"
                strokeDasharray="3,3"
              />
            )}
          </svg>
          
          {/* 图例 */}
          <div className="flex justify-center gap-4 mt-1 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-blue-600"></div>
              <span>白胜</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-orange-600"></div>
              <span>和棋</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-red-600"></div>
              <span>黑胜</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const currentStepInfo = getCurrentStepInfo();
  const currentFen = getCurrentFen();
  
  // 获取初始FEN的行棋方作为固定视角
  const initialActiveColor = fen.split(' ')[1] || 'w';
  
  // 计算推演走法信息
  const getSelfplayMoves = () => {
    if (!selfplayData?.game_history) return undefined;
    
    let lastMove = undefined;
    let nextMove = undefined;
    
    // 上一步走法（当前步骤的走法）
    if (currentStep > 0 && currentStep <= selfplayData.game_history.length) {
      const step = selfplayData.game_history[currentStep - 1];
      lastMove = step?.move;
    }
    
    // 下一步走法（下一个步骤的走法）
    if (currentStep < selfplayData.game_history.length) {
      const nextStep = selfplayData.game_history[currentStep];
      nextMove = nextStep?.move;
    }
    
    return { lastMove, nextMove };
  };
  
  const selfplayMoves = getSelfplayMoves();
  const boardHTML = generateFENChessBoard(currentFen, activations, sampleIndex, analysisName, contextId, undefined, globalAnalysisCollapsed, initialActiveColor, selfplayMoves);

  return (
    <div className="bg-white p-4 rounded-lg border shadow-sm hover:shadow-md transition-shadow">
      <div className="flex justify-between items-center mb-2">
        <h4 className="text-sm font-semibold">🎮 Self-play 推演</h4>
        {!selfplayData && !isLoading && (
          <button 
            onClick={startSelfplayAnalysis}
            className="px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600"
          >
            开始推演
          </button>
        )}
      </div>

      <div dangerouslySetInnerHTML={{ __html: boardHTML }} />
      
      {/* 步骤控制 */}
      {selfplayData && (
        <div className="mt-4">
          <div className="flex items-center space-x-2 mb-2">
            <span className="text-xs text-gray-600">步骤:</span>
                         <input
               type="range"
               min="0"
               max={selfplayData.game_history?.length || 0}
               value={currentStep}
               onChange={handleStepChange}
               className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
               style={{
                 background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${(currentStep / (selfplayData.game_history?.length || 1)) * 100}%, #e5e7eb ${(currentStep / (selfplayData.game_history?.length || 1)) * 100}%, #e5e7eb 100%)`
               }}
             />
            <span className="text-xs text-gray-600 min-w-[60px]">
              {currentStep}/{selfplayData.game_history?.length || 0}
            </span>
          </div>
          
                               {/* 当前步骤信息 */}
          <div className="text-xs bg-gray-50 p-2 rounded">
            {currentStep === 0 ? (
              <div>
                <div><strong>初始局面</strong></div>
                <div className="mt-1">
                  <strong>WDL:</strong> {formatWDL(getCurrentWDL())}
                </div>
              </div>
            ) : (
              <div>
                <div><strong>第{currentStep}步</strong> - {currentStepInfo.player === 'white' ? '白棋' : '黑棋'}</div>
                <div>走法: {currentStepInfo.move_san} ({currentStepInfo.move})</div>
                <div>概率: {(currentStepInfo.probability * 100).toFixed(1)}%</div>
                <div className="mt-1">
                  <strong>WDL:</strong> {formatWDL(getCurrentWDL())}
                </div>
                
                {/* 显示WDL变化 */}
                {currentStepInfo.wdl_before && currentStepInfo.wdl_after && (
                  <div className="mt-2 p-2 bg-blue-50 rounded">
                    <div className="font-semibold">WDL变化:</div>
                    <div className="text-xs">
                      <div>移动前: {formatWDL(currentStepInfo.wdl_before)}</div>
                      <div>移动后: {formatWDL(currentStepInfo.wdl_after)}</div>
                      
                      {/* 计算并显示变化量 */}
                      {(() => {
                        const before = currentStepInfo.wdl_before;
                        const after = currentStepInfo.wdl_after;
                        if (before && after && !before.error && !after.error) {
                          const whiteDiff = after.white_win_prob - before.white_win_prob;
                          const drawDiff = after.draw_prob - before.draw_prob;
                          const blackDiff = after.black_win_prob - before.black_win_prob;
                          
                          return (
                            <div className="mt-1 font-medium">
                              变化: 白{whiteDiff >= 0 ? '+' : ''}{whiteDiff.toFixed(1)}% | 
                              和{drawDiff >= 0 ? '+' : ''}{drawDiff.toFixed(1)}% | 
                              黑{blackDiff >= 0 ? '+' : ''}{blackDiff.toFixed(1)}%
                            </div>
                          );
                        }
                        return null;
                      })()}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
          
                    {/* WDL变化趋势图 */}
          {generateWDLChart()}
          
          {/* 候选走法对比卡片 */}
          {currentStep > 0 && currentStepInfo.top_moves && (
            <div className="mt-2">
              <div className="text-xs bg-green-50 p-3 rounded border">
                <div className="font-semibold mb-2 text-green-800 flex items-center gap-2">
                  🎯 第{currentStep}步候选走法对比
                  {isBranching && (
                    <span className="text-blue-600 text-xs animate-pulse">
                      正在重新推演...
                    </span>
                  )}
                </div>
                
                {!isBranching && (
                  <div className="text-xs text-green-600 mb-2 italic">
                    💡 点击其他候选走法可重新推演
                  </div>
                )}
                
                <div className="space-y-1">
                  {currentStepInfo.top_moves.slice(0, 5).map((move: any, idx: number) => {
                    const isSelected = move[0] === currentStepInfo.move;
                    return (
                      <div 
                        key={idx} 
                        className={`flex justify-between items-center p-2 rounded text-xs transition-all duration-200 ${
                          isSelected 
                            ? 'bg-yellow-100 border border-yellow-300 font-bold' 
                            : isBranching 
                              ? 'bg-gray-100 border border-gray-200 cursor-not-allowed' 
                              : 'bg-white border border-gray-200 hover:bg-blue-50 hover:border-blue-300 cursor-pointer'
                        }`}
                        onClick={() => {
                          if (!isSelected && !isBranching && selfplayData) {
                            handleBranchSelfplay(currentStep, move[0]);
                          }
                        }}
                        title={
                          isSelected 
                            ? '当前选择的走法' 
                            : isBranching 
                              ? '正在重新推演中...' 
                              : `点击选择 ${move[0]} 并重新推演`
                        }
                      >
                        <div className="flex items-center gap-2">
                          <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs ${
                            isSelected 
                              ? 'bg-yellow-500 text-white' 
                              : 'bg-gray-300 text-gray-700'
                          }`}>
                            {idx + 1}
                          </span>
                          <span className="font-medium">
                            {move[0]}
                          </span>
                          {isSelected && (
                            <span className="text-yellow-600 font-bold">← 已选择</span>
                          )}
                          {!isSelected && !isBranching && (
                            <span className="text-blue-600 text-xs opacity-70 hover:opacity-100 transition-opacity">
                              点击重新推演
                            </span>
                          )}
                          {isBranching && (
                            <span className="text-gray-500 text-xs">
                              推演中...
                            </span>
                          )}
                        </div>
                        
                        <div className="flex items-center gap-3">
                          <span className="text-blue-600 font-medium">
                            {(move[1] * 100).toFixed(1)}%
                          </span>
                          <span className="text-gray-500 text-xs">
                            索引: {move[2]}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
                
                {/* 统计信息 */}
                <div className="mt-2 pt-2 border-t border-green-200 text-xs text-green-700">
                  <div className="flex justify-between">
                    <span>选择概率: {(currentStepInfo.probability * 100).toFixed(1)}%</span>
                    <span>选择排名: {(() => {
                      const selectedMove = currentStepInfo.move;
                      const rank = currentStepInfo.top_moves.findIndex((move: any) => move[0] === selectedMove) + 1;
                      return rank > 0 ? `第${rank}名` : '未在前5名';
                    })()}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
            
            {/* 箭头图例 */}
            <div className="text-xs bg-blue-50 p-2 rounded">
              <div className="font-semibold mb-1">箭头说明:</div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-1 bg-green-500"></div>
                <span>绿色: 上一步走法</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-1 bg-blue-500" style={{backgroundImage: 'linear-gradient(to right, rgb(59 130 246) 50%, transparent 50%)', backgroundSize: '4px 1px'}}></div>
                <span>蓝色虚线: 下一步走法</span>
              </div>
            </div>
          
          {/* 游戏结果 */}
          {selfplayData.result && (
            <div className="text-xs mt-2 p-2 bg-blue-50 rounded">
              <div><strong>推演结果:</strong> {selfplayData.result}</div>
              <div><strong>总步数:</strong> {selfplayData.total_moves}</div>
              {selfplayData.termination && (
                <div><strong>结束方式:</strong> {selfplayData.termination}</div>
              )}
            </div>
          )}
        </div>
      )}
      
      {/* 加载状态 */}
      {isLoading && (
        <div className="mt-4 text-center text-xs text-gray-500">
          🔄 正在进行Self-play推演...
        </div>
      )}

      
    </div>
  );
};

const SimpleChessBoard = ({ fen, activations, sampleIndex, analysisName, contextId, delayMs = 0, autoAnalyze = true, includeInStats = false, globalAnalysisCollapsed = false, showSelfplay = false }: {
  fen: string;
  activations?: number[];
  sampleIndex?: number;
  analysisName?: string;
  contextId?: number;
  delayMs?: number;  // 延迟分析的毫秒数
  autoAnalyze?: boolean;  // 是否自动开始分析
  includeInStats?: boolean;  // 是否包含在统计中
  globalAnalysisCollapsed?: boolean;  // 全局分析折叠状态
  showSelfplay?: boolean;  // 是否显示self-play功能
}) => {
  // 生成唯一的分析状态键
  const analysisKey = `${fen}_${sampleIndex}_${contextId}`;
  
  // 从全局状态管理器获取初始状态
  const initialState = globalAnalysisStateManager.getAnalysisState(analysisKey);
  
  const [stockfishAnalysis, setStockfishAnalysis] = useState<any>(initialState.stockfishAnalysis);
  const [isLoading, setIsLoading] = useState<boolean>(initialState.isLoading);
  const [analysisStarted, setAnalysisStarted] = useState<boolean>(initialState.analysisStarted);
  const [isAnalysisCollapsed, setIsAnalysisCollapsed] = useState<boolean>(false);  // 分析信息折叠状态
  const [analysisCompleted, setAnalysisCompleted] = useState<boolean>(initialState.analysisCompleted);

  // 不再在组件创建时自动增加计数，改为由外部统一管理
  // useEffect(() => {
  //   if (includeInStats) {
  //     globalStatsManager.incrementTotal();
  //   }
  // }, [includeInStats]);

  useEffect(() => {
    if (!autoAnalyze || analysisStarted) return;
    
    // 设置延迟分析
    const timer = setTimeout(() => {
      console.log(`🚀 启动分析 (延迟${delayMs}ms): ${fen.substring(0, 30)}...`);
      
      setAnalysisStarted(true);
      setStockfishAnalysis(null);
      setIsLoading(true);
      
      // 更新全局状态
      globalAnalysisStateManager.setAnalysisState(analysisKey, {
        stockfishAnalysis: null,
        isLoading: true,
        analysisStarted: true,
        analysisCompleted: false
      });
      
      const analyzePosition = async () => {
        try {
          console.log(`📡 发送Stockfish请求 for ${fen.substring(0, 20)}...`);
          
          const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/analyze/stockfish`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fen }),
          });

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
          }

          const result = await response.json();
          
          console.log(`✅ 收到结果 for ${fen.substring(0, 20)}...`, {
            bestMove: result.best_move || result.bestMove,
            status: result.status
          });
          
                   // 规范化结果
         const normalizedResult = {
           ...result,
           bestMove: result.best_move || result.bestMove,
           ponder: result.ponder,
           status: result.status || 'success',
           error: result.error,
           fen: result.fen || fen,
           isCheck: result.is_check || result.isCheck,
           rules: result.rules,      // 保留规则分析数据
           material: result.material, // 保留物质力量数据
           wdl: result.wdl          // 保留WDL数据
         };
          
                     setStockfishAnalysis(normalizedResult);
           setIsLoading(false);
          setAnalysisCompleted(true);  // 标记分析已完成
          
          // 更新全局状态
          globalAnalysisStateManager.setAnalysisState(analysisKey, {
            stockfishAnalysis: normalizedResult,
            isLoading: false,
            analysisStarted: true,
            analysisCompleted: true
          });
           
           // 上报综合分析数据（仅当包含在统计中时）
           if (includeInStats && normalizedResult.status === 'success') {
             globalStatsManager.updateAnalysis(normalizedResult);
             
             // 统计激活位置的棋子类型（如果有激活值数据）
             if (activations && activations.length === 64) {
               globalStatsManager.updateActivationStatistics(fen, activations);
             }
           }
          
        } catch (error: any) {
          console.error(`❌ 分析失败 for ${fen.substring(0, 20)}...`, error);
          setStockfishAnalysis({ status: 'error', error: error.message, fen });
          setIsLoading(false);
          setAnalysisCompleted(true);  // 即使失败也标记为已完成
          
          // 更新全局状态
          globalAnalysisStateManager.setAnalysisState(analysisKey, {
            stockfishAnalysis: { status: 'error', error: error.message, fen },
            isLoading: false,
            analysisStarted: true,
            analysisCompleted: true
          });
        }
      };

      analyzePosition();
    }, delayMs);

    return () => clearTimeout(timer);
  }, [fen, autoAnalyze, delayMs, analysisStarted]);

  // 生成棋盘HTML
  const boardHTML = generateFENChessBoard(fen, activations, sampleIndex, analysisName, contextId, stockfishAnalysis, globalAnalysisCollapsed);

  // 如果启用了self-play功能，使用SelfPlayChessBoard组件
  if (showSelfplay) {
  return (
      <SelfPlayChessBoard 
        fen={fen}
        activations={activations}
        sampleIndex={sampleIndex}
        analysisName={analysisName}
        contextId={contextId}
        delayMs={delayMs}
        autoAnalyze={autoAnalyze}
        includeInStats={includeInStats}
        globalAnalysisCollapsed={globalAnalysisCollapsed}
      />
    );
  }

  return (
    <div className="bg-white p-4 rounded-lg border shadow-sm hover:shadow-md transition-shadow">
      <div dangerouslySetInnerHTML={{ __html: boardHTML }} />
      
      {/* 分析状态卡片 */}
      <div style={{ 
        marginTop: '8px', 
        padding: '6px 8px', 
        backgroundColor: '#f8f9fa', 
        borderRadius: '4px', 
        border: '1px solid #e9ecef',
        fontSize: '11px'
      }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          marginBottom: '4px',
          cursor: 'pointer'
        }} onClick={() => setIsAnalysisCollapsed(!isAnalysisCollapsed)}>
          <span style={{ fontSize: '10px', color: '#666' }}>
            {isAnalysisCollapsed ? '📋 展开分析' : '📋 折叠分析'}
          </span>
          <span style={{ fontSize: '12px' }}>
            {isAnalysisCollapsed ? '▼' : '▲'}
          </span>
        </div>
        
        {/* 分析内容 */}
        {!isAnalysisCollapsed && !globalAnalysisCollapsed && (
          <>
        {!analysisStarted ? (
          <div style={{ color: '#6c757d' }}>
            <div>⏳ 等待分析...</div>
          </div>
        ) : isLoading ? (
          <div style={{ color: '#856404' }}>
            <div>🔄 正在分析中...</div>
          </div>
        ) : stockfishAnalysis?.status === 'success' ? (
          <div style={{ color: '#155724' }}>
            <div>✅ 分析完成</div>
            {stockfishAnalysis.bestMove && <div style={{ color: '#0c5460', fontSize: '12px' }}>最佳: {stockfishAnalysis.bestMove}</div>}
            {stockfishAnalysis.ponder && <div style={{ color: '#0c5460', fontSize: '10px' }}>预想: {stockfishAnalysis.ponder}</div>}
          </div>
        ) : stockfishAnalysis?.status === 'error' ? (
          <div style={{ color: '#721c24' }}>
            <div>❌ 分析失败</div>
            <div style={{ fontSize: '9px', fontWeight: 'normal' }}>{stockfishAnalysis.error}</div>
          </div>
        ) : (
          <div style={{ color: '#666' }}>
            <div>🔄 准备分析...</div>
          </div>
        )}
          </>
        )}
        
        {/* 折叠时显示简要状态 */}
        {(isAnalysisCollapsed || globalAnalysisCollapsed) && (
          <div style={{ 
            fontSize: '10px', 
            color: analysisCompleted ? (stockfishAnalysis?.status === 'success' ? '#155724' : '#721c24') : '#856404',
            fontWeight: 'bold'
          }}>
            {!analysisStarted ? '⏳ 等待' : 
             isLoading ? '🔄 分析中' : 
             analysisCompleted ? (stockfishAnalysis?.status === 'success' ? '✅ 已完成' : '❌ 失败') : 
             '🔄 准备中'}
          </div>
        )}
      </div>
    </div>
  );
};

// 生成基于FEN的棋盘HTML函数
const generateFENChessBoard = (fen: string, activations?: number[], sampleIndex?: number, analysisName?: string, contextId?: number, stockfishAnalysis?: any, globalAnalysisCollapsed?: boolean, fixedPerspective?: string, selfplayMoves?: { lastMove?: string, nextMove?: string }): string => {
  console.log("=== 开始解析FEN ===");
  console.log("FEN字符串:", fen);
  console.log("激活值数组长度:", activations?.length);
  console.log("样本序号:", sampleIndex);
  console.log("Stockfish分析:", stockfishAnalysis);
  
  // 解析最佳走法的起点和终点
  let fromSquare = null;
  let toSquare = null;
  let ponderFromSquare = null;
  let ponderToSquare = null;
  
  // 从stockfishAnalysis中提取最佳走法对象
  let bestMoveObj = null;
  if (stockfishAnalysis?.status === "success" && stockfishAnalysis?.bestMove) {
    bestMoveObj = stockfishAnalysis.bestMove;
    console.log("提取的最佳走法对象:", bestMoveObj);
  }
  
  // 解析最佳走法坐标
  if (bestMoveObj && bestMoveObj.length >= 4) {
    const fromSquareStr = bestMoveObj.substring(0, 2);
    const toSquareStr = bestMoveObj.substring(2, 4);
    console.log("最佳走法分解:", { bestMoveObj, fromSquareStr, toSquareStr });
    
    fromSquare = {
      file: fromSquareStr.charCodeAt(0) - 'a'.charCodeAt(0), // 0-7
      rank: 8 - parseInt(fromSquareStr[1]) // 0-7，0=第8行
    };
    toSquare = {
      file: toSquareStr.charCodeAt(0) - 'a'.charCodeAt(0), // 0-7
      rank: 8 - parseInt(toSquareStr[1]) // 0-7，0=第8行
    };
    console.log("最佳走法坐标:", { fromSquare, toSquare });
  }
  
  // 解析ponder走法坐标
  if (stockfishAnalysis?.ponder && stockfishAnalysis.ponder.length >= 4) {
    const ponderMove = stockfishAnalysis.ponder;
    const fromSquareStr = ponderMove.substring(0, 2);
    const toSquareStr = ponderMove.substring(2, 4);
    console.log("Ponder走法分解:", { ponderMove, fromSquareStr, toSquareStr });
    
    ponderFromSquare = {
      file: fromSquareStr.charCodeAt(0) - 'a'.charCodeAt(0), // 0-7
      rank: 8 - parseInt(fromSquareStr[1]) // 0-7，0=第8行
    };
    ponderToSquare = {
      file: toSquareStr.charCodeAt(0) - 'a'.charCodeAt(0), // 0-7
      rank: 8 - parseInt(toSquareStr[1]) // 0-7，0=第8行
    };
    console.log("Ponder走法坐标:", { ponderFromSquare, ponderToSquare });
  }

  // 解析推演走法坐标
  let lastMoveFromSquare = null;
  let lastMoveToSquare = null;
  let nextMoveFromSquare = null;
  let nextMoveToSquare = null;

  // 解析上一步走法
  if (selfplayMoves?.lastMove && selfplayMoves.lastMove.length >= 4) {
    const lastMove = selfplayMoves.lastMove;
    const fromSquareStr = lastMove.substring(0, 2);
    const toSquareStr = lastMove.substring(2, 4);
    console.log("上一步走法分解:", { lastMove, fromSquareStr, toSquareStr });
    
    lastMoveFromSquare = {
      file: fromSquareStr.charCodeAt(0) - 'a'.charCodeAt(0), // 0-7
      rank: 8 - parseInt(fromSquareStr[1]) // 0-7，0=第8行
    };
    lastMoveToSquare = {
      file: toSquareStr.charCodeAt(0) - 'a'.charCodeAt(0), // 0-7
      rank: 8 - parseInt(toSquareStr[1]) // 0-7，0=第8行
    };
    console.log("上一步走法坐标:", { lastMoveFromSquare, lastMoveToSquare });
  }

  // 解析下一步走法
  if (selfplayMoves?.nextMove && selfplayMoves.nextMove.length >= 4) {
    const nextMove = selfplayMoves.nextMove;
    const fromSquareStr = nextMove.substring(0, 2);
    const toSquareStr = nextMove.substring(2, 4);
    console.log("下一步走法分解:", { nextMove, fromSquareStr, toSquareStr });
    
    nextMoveFromSquare = {
      file: fromSquareStr.charCodeAt(0) - 'a'.charCodeAt(0), // 0-7
      rank: 8 - parseInt(fromSquareStr[1]) // 0-7，0=第8行
    };
    nextMoveToSquare = {
      file: toSquareStr.charCodeAt(0) - 'a'.charCodeAt(0), // 0-7
      rank: 8 - parseInt(toSquareStr[1]) // 0-7，0=第8行
    };
    console.log("下一步走法坐标:", { nextMoveFromSquare, nextMoveToSquare });
  }
  
  // 解析FEN格式
  const fenParts = fen.split(' ');
  console.log("FEN分割后的部分:", fenParts);
  
  if (fenParts.length < 6) {
    console.error("FEN格式错误: 部分数量不足", fenParts.length);
    return `<div style="color: red;">FEN格式错误: 应包含6个部分，实际包含${fenParts.length}个部分<br>FEN: ${fen}</div>`;
  }
  
  const [boardFen, currentActiveColor, castling, enPassant, halfmove, fullmove] = fenParts;
  // 使用固定视角参数，如果没有提供则使用当前FEN的行棋方
  const activeColor = fixedPerspective || currentActiveColor;
  console.log("解析的FEN部分:", { boardFen, currentActiveColor, activeColor: activeColor, castling, enPassant, halfmove, fullmove });
  
  // 将FEN棋盘转换为8x8数组
  const board: string[][] = Array(8).fill(null).map(() => Array(8).fill(''));
  const rows = boardFen.split('/');
  console.log("棋盘行分割结果:", rows);
  
  if (rows.length !== 8) {
    console.error("FEN棋盘格式错误: 行数不是8", rows.length);
    return `<div style="color: red;">FEN棋盘格式错误: 应包含8行，实际包含${rows.length}行<br>棋盘部分: ${boardFen}</div>`;
  }
    
    for (let i = 0; i < 8; i++) {
    let col = 0;
    console.log(`解析第${i}行: ${rows[i]}`);
    for (const char of rows[i]) {
      if (/\d/.test(char)) {
        // 数字表示空格数量
        const emptySquares = parseInt(char);
        console.log(`在第${i}行第${col}列添加${emptySquares}个空格`);
        col += emptySquares;
          } else {
        // 棋子
        if (col < 8) {
          board[i][col] = char;
          console.log(`在第${i}行第${col}列放置棋子: ${char}`);
          col++;
        } else {
          console.error(`第${i}行列数超出范围: ${col}`);
        }
      }
    }
    console.log(`第${i}行解析完成，最终列数: ${col}`);
  }
  
  console.log("最终棋盘数组:", board);
  
  // 处理激活值 - 假设激活值对应64个棋盘位置
  let normalizedActivations: number[] = [];
  if (activations && activations.length >= 64) {
    // 取前64个激活值对应棋盘位置
    const boardActivations = activations.slice(0, 64);
    console.log("棋盘激活值:", boardActivations);
    
    // 归一化激活值到0-1范围
    const minActivation = Math.min(...boardActivations);
    const maxActivation = Math.max(...boardActivations);
    const range = maxActivation - minActivation;
    
    if (range > 0) {
      normalizedActivations = boardActivations.map(val => (val - minActivation) / range);
    } else {
      normalizedActivations = boardActivations.map(() => 0);
    }
    
    console.log("激活值范围:", { minActivation, maxActivation, range });
    console.log("归一化激活值:", normalizedActivations.slice(0, 10));
  } else {
    console.log("没有足够的激活值数据");
  }
  
  // 棋子符号映射
  const pieceSymbols: Record<string, string> = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
  };
  
  // 获取激活值对应的颜色
  const getActivationColor = (activation: number, isLight: boolean): string => {
    if (activation === 0) {
      return isLight ? '#f0d9b5' : '#b58863'; // 默认棋盘颜色
    }
    
    // 根据激活值强度调整颜色
    const intensity = Math.min(activation, 1);
    
    if (isLight) {
      // 浅色格子：从默认颜色渐变到红色
      const r = Math.round(240 + (255 - 240) * intensity);
      const g = Math.round(217 - 217 * intensity);
      const b = Math.round(181 - 181 * intensity);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // 深色格子：从默认颜色渐变到深红色
      const r = Math.round(181 + (139 - 181) * intensity);
      const g = Math.round(136 - 136 * intensity);
      const b = Math.round(99 - 99 * intensity);
      return `rgb(${r}, ${g}, ${b})`;
    }
  };
  
  // 生成HTML
  let html = `
    <div style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #fafafa; max-width: 320px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <h4 style="color: #333; margin: 0; font-size: 14px;">棋盘 ${activations ? '(激活值)' : ''}</h4>
        <div style="display: flex; gap: 4px; align-items: center;">
          ${sampleIndex !== undefined ? `<div style="background: #007bff; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold;">#${sampleIndex}</div>` : ''}
          ${contextId !== undefined ? `<div style="background: #28a745; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold;">ID:${contextId}</div>` : ''}
        </div>
      </div>
      ${analysisName ? `<div style="color: #666; font-size: 10px; margin-bottom: 8px;">Analysis: ${analysisName}</div>` : ''}
      <div style="display: inline-block; border: 2px solid #333; margin-bottom: 10px;">
        <table style="border-collapse: collapse; font-size: 20px;">
  `;
  
  // 生成棋盘行列标记
  html += '<tr><td style="width: 20px; height: 20px; text-align: center; font-size: 10px; color: #666;"></td>';
  for (let col = 0; col < 8; col++) {
    // 列标记保持不变，始终是A-H
    html += `<td style="width: 35px; height: 20px; text-align: center; font-size: 10px; color: #666;">${String.fromCharCode(65 + col)}</td>`;
  }
  html += '</tr>';
  
  for (let i = 0; i < 8; i++) {
    html += '<tr>';
    // 行号标记 - 如果是黑方行棋，翻转行号
    const rowNumber = activeColor === 'b' ? (i + 1) : (8 - i);
    html += `<td style="width: 20px; height: 35px; text-align: center; font-size: 10px; color: #666;">${rowNumber}</td>`;
    
    for (let j = 0; j < 8; j++) {
      // 激活值数组的索引：始终翻转，让激活值显示在行棋方这一侧
      const activationIndex = (7 - i) * 8 + j;  // 始终翻转索引
      const activation = normalizedActivations[activationIndex] || 0;
      
      // 棋盘位置：黑方行棋时只进行上下翻转，保持A列在最左边
      let boardRow, boardCol;
      if (activeColor === 'b') {
        // 黑方行棋：只翻转行，不翻转列
        boardRow = 7 - i;  // 行翻转
        boardCol = j;      // 列不翻转，保持A列在最左边
      } else {
        // 白方行棋：正常对应
        boardRow = i;
        boardCol = j;
      }
      
      const piece = board[boardRow][boardCol];
      const isLight = (i + j) % 2 === 0;
      const pieceSymbol = pieceSymbols[piece] || '';
      
      // 根据激活值设置背景颜色
      let bgColor = getActivationColor(activation, isLight);
      


      // 解析模型推理的最佳走法坐标（用于背景高亮）
      let modelFromSquareForBg = null;
      let modelToSquareForBg = null;
      if (stockfishAnalysis?.model && stockfishAnalysis.model.best_move && !stockfishAnalysis.model.error && stockfishAnalysis.model.best_move.length >= 4) {
        const modelBestMove = stockfishAnalysis.model.best_move;
        const fromSquareStr = modelBestMove.substring(0, 2);
        const toSquareStr = modelBestMove.substring(2, 4);
        
        modelFromSquareForBg = {
          file: fromSquareStr.charCodeAt(0) - 'a'.charCodeAt(0), // 0-7
          rank: 8 - parseInt(fromSquareStr[1]) // 0-7，0=第8行
        };
        modelToSquareForBg = {
          file: toSquareStr.charCodeAt(0) - 'a'.charCodeAt(0), // 0-7
          rank: 8 - parseInt(toSquareStr[1]) // 0-7，0=第8行
        };
      }


      
      // 棋子颜色
      const textColor = piece && piece >= 'A' && piece <= 'Z' ? '#fff' : '#000';
      const textShadow = piece && piece >= 'A' && piece <= 'Z' ? '1px 1px 2px rgba(0,0,0,0.8)' : '1px 1px 1px rgba(255,255,255,0.5)';
      
      // 激活值显示
      const rawActivation = activations && activationIndex < activations.length ? activations[activationIndex] : 0;
      const activationText = activations && activationIndex < activations.length ? 
        `<div style="position: absolute; top: 1px; right: 1px; font-size: 6px; color: #333; background: rgba(255,255,255,0.7); padding: 0px 1px; border-radius: 1px;">${rawActivation.toFixed(2)}</div>` : '';
      
      // 仅在起点格子绘制箭头
      let moveIndicator = '';
      
      // 最佳走法箭头（黄色）
      if (fromSquare && toSquare && fromSquare.file === boardCol && fromSquare.rank === boardRow) {
        let dx = (toSquare.file - fromSquare.file);
        let dy = (toSquare.rank - fromSquare.rank);
        
        // 如果是黑方行棋，棋盘已经翻转显示，所以箭头的方向需要调整
        if (activeColor === 'b') {
          // 黑方行棋时，棋盘只进行了上下翻转，所以箭头方向需要相应调整
          dy = -dy;  // 上下翻转
        }
        
        const angle = Math.atan2(dy, dx) * 180 / Math.PI; // 正常y轴（屏幕坐标）
        const length = Math.sqrt(dx * dx + dy * dy) * 35;
        moveIndicator = `
          <div style="position: absolute; top: 50%; left: 50%; width: ${length}px; height: 4px;
                      background: rgba(255,255,0,0.7); transform-origin: left center;
                      transform: translate(0,-50%) rotate(${angle}deg); z-index: 10;">
            <div style="position: absolute; right: -8px; top: -6px; width: 0; height: 0;
                        border-left: 12px solid rgba(255,255,0,0.9);
                        border-top: 6px solid transparent; border-bottom: 6px solid transparent;"></div>
          </div>`;
      }
      
      // Ponder走法双箭头（橙色）- 表示预想下一步
      if (ponderFromSquare && ponderToSquare && ponderFromSquare.file === boardCol && ponderFromSquare.rank === boardRow) {
        let dx = (ponderToSquare.file - ponderFromSquare.file);
        let dy = (ponderToSquare.rank - ponderFromSquare.rank);
        
        // 如果是黑方行棋，棋盘已经翻转显示，所以箭头的方向需要调整
        if (activeColor === 'b') {
          // 黑方行棋时，棋盘只进行了上下翻转，所以箭头方向需要相应调整
          dy = -dy;  // 上下翻转
        }
        
        const angle = Math.atan2(dy, dx) * 180 / Math.PI; // 正常y轴（屏幕坐标）
        const length = Math.sqrt(dx * dx + dy * dy) * 35;
        moveIndicator += `
          <div style="position: absolute; top: 40%; left: 50%; width: ${length}px; height: 2px;
                      background: rgba(255,140,0,0.6); transform-origin: left center;
                      transform: translate(0,-50%) rotate(${angle}deg); z-index: 9;">
            <div style="position: absolute; right: -4px; top: -3px; width: 0; height: 0;
                        border-left: 8px solid rgba(255,140,0,0.8);
                        border-top: 3px solid transparent; border-bottom: 3px solid transparent;"></div>
          </div>
          <div style="position: absolute; top: 50%; left: 50%; width: ${length}px; height: 2px;
                      background: rgba(255,140,0,0.6); transform-origin: left center;
                      transform: translate(0,-50%) rotate(${angle}deg); z-index: 9;">
            <div style="position: absolute; right: -4px; top: -3px; width: 0; height: 0;
                        border-left: 8px solid rgba(255,140,0,0.8);
                        border-top: 3px solid transparent; border-bottom: 3px solid transparent;"></div>
          </div>`;
      }

      // 解析模型推理的最佳走法坐标
      let modelFromSquare = null;
      let modelToSquare = null;
      if (stockfishAnalysis?.model && stockfishAnalysis.model.best_move && !stockfishAnalysis.model.error && stockfishAnalysis.model.best_move.length >= 4) {
        const modelBestMove = stockfishAnalysis.model.best_move;
        const fromSquareStr = modelBestMove.substring(0, 2);
        const toSquareStr = modelBestMove.substring(2, 4);
        
        modelFromSquare = {
          file: fromSquareStr.charCodeAt(0) - 'a'.charCodeAt(0), // 0-7
          rank: 8 - parseInt(fromSquareStr[1]) // 0-7，0=第8行
        };
        modelToSquare = {
          file: toSquareStr.charCodeAt(0) - 'a'.charCodeAt(0), // 0-7
          rank: 8 - parseInt(toSquareStr[1]) // 0-7，0=第8行
        };
      }

      // 模型推理最佳走法箭头（黑色）
      if (modelFromSquare && modelToSquare && modelFromSquare.file === boardCol && modelFromSquare.rank === boardRow) {
        let dx = (modelToSquare.file - modelFromSquare.file);
        let dy = (modelToSquare.rank - modelFromSquare.rank);
        
        // 如果是黑方行棋，棋盘已经翻转显示，所以箭头的方向需要调整
        if (activeColor === 'b') {
          // 黑方行棋时，棋盘只进行了上下翻转，所以箭头方向需要相应调整
          dy = -dy;  // 上下翻转
        }
        
        const angle = Math.atan2(dy, dx) * 180 / Math.PI; // 正常y轴（屏幕坐标）
        const length = Math.sqrt(dx * dx + dy * dy) * 35;
        moveIndicator += `
          <div style="position: absolute; top: 55%; left: 50%; width: ${length}px; height: 2px;
                      background: rgba(0,0,0,0.5); transform-origin: left center;
                      transform: translate(0,-50%) rotate(${angle}deg); z-index: 11;">
            <div style="position: absolute; right: -4px; top: -3px; width: 0; height: 0;
                        border-left: 8px solid rgba(0,0,0,0.6);
                        border-top: 3px solid transparent; border-bottom: 3px solid transparent;"></div>
          </div>`;
      }

      // 推演上一步走法箭头（绿色）
      if (lastMoveFromSquare && lastMoveToSquare && lastMoveFromSquare.file === boardCol && lastMoveFromSquare.rank === boardRow) {
        let dx = (lastMoveToSquare.file - lastMoveFromSquare.file);
        let dy = (lastMoveToSquare.rank - lastMoveFromSquare.rank);
        
        // 如果是黑方行棋，棋盘已经翻转显示，所以箭头的方向需要调整
        if (activeColor === 'b') {
          // 黑方行棋时，棋盘只进行了上下翻转，所以箭头方向需要相应调整
          dy = -dy;  // 上下翻转
        }
        
        const angle = Math.atan2(dy, dx) * 180 / Math.PI; // 正常y轴（屏幕坐标）
        const length = Math.sqrt(dx * dx + dy * dy) * 35;
        moveIndicator += `
          <div style="position: absolute; top: 60%; left: 50%; width: ${length}px; height: 3px;
                      background: rgba(0,200,0,0.7); transform-origin: left center;
                      transform: translate(0,-50%) rotate(${angle}deg); z-index: 12;">
            <div style="position: absolute; right: -5px; top: -4px; width: 0; height: 0;
                        border-left: 10px solid rgba(0,200,0,0.8);
                        border-top: 4px solid transparent; border-bottom: 4px solid transparent;"></div>
          </div>`;
      }

      // 推演下一步走法箭头（蓝色，虚线）
      if (nextMoveFromSquare && nextMoveToSquare && nextMoveFromSquare.file === boardCol && nextMoveFromSquare.rank === boardRow) {
        let dx = (nextMoveToSquare.file - nextMoveFromSquare.file);
        let dy = (nextMoveToSquare.rank - nextMoveFromSquare.rank);
        
        // 如果是黑方行棋，棋盘已经翻转显示，所以箭头的方向需要调整
        if (activeColor === 'b') {
          // 黑方行棋时，棋盘只进行了上下翻转，所以箭头方向需要相应调整
          dy = -dy;  // 上下翻转
        }
        
        const angle = Math.atan2(dy, dx) * 180 / Math.PI; // 正常y轴（屏幕坐标）
        const length = Math.sqrt(dx * dx + dy * dy) * 35;
        moveIndicator += `
          <div style="position: absolute; top: 65%; left: 50%; width: ${length}px; height: 2px;
                      background: linear-gradient(to right, rgba(0,100,255,0.6) 50%, transparent 50%);
                      background-size: 8px 2px; transform-origin: left center;
                      transform: translate(0,-50%) rotate(${angle}deg); z-index: 13;">
            <div style="position: absolute; right: -4px; top: -3px; width: 0; height: 0;
                        border-left: 8px solid rgba(0,100,255,0.7);
                        border-top: 3px solid transparent; border-bottom: 3px solid transparent;"></div>
          </div>`;
      }
      
      html += `
        <td style="width: 35px; height: 35px; background-color: ${bgColor}; 
                  text-align: center; vertical-align: middle; border: 1px solid #999;
                  color: ${textColor}; text-shadow: ${textShadow}; font-weight: bold; position: relative; 
                  transition: background-color 0.3s;">
          ${pieceSymbol}
          ${activationText}
          ${moveIndicator}
        </td>
      `;
    }
    html += '</tr>';
  }
  
  html += `
        </table>
      </div>
      <div style="margin-top: 8px; font-size: 11px; color: #666; line-height: 1.4;">
  `;
  
  // 激活值说明
  if (activations && activations.length > 0) {
    const boardActivations = activations.slice(0, 64);
    const minVal = Math.min(...boardActivations);
    const maxVal = Math.max(...boardActivations);
    
    html += `
      <div style="margin-bottom: 6px; padding: 4px; background: #f8f9fa; border-radius: 3px; border-left: 3px solid #007bff;">
        <strong>激活范围:</strong> ${minVal.toFixed(3)} ~ ${maxVal.toFixed(3)}<br>
        <small style="font-size: 9px;">颜色深浅表示激活强度</small>
      </div>
    `;
  }
  
  // 游戏信息（始终显示）
  const playerText = activeColor === 'w' ? '白方' : '黑方';
  html += `<div><strong>当前:</strong> ${playerText} | <strong>回合:</strong> ${fullmove}</div>`;
      
  // 分析信息（受全局折叠状态控制）
  if (!globalAnalysisCollapsed) {
  // 显示最佳走法信息
  if (stockfishAnalysis?.status === "success" && stockfishAnalysis?.bestMove) {
    html += `
      <div style="margin-top: 4px; padding: 4px; background: #e8f4fd; border-radius: 3px; border-left: 3px solid #007bff;">
        <strong>最佳走法:</strong> ${stockfishAnalysis.bestMove}
        ${stockfishAnalysis.isCheck ? ' <span style="color: #ff4444;">(将军!)</span>' : ''}
      </div>
    `;
  }
  
  // 显示物质力量分析结果
  if (stockfishAnalysis?.material && Object.keys(stockfishAnalysis.material).length > 0) {
    const material = stockfishAnalysis.material;
    
    if (!material.error) {
      const whiteTotal = material.white_material || 0;
      const blackTotal = material.black_material || 0;
      const difference = material.material_difference || 0;
      const advantage = material.material_advantage || 'equal';
      
      // 构建详细的棋子统计
      const whitePieces = material.white_pieces || {};
      const blackPieces = material.black_pieces || {};
      
      const whiteDetails = [
        `兵${whitePieces.pawns || 0}`,
        `马${whitePieces.knights || 0}`,
        `象${whitePieces.bishops || 0}`,
        `车${whitePieces.rooks || 0}`,
        `后${whitePieces.queens || 0}`
      ].filter(item => !item.endsWith('0')).join(', ');
      
      const blackDetails = [
        `兵${blackPieces.pawns || 0}`,
        `马${blackPieces.knights || 0}`,
        `象${blackPieces.bishops || 0}`,
        `车${blackPieces.rooks || 0}`,
        `后${blackPieces.queens || 0}`
      ].filter(item => !item.endsWith('0')).join(', ');
      
      const advantageColor = advantage === 'white' ? '#4CAF50' : advantage === 'black' ? '#f44336' : '#666';
      const advantageText = advantage === 'white' ? '白方优势' : advantage === 'black' ? '黑方优势' : '物质平衡';
      
      html += `
        <div style="margin-top: 6px; padding: 4px; background: #fff8e1; border-radius: 3px; border-left: 3px solid #ff9800;">
          <div style="font-size: 10px; font-weight: bold; color: #e65100; margin-bottom: 2px;">💰 物质力量</div>
          <div style="font-size: 9px; color: ${advantageColor}; margin-bottom: 2px;">
            <strong>总计:</strong> 白方${whiteTotal} vs 黑方${blackTotal} (${advantageText} ${Math.abs(difference)})
          </div>
          ${whiteDetails ? `
            <div style="font-size: 8px; color: #666; margin-bottom: 1px;">
              <strong>白方:</strong> ${whiteDetails}
            </div>
          ` : ''}
          ${blackDetails ? `
            <div style="font-size: 8px; color: #666;">
              <strong>黑方:</strong> ${blackDetails}
            </div>
          ` : ''}
        </div>
      `;
    }
  }
  
  // 显示WDL评估结果
  if (stockfishAnalysis?.wdl && Object.keys(stockfishAnalysis.wdl).length > 0) {
    const wdl = stockfishAnalysis.wdl;
    
    if (!wdl.error) {
      const winProb = wdl.win_probability || 0;
      const drawProb = wdl.draw_probability || 0;
      const lossProb = wdl.loss_probability || 0;
      const activeColor = wdl.active_color || 'white';
      const activeColorChinese = activeColor === 'white' ? '白方' : '黑方';
      
      // 转换为百分比
      const winPercent = Math.round(winProb * 100);
      const drawPercent = Math.round(drawProb * 100);
      const lossPercent = Math.round(lossProb * 100);
      
      // 确定主导结果的颜色
      const dominantResult = winProb > drawProb && winProb > lossProb ? 'win' : 
                            drawProb > winProb && drawProb > lossProb ? 'draw' : 'loss';
      const resultColor = dominantResult === 'win' ? '#4CAF50' : 
                         dominantResult === 'draw' ? '#ff9800' : '#f44336';
      
      html += `
        <div style="margin-top: 6px; padding: 4px; background: #f3e5f5; border-radius: 3px; border-left: 3px solid #9c27b0;">
          <div style="font-size: 10px; font-weight: bold; color: #6a1b9a; margin-bottom: 2px;">📊 Stockfish WDL (${activeColorChinese}视角)</div>
          <div style="font-size: 9px; margin-bottom: 2px;">
            <span style="color: #4CAF50; font-weight: bold;">胜${winPercent}%</span> | 
            <span style="color: #ff9800; font-weight: bold;">和${drawPercent}%</span> | 
            <span style="color: #f44336; font-weight: bold;">负${lossPercent}%</span>
          </div>
          ${wdl.evaluation_type === 'mate' ? `
            <div style="font-size: 8px; color: ${resultColor}; font-weight: bold;">
              ${wdl.mate_in > 0 ? `${Math.abs(wdl.mate_in)}步将死` : `被${Math.abs(wdl.mate_in)}步将死`}
            </div>
          ` : wdl.evaluation_cp !== null ? `
            <div style="font-size: 8px; color: #666;">
              评估: ${wdl.evaluation_cp > 0 ? '+' : ''}${wdl.evaluation_cp}厘兵
            </div>
          ` : ''}
          ${wdl.principal_variation && wdl.principal_variation.length > 0 ? `
            <div style="font-size: 8px; color: #666; margin-top: 1px;">
              主要变着: ${wdl.principal_variation.slice(0, 3).join(' ')}${wdl.principal_variation.length > 3 ? '...' : ''}
            </div>
          ` : ''}
        </div>
      `;
    }
  }

  // 显示模型推理结果
  if (stockfishAnalysis?.model && Object.keys(stockfishAnalysis.model).length > 0) {
    const model = stockfishAnalysis.model;
    
    if (!model.error) {
      html += `
        <div style="margin-top: 6px; padding: 4px; background: #e8f5e8; border-radius: 3px; border-left: 3px solid #4CAF50;">
          <div style="font-size: 10px; font-weight: bold; color: #2e7d32; margin-bottom: 2px;">🤖 模型推理结果</div>
      `;
      
      // 显示最佳走法
      if (model.best_move) {
        html += `
          <div style="font-size: 9px; margin-bottom: 2px;">
            <strong>最佳走法:</strong> ${model.best_move_san || model.best_move} 
            ${model.best_move_probability ? `(${model.best_move_probability}%)` : ''}
          </div>
        `;
      }
      
      // 显示策略分析
      if (model.policy_analysis && model.policy_analysis.best_moves) {
        const topMoves = model.policy_analysis.best_moves.slice(0, 3);
        if (topMoves.length > 0) {
          html += `
            <div style="font-size: 8px; color: #666; margin-bottom: 2px;">
              <strong>候选走法:</strong> ${topMoves.map((move: any) => 
                `${move.san}(${move.probability}%)`
              ).join(', ')}
            </div>
          `;
        }
      }
      
      // 显示模型WDL
      if (model.wdl_analysis) {
        const modelWdl = model.wdl_analysis;
        html += `
          <div style="font-size: 9px; margin-bottom: 2px;">
            <strong>模型WDL:</strong> 
            <span style="color: #4CAF50;">胜${modelWdl.win_percent}%</span> | 
            <span style="color: #ff9800;">和${modelWdl.draw_percent}%</span> | 
            <span style="color: #f44336;">负${modelWdl.loss_percent}%</span>
          </div>
        `;
      }
      
      // 显示价值评估
      if (model.value_analysis) {
        const valueAnalysis = model.value_analysis;
        html += `
          <div style="font-size: 8px; color: #666; margin-bottom: 2px;">
            <strong>价值评估:</strong> ${valueAnalysis.raw_value} 
            ${valueAnalysis.normalized_value ? `(归一化: ${valueAnalysis.normalized_value})` : ''}
          </div>
        `;
      }
      
      // 显示原始输出形状信息
      if (model.raw_outputs) {
        const rawOutputs = model.raw_outputs;
        html += `
          <div style="font-size: 7px; color: #999; margin-top: 2px;">
            <strong>输出维度:</strong> 
            ${rawOutputs.policy_logits_shape ? `策略${rawOutputs.policy_logits_shape.join('×')} ` : ''}
            ${rawOutputs.wdl_probs_shape ? `WDL${rawOutputs.wdl_probs_shape.join('×')} ` : ''}
            ${rawOutputs.value_shape ? `价值${rawOutputs.value_shape.join('×')}` : ''}
          </div>
        `;
      }
      
      html += `</div>`;
    } else {
      // 显示错误信息
      html += `
        <div style="margin-top: 6px; padding: 4px; background: #ffebee; border-radius: 3px; border-left: 3px solid #f44336;">
          <div style="font-size: 10px; font-weight: bold; color: #c62828; margin-bottom: 2px;">🤖 模型推理</div>
          <div style="font-size: 8px; color: #d32f2f;">
            错误: ${model.error}
          </div>
        </div>
      `;
    }
  }

  // 显示规则分析结果
  if (stockfishAnalysis?.rules && Object.keys(stockfishAnalysis.rules).length > 0) {
    const rules = stockfishAnalysis.rules;
    
    // 收集为真的规则
    const trueRules = [];
    const falseRules = [];
    const specialRules = [];
    
    // 基本检查
    if (rules.is_rook_under_attack) trueRules.push("车被攻击");
    if (rules.is_knight_under_attack) trueRules.push("马被攻击");
    if (rules.is_bishop_under_attack) trueRules.push("象被攻击");
    if (rules.is_queen_under_attack) trueRules.push("后被攻击");
    
    // 王的状态
    if (rules.is_king_in_check) trueRules.push("将军");
    if (rules.is_checkmate) trueRules.push("将死");
    if (rules.is_stalemate) trueRules.push("逼和");
    
    // 棋子配置
    if (rules.is_bishop_pair) trueRules.push("双象");
    
    // 战术分析
    if (rules.has_pinned_pieces) {
      trueRules.push(`钉住棋子(${rules.pinned_pieces?.length || 0}个)`);
    }
    if (rules.is_in_fork) trueRules.push("面临叉攻");
    
    // 兵结构分析
    if (rules.has_isolated_pawns) {
      const whiteCount = rules.isolated_pawns_white?.length || 0;
      const blackCount = rules.isolated_pawns_black?.length || 0;
      if (whiteCount > 0 || blackCount > 0) {
        trueRules.push(`孤兵(白${whiteCount},黑${blackCount})`);
      }
    }
    if (rules.has_doubled_pawns) {
      const whiteCount = Object.keys(rules.doubled_pawns_white || {}).length;
      const blackCount = Object.keys(rules.doubled_pawns_black || {}).length;
      if (whiteCount > 0 || blackCount > 0) {
        trueRules.push(`重兵(白${whiteCount}列,黑${blackCount}列)`);
      }
    }
    if (rules.has_passed_pawns) {
      const whiteCount = rules.passed_pawns_white?.length || 0;
      const blackCount = rules.passed_pawns_black?.length || 0;
      if (whiteCount > 0 || blackCount > 0) {
        trueRules.push(`通路兵(白${whiteCount},黑${blackCount})`);
      }
    }
    
    // 中心控制
    if (rules.center_control) {
      const white = rules.center_control.white || 0;
      const black = rules.center_control.black || 0;
      if (white > black) {
        specialRules.push(`中心控制: 白方优势(${white}-${black})`);
      } else if (black > white) {
        specialRules.push(`中心控制: 黑方优势(${black}-${white})`);
      } else {
        specialRules.push(`中心控制: 平衡(${white}-${black})`);
      }
    }
    
    html += `
      <div style="margin-top: 6px; padding: 4px; background: #f0f8ff; border-radius: 3px; border-left: 3px solid #4CAF50;">
        <div style="font-size: 10px; font-weight: bold; color: #2e7d32; margin-bottom: 2px;">🧩 规则分析</div>
        ${trueRules.length > 0 ? `
          <div style="font-size: 9px; color: #d32f2f; margin-bottom: 2px;">
            <strong>检测到:</strong> ${trueRules.join(', ')}
          </div>
        ` : ''}
        ${specialRules.length > 0 ? `
          <div style="font-size: 9px; color: #1976d2; margin-bottom: 2px;">
            ${specialRules.join(', ')}
          </div>
        ` : ''}
        ${trueRules.length === 0 && specialRules.length === 0 ? `
          <div style="font-size: 9px; color: #666;">无特殊局面特征</div>
        ` : ''}
      </div>
    `;
    }
  }
  
  // 显示完整FEN（始终显示）
  html += `
    <div style="margin-top: 6px; font-size: 9px; font-family: monospace; 
               background: #e8f4fd; padding: 4px; border-radius: 3px; word-break: break-all; line-height: 1.3; border-left: 3px solid #007bff;">
      <strong>FEN:</strong> ${fen}
    </div>
    ${contextId !== undefined ? `<div style="margin-top: 4px; font-size: 10px; color: #28a745; font-weight: bold;">Context ID: ${contextId}</div>` : ''}
  `;
  
  // 分析信息（受全局折叠状态控制）
  if (!globalAnalysisCollapsed) {
    html += `
          <div style="margin-top: 4px; font-size: 10px; color: #333;">
        <strong>最佳走法:</strong> ${stockfishAnalysis?.bestMove || '无可用最佳走法'}
        ${stockfishAnalysis?.ponder ? `<br><strong>Ponder走法:</strong> ${stockfishAnalysis.ponder}` : ''}
      </div>
  `;
  }
  
  return html;
};

function renderBoardState(boardState: string, activations?: number[]) {
  if (boardState.length !== 64) {
    return <div style={{ color: 'red' }}>board_state 长度错误: {boardState.length}</div>;
  }
  // 生成8x8棋盘
  const rows = [];
  for (let i = 0; i < 8; i++) {
    const cells = [];
    for (let j = 0; j < 8; j++) {
      const idx = i * 8 + j;
      const piece = boardState[idx];
      const activation = activations?.[idx] ?? 0;
      // 颜色映射（可自定义）
      const bgColor = activation
        ? `rgba(255,0,0,${Math.min(1, activation)})`
        : (i + j) % 2 === 0
        ? '#f0d9b5'
        : '#b58863';
      cells.push(
        <td
          key={j}
          style={{
            width: 40,
            height: 40,
            background: bgColor,
            textAlign: 'center',
            verticalAlign: 'middle',
            fontSize: 24,
            color: /[A-Z]/.test(piece) ? '#fff' : '#000',
            position: 'relative',
          }}
        >
          {piece !== '.' ? piece : ''}
          {activations && (
            <div
              style={{
                position: 'absolute',
                bottom: 2,
                right: 2,
                fontSize: 10,
                color: '#333',
                background: 'rgba(255,255,255,0.7)',
                borderRadius: 2,
                padding: '0 2px',
              }}
            >
              {activation.toFixed(2)}
            </div>
          )}
        </td>
      );
    }
    rows.push(<tr key={i}>{cells}</tr>);
  }
  return (
    <table style={{ borderCollapse: 'collapse', margin: 'auto' }}>
      <tbody>{rows}</tbody>
    </table>
  );
}

function fenToBoardStr(fen: string): string {
  // 只取棋盘部分
  const parts = fen.split(" ");
  const boardFen = parts[0];
  let boardStr = "";
  for (const char of boardFen) {
    if (char === "/") continue;
    if (/\d/.test(char)) {
      boardStr += ".".repeat(Number(char));
    } else {
      boardStr += char;
    }
  }
  return boardStr;
}

export const FeaturesPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const [globalAnalysisCollapsed, setGlobalAnalysisCollapsed] = useState<boolean>(false);  // 全局分析折叠状态
  const [showSelfplay, setShowSelfplay] = useState<boolean>(false);  // Self-play显示状态
  const [currentFeatureKey, setCurrentFeatureKey] = useState<string | null>(null);  // 跟踪当前feature
  
  // 检查是否从circuits页面跳转过来
  const fromCircuits = searchParams.get('from') === 'circuits';
  const featureInfo = searchParams.get('feature');
  
  // 调试信息
  console.log('🔍 Feature页面参数检查:');
  console.log('  所有URL参数:', Object.fromEntries(searchParams.entries()));
  console.log('  from参数:', searchParams.get('from'));
  console.log('  fromCircuits:', fromCircuits);
  console.log('  featureInfo:', featureInfo);
  console.log('  returnState参数:', searchParams.get('returnState'));
  console.log('  dictionary参数:', searchParams.get('dictionary'));
  console.log('  featureIndex参数:', searchParams.get('featureIndex'));
  
  // 解析返回状态信息
  const returnStateParam = searchParams.get('returnState');
  let returnState = null;
  if (returnStateParam) {
    try {
      returnState = JSON.parse(decodeURIComponent(returnStateParam));
    } catch (error: any) {
      console.error('解析返回状态失败:', error);
    }
  }
  
  // 返回Circuit页面的处理函数
  const handleReturnToCircuits = useCallback(() => {
    if (returnState) {
      // 构建返回URL，包含所有状态信息
      const returnUrl = `/circuits?returnState=${encodeURIComponent(JSON.stringify(returnState))}`;
      navigate(returnUrl);
    } else {
      // 如果没有状态信息，使用浏览器后退
      window.history.back();
    }
  }, [returnState, navigate]);
  
  // 优化全局折叠按钮的点击处理函数
  const handleGlobalAnalysisToggle = useCallback(() => {
    setGlobalAnalysisCollapsed(prev => !prev);
  }, []);

  // Self-play开关处理函数
  const handleSelfplayToggle = useCallback(() => {
    setShowSelfplay(prev => !prev);
  }, []);

  // 在页面加载时重置统计
  useEffect(() => {
    globalStatsManager.reset(0);
  }, []);
  
  // 取消所有feature相关的推演任务
  const cancelAllFeatureTasks = async () => {
    try {
      console.log('🔄 开始取消所有feature相关的任务...');
      
      // 首先取消所有会话（包括引擎分析）
      const sessionResponse = await fetch(`${import.meta.env.VITE_BACKEND_URL}/cancel_all_sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (sessionResponse.ok) {
        const sessionResult = await sessionResponse.json();
        console.log(`🛑 取消了 ${sessionResult.cancelled_count} 个会话和任务`);
      }
      
      // 然后取消所有self-play任务
      const taskResponse = await fetch(`${import.meta.env.VITE_BACKEND_URL}/tasks/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pattern: 'selfplay_' })  // 取消所有推演任务
      });
      
      if (taskResponse.ok) {
        const taskResult = await taskResponse.json();
        console.log(`🛑 额外取消了 ${taskResult.cancelled_count} 个推演任务`);
        return taskResult.cancelled_count;
      } else {
        console.error('❌ 取消任务失败，HTTP状态:', taskResponse.status);
        const errorText = await taskResponse.text();
        console.error('❌ 错误详情:', errorText);
      }
    } catch (error) {
      console.error('❌ 取消feature任务失败:', error);
      console.error('❌ 错误堆栈:', error.stack);
    }
    return 0;
  };
  
  // 强制清理所有任务（紧急情况使用）
  const forceCleanAllTasks = async () => {
    try {
      console.log('🚨 强制清理所有任务...');
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/tasks/force_clear`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log(`🚨 强制清理了 ${result.cleared_count} 个任务`);
        return result.cleared_count;
      } else {
        console.error('❌ 强制清理失败，HTTP状态:', response.status);
      }
    } catch (error) {
      console.error('❌ 强制清理任务失败:', error);
    }
    return 0;
  };


  const [dictionariesState, fetchDictionaries] = useAsyncFn(async () => {
    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries`)
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [selectedDictionary, setSelectedDictionary] = useState<string | null>(null);

  const [analysesState, fetchAnalyses] = useAsyncFn(async (dictionary: string) => {
    if (!dictionary) return [];

    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/analyses`)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(null);

  // Metric filtering state
  const [metricsState, fetchMetrics] = useAsyncFn(async (dictionary: string) => {
    if (!dictionary) return [];

    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/metrics`)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.json())
      .then((res) => z.object({ metrics: z.array(z.string()) }).parse(res).metrics);
  });

  const [metricFilters, setMetricFilters] = useState<Record<string, { min?: number; max?: number }>>({});
  const [featureCount, setFeatureCount] = useState<number | null>(null);

  const [featureIndex, setFeatureIndex] = useState<number>(0);
  const [inputValue, setInputValue] = useState<string>("0");
  const [loadingRandomFeature, setLoadingRandomFeature] = useState<boolean>(false);
  
  // 当dictionary改变时清理所有任务和状态
  useEffect(() => {
    if (selectedDictionary) {
      console.log('📚 Dictionary变化，清理所有任务和状态');
      // 清理分析状态
      globalAnalysisStateManager.clear();
      globalStatsManager.reset(0);
    }
  }, [selectedDictionary]);
  
  // 当特征改变时清理分析状态
  useEffect(() => {
    if (featureIndex !== 0) {
      globalAnalysisStateManager.clear();
      // 传递一个非零值给reset，这样就不会重置统计数据
      globalStatsManager.reset(1);
    }
  }, [featureIndex]);

  // Debounce the input value to avoid excessive updates
  useDebounce(
    () => {
      const parsed = parseInt(inputValue);
      if (!isNaN(parsed) && parsed !== featureIndex) {
        setFeatureIndex(parsed);
      }
    },
    300,
    [inputValue]
  );

  const handleFeatureIndexChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  }, []);

  // Function to count features matching filters
  const [countState, countFeatures] = useAsyncFn(
    async (
      dictionary: string | null,
      analysisName: string | null = null,
      metricFilters?: Record<string, { min?: number; max?: number }>
    ) => {
      if (!dictionary) {
        return 0;
      }

      // Build query parameters
      const params = new URLSearchParams();
      if (analysisName) {
        params.append("feature_analysis_name", analysisName);
      }
      
      // Add metric filters if provided
      if (metricFilters) {
        const mongoFilters: Record<string, Record<string, number>> = {};
        
        for (const [metricName, filter] of Object.entries(metricFilters)) {
          const mongoFilter: Record<string, number> = {};
          if (filter.min !== undefined) {
            mongoFilter["$gte"] = filter.min;
          }
          if (filter.max !== undefined) {
            mongoFilter["$lte"] = filter.max;
          }
          
          if (Object.keys(mongoFilter).length > 0) {
            mongoFilters[metricName] = mongoFilter;
          }
        }
        
        if (Object.keys(mongoFilters).length > 0) {
          params.append("metric_filters", JSON.stringify(mongoFilters));
        }
      }

      const queryString = params.toString();
      const url = `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/count${queryString ? `?${queryString}` : ""}`;

      const response = await fetch(url, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const data = await response.json();
      const count = z.object({ count: z.number() }).parse(data).count;
      setFeatureCount(count);
      return count;
    }
  );

  const [featureState, fetchFeature] = useAsyncFn(
    async (
      dictionary: string | null,
      featureIndex: number | string = "random",
      analysisName: string | null = null,
      metricFilters?: Record<string, { min?: number; max?: number }>
    ) => {
      if (!dictionary) {
        alert("Please select a dictionary first");
        return;
      }

             console.log('🚀 开始切换到新feature:', featureIndex);
       
       // 切换feature时，取消之前所有的推演任务
       await cancelAllFeatureTasks();
       
       // 重置规则统计（暂时不知道新feature的激活次数，在渲染时会重新设置）
       globalStatsManager.reset(0);

       setLoadingRandomFeature(featureIndex === "random");

      // Build query parameters
      const params = new URLSearchParams();
      if (analysisName) {
        params.append("feature_analysis_name", analysisName);
      }
      
      // Add metric filters if provided and we're fetching a random feature
      if (metricFilters && featureIndex === "random") {
        const mongoFilters: Record<string, Record<string, number>> = {};
        
        for (const [metricName, filter] of Object.entries(metricFilters)) {
          const mongoFilter: Record<string, number> = {};
          if (filter.min !== undefined) {
            mongoFilter["$gte"] = filter.min;
          }
          if (filter.max !== undefined) {
            mongoFilter["$lte"] = filter.max;
          }
          
          if (Object.keys(mongoFilter).length > 0) {
            mongoFilters[metricName] = mongoFilter;
          }
        }
        
        if (Object.keys(mongoFilters).length > 0) {
          params.append("metric_filters", JSON.stringify(mongoFilters));
        }
      }

      const queryString = params.toString();
      const url = `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}${queryString ? `?${queryString}` : ""}`;

      const feature = await fetch(url, {
        method: "GET",
        headers: {
          Accept: "application/x-msgpack",
        },
      })
        .then(async (res) => {
          if (!res.ok) {
            throw new Error(await res.text());
          }
          return res;
        })
        .then(async (res) => await res.arrayBuffer())
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .then((res) => decode(new Uint8Array(res)) as any)
        .then((res) =>
          camelcaseKeys(res, {
            deep: true,
            stopPaths: ["sample_groups.samples.context", "sample_groups.samples.context_idx"],
          })
        )
        .then((res) => FeatureSchema.parse(res));
      
      setFeatureIndex(feature.featureIndex);
      setSelectedAnalysis(feature.analysisName);
      setSearchParams({
        dictionary,
        featureIndex: feature.featureIndex.toString(),
        analysis: feature.analysisName,
      });
      console.log("Feature:", feature);
      return feature;
    }
  );

  useMount(async () => {
    await fetchDictionaries();
    if (searchParams.get("dictionary")) {
      const dict = searchParams.get("dictionary")!;
      const analysisParam = searchParams.get("analysis");
      setSelectedDictionary(dict);

      fetchAnalyses(dict).then((analyses) => {
        if (analyses.length > 0) {
          setSelectedAnalysis(analysisParam || analyses[0]);
        }
      });

      fetchMetrics(dict);

      if (searchParams.get("featureIndex")) {
        setFeatureIndex(parseInt(searchParams.get("featureIndex")!));
        fetchFeature(dict, searchParams.get("featureIndex")!, analysisParam || null);
      }
    }
  });

  useEffect(() => {
    if (dictionariesState.value && selectedDictionary === null) {
      setSelectedDictionary(dictionariesState.value[0]);
      fetchAnalyses(dictionariesState.value[0]).then((analyses) => {
        if (analyses.length > 0) {
          setSelectedAnalysis(analyses[0]);
        }
      });

      fetchMetrics(dictionariesState.value[0]);
      fetchFeature(dictionariesState.value[0], "random", selectedAnalysis);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dictionariesState.value]);

  useEffect(() => {
    if (selectedDictionary) {
      fetchAnalyses(selectedDictionary);
      fetchMetrics(selectedDictionary);
      setSelectedAnalysis(null);
      setMetricFilters({});
      setFeatureCount(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDictionary]);

  // Handle metric filter changes
  const handleMetricFilterChange = useCallback((metricName: string, type: 'min' | 'max', value: string) => {
    setMetricFilters(prev => ({
      ...prev,
      [metricName]: {
        ...prev[metricName],
        [type]: value === '' ? undefined : parseFloat(value)
      }
    }));
    // Clear count when filters change
    setFeatureCount(null);
  }, []);

  // Handle clear filters
  const handleClearFilters = useCallback(() => {
    setMetricFilters({});
    setFeatureCount(null);
  }, []);

  // Memoize sections calculation
  const sections = useMemo(() => [
    {
      title: "Histogram",
      id: "Histogram",
    },
    {
      title: "Decoder Norms",
      id: "DecoderNorms",
    },
    {
      title: "Similarity Matrix",
      id: "DecoderSimilarityMatrix",
    },
    {
      title: "Inner Product Matrix",
      id: "DecoderInnerProductMatrix",
    },
    {
      title: "Logits",
      id: "Logits",
    },
    {
      title: "Top Activation",
      id: "Activation",
    },
  ].filter((section) => (featureState.value && featureState.value.logits != null) || section.id !== "Logits"), [featureState.value]);

  return (
    <div id="Top">
      <AppNavbar />
      <div className="pt-4 pb-20 px-20 flex flex-col items-center gap-12">
        {/* 返回按钮 - 当从circuits页面跳转过来时显示 */}
        {fromCircuits && (
          <div className="w-full max-w-6xl">
            <div className="flex items-center gap-3 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <div className="flex items-center gap-2 flex-1">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <div className="flex flex-col">
                  <span className="text-sm font-medium text-blue-800">
                    从电路页面跳转查看Feature
                  </span>
                  {returnState?.nodeInfo && (
                    <div className="text-xs text-blue-600 mt-1">
                      节点信息: {returnState.nodeInfo.type} (层{returnState.nodeInfo.layer}, 位置{returnState.nodeInfo.position})
                      {returnState.nodeInfo.featureIndex !== undefined && `, 特征索引${returnState.nodeInfo.featureIndex}`}
                    </div>
                  )}
                </div>
              </div>
              <Button 
                onClick={handleReturnToCircuits}
                variant="outline" 
                size="sm"
                className="ml-auto"
              >
                ← 返回电路页面
              </Button>
            </div>
          </div>
        )}
        <div className="container grid grid-cols-[auto_600px_auto_auto] justify-center items-center gap-4">
          <span className="font-bold justify-self-end">Select dictionary:</span>
          <Select
            disabled={dictionariesState.loading || featureState.loading}
            value={selectedDictionary || undefined}
            onValueChange={async (value) => {
              console.log('📚 切换Dictionary:', value);
              // 切换dictionary时取消所有正在进行的任务
              await cancelAllFeatureTasks();
              setSelectedDictionary(value);
            }}
          >
            <SelectTrigger className="bg-white">
              <SelectValue placeholder="Select a dictionary" />
            </SelectTrigger>
            <SelectContent>
              {dictionariesState.value?.map((dictionary, i) => (
                <SelectItem key={i} value={dictionary}>
                  {dictionary}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            disabled={dictionariesState.loading || featureState.loading}
            onClick={async () => {
              await fetchFeature(selectedDictionary, "random", selectedAnalysis);
            }}
          >
            Go
          </Button>
          <span className="font-bold"></span>

          <span className="font-bold justify-self-end">Select analysis:</span>
          <Select
            disabled={analysesState.loading || !selectedDictionary || featureState.loading}
            value={selectedAnalysis || undefined}
            onValueChange={setSelectedAnalysis}
          >
            <SelectTrigger className="bg-white">
              <SelectValue placeholder="Select an analysis" />
            </SelectTrigger>
            <SelectContent>
              {analysesState.value?.map((analysis, i) => (
                <SelectItem key={i} value={analysis}>
                  {analysis}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            disabled={analysesState.loading || !selectedDictionary || featureState.loading}
            onClick={async () => {
              await fetchFeature(selectedDictionary, featureIndex, selectedAnalysis);
            }}
          >
            Apply
          </Button>
          <span className="font-bold"></span>

          <span className="font-bold justify-self-end">Choose a specific feature:</span>
          <Input
            disabled={dictionariesState.loading || selectedDictionary === null || featureState.loading}
            id="feature-input"
            className="bg-white"
            type="number"
            value={inputValue}
            onChange={handleFeatureIndexChange}
          />
          <Button
            disabled={dictionariesState.loading || selectedDictionary === null || featureState.loading}
            onClick={async () => await fetchFeature(selectedDictionary, featureIndex, selectedAnalysis)}
          >
            Go
          </Button>
          <Button
            disabled={dictionariesState.loading || selectedDictionary === null || featureState.loading}
            onClick={async () => {
              await fetchFeature(selectedDictionary, "random", selectedAnalysis, metricFilters);
            }}
          >
            Show Random Feature
          </Button>

          {/* Metric filters section */}
          {metricsState.value && metricsState.value.length > 0 && (
            <>
              <span className="font-bold justify-self-end">Metric filters:</span>
              <div className="bg-white p-4 rounded-lg border grid grid-cols-2 gap-4 col-span-2">
                {metricsState.value.map((metric) => (
                  <div key={metric} className="flex flex-col gap-2">
                    <label className="text-sm font-medium">{metric}</label>
                    <div className="flex gap-2">
                      <Input
                        placeholder="Min"
                        type="number"
                        step="any"
                        value={metricFilters[metric]?.min?.toString() || ''}
                        onChange={(e) => handleMetricFilterChange(metric, 'min', e.target.value)}
                        className="text-xs"
                      />
                      <Input
                        placeholder="Max"
                        type="number"
                        step="any"
                        value={metricFilters[metric]?.max?.toString() || ''}
                        onChange={(e) => handleMetricFilterChange(metric, 'max', e.target.value)}
                        className="text-xs"
                      />
                    </div>
                  </div>
                ))}
              </div>
              <span className="font-bold"></span>

              <span className="font-bold justify-self-end">Filter actions:</span>
              <div className="flex gap-2 items-center">
                <Button
                  disabled={dictionariesState.loading || selectedDictionary === null || countState.loading}
                  onClick={async () => {
                    await countFeatures(selectedDictionary, selectedAnalysis, metricFilters);
                  }}
                >
                  Count Features
                </Button>
                <Button
                  variant="outline"
                  disabled={dictionariesState.loading || selectedDictionary === null}
                  onClick={handleClearFilters}
                >
                  Clear Filters
                </Button>
                {featureCount !== null && (
                  <span className="text-sm font-medium ml-2">
                    {countState.loading ? "Counting..." : `Found ${featureCount} features`}
                  </span>
                )}
              </div>
              <span className="font-bold"></span>
              <span className="font-bold"></span>
            </>
          )}

          {/* Self-play 控制开关 */}
          <span className="font-bold justify-self-end">显示选项:</span>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={globalAnalysisCollapsed}
                onChange={handleGlobalAnalysisToggle}
                className="rounded"
              />
              全局折叠分析
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={showSelfplay}
                onChange={handleSelfplayToggle}
                className="rounded"
              />
              🎮 Self-play 推演
            </label>
          </div>
          <span className="font-bold"></span>
          <span className="font-bold"></span>
        </div>

        {featureState.loading && !loadingRandomFeature && (
          <div>
            Loading Feature <span className="font-bold">#{featureIndex}</span>...
          </div>
        )}
        {featureState.loading && loadingRandomFeature && <div>Loading Random Living Feature...</div>}
        {featureState.error && <div className="text-red-500 font-bold">Error: {featureState.error.message}</div>}
        {!featureState.loading && featureState.value && (
          <div className="flex flex-col gap-8 w-full">
            {/* 检查是否包含象棋相关数据 */}
            {(() => {
              const feature = featureState.value;
              

              
              const chessBoards: JSX.Element[] = [];
              
              // 添加详细统计信息变量
              let totalSamples = 0;
              let samplesWithText = 0;
              let samplesWithActivations = 0;
              let validFENFound = 0;
              let failedFENGeneration = 0;
              let boardIndex = 0; // 用于计算延迟时间
              
              // 获取激活次数限制，用于限制统计分析的棋盘数量
              const maxActivationTimes = feature.actTimes || 0;
              console.log(`Feature ${feature.featureIndex} 的激活次数: ${maxActivationTimes}`);
              
              // 重置统计管理器并设置正确的最大棋盘限制
              globalStatsManager.reset(maxActivationTimes);
              
              // 查找所有可能包含FEN的sample（不限制字典名）
              for (const [groupIndex, group] of feature.sampleGroups.entries()) {
                totalSamples += group.samples.length;
                
                // 如果已达到激活次数限制，跳出外层循环
                if (validFENFound >= maxActivationTimes) {
                  console.log(`外层循环: 已达到激活次数限制 ${maxActivationTimes}，停止处理后续样本组`);
                  break;
                }
                
                for (const [sampleIndex, sample] of group.samples.entries()) {
                  
                  // 如果已达到激活次数限制，跳出内层循环
                  if (validFENFound >= maxActivationTimes) {
                    console.log(`内层循环: 已达到激活次数限制 ${maxActivationTimes}，停止处理后续样本`);
                    break;
                  }
                  
                  if (sample.text) samplesWithText++;
                  if (sample.featureActs && sample.featureActs.length >= 64) samplesWithActivations++;
                  
                  if (sample.text) {
                    // 更详细的FEN检测和错误诊断
                    const lines = sample.text.split('\n');
                    
                    for (const [lineIndex, line] of lines.entries()) {
                      // 如果已达到激活次数限制，跳出行循环
                      if (validFENFound >= maxActivationTimes) {
                        console.log(`行循环: 已达到激活次数限制 ${maxActivationTimes}，停止处理后续行`);
                        break;
                      }
                      
                      const trimmed = line.trim();
                      
                      // 检查是否包含FEN的各个部分
                      const parts = trimmed.split(/\s+/);
                      
                      if (parts.length >= 6) {
                        const [boardPart, activeColor, castling, enPassant, halfmove, fullmove] = parts;
                        
                        // 检查棋盘部分是否符合FEN格式
                        const boardRows = boardPart.split('/');
                        
                        if (boardRows.length === 8) {
                          // 检查每一行是否符合FEN格式
                          let validBoard = true;
                          for (const row of boardRows) {
                            if (!/^[rnbqkpRNBQKP1-8]+$/.test(row)) {
                              validBoard = false;
                              break;
                            }
                          }
                          
                          // 检查行棋方
                          if (validBoard && /^[wb]$/.test(activeColor)) {
                            
                            // 检查是否已达到激活次数限制
                            if (validFENFound >= maxActivationTimes) {
                              console.log(`已达到激活次数限制 ${maxActivationTimes}，停止生成棋盘`);
                              break; // 达到激活次数限制，停止生成棋盘
                            }
                            
                            // 使用SimpleChessBoard组件进行渐进式Stockfish分析
                            try {
                              // 生成绝对唯一的key
                              const uniqueKey = `chess-${groupIndex}-${sampleIndex}-${lineIndex}-${Date.now()}-${++boardCounter}-${Math.random().toString(36).substr(2, 9)}`;
                              console.log(`🎯 创建棋盘key: ${uniqueKey} (第${validFENFound + 1}/${maxActivationTimes}个)`);
                              
                              // 计算延迟时间：每个棋盘延迟1秒，避免同时请求过多
                              const delayTime = boardIndex * 1000; // 每个棋盘延迟1秒
                              
                              chessBoards.push(
                                <SimpleChessBoard
                                  key={uniqueKey}
                                  fen={trimmed}
                                  activations={sample.featureActs}
                                  sampleIndex={sampleIndex}
                                  analysisName={group.analysisName}
                                  contextId={(sample as any).context_idx}
                                  delayMs={delayTime}
                                  autoAnalyze={true}
                                  includeInStats={true}  // 所有生成的棋盘都参与统计
                                  globalAnalysisCollapsed={globalAnalysisCollapsed}
                                  showSelfplay={showSelfplay}
                                />
                              );
                              
                              boardIndex++; // 递增棋盘索引
                              validFENFound++;
                              break; // 找到一个有效FEN就跳出行循环
                            } catch (error) {
                              console.error('生成棋盘时出错:', error, '对于FEN:', trimmed);
                            }
                          } else {
                            console.log('无效的行棋方或棋盘');
                          }
                        } else {
                          console.log('棋盘行数不是8:', boardRows.length);
                        }
                      } else {
                        console.log('FEN部分数量不足6:', parts.length);
                      }
                    }
                  } else {
                    console.log('无效的行棋方或棋盘');
                  }
                }
              }
              
              if (chessBoards.length > 0) {
                console.log(`🎯 最终生成了 ${chessBoards.length} 个棋盘，激活次数限制: ${maxActivationTimes}`);
                
                // 直接设置正确的总棋盘数，而不是依赖组件的自动计数
                globalStatsManager.setTotalBoards(chessBoards.length);
                
                return (
                  <div className="flex flex-col items-center gap-8 my-8">
                    <div className="text-center">
                      <h2 className="text-xl font-bold">Top Activation Chess Boards</h2>
                      <div className="text-sm text-gray-600 mt-2">
                        显示 {chessBoards.length} 个棋盘 (激活次数限制: {maxActivationTimes}) - 🕐 渐进式Stockfish分析中...
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        (总样本: {totalSamples}, 有文本: {samplesWithText}, 有激活值: {samplesWithActivations})
                      </div>
                    </div>
                    
                    {/* Decoder Norm 显示 */}
                    {feature.decoderNorms && (() => {
                      const decoderNorms = feature.decoderNorms!; // 类型断言，因为我们已经检查了存在性
                      return (
                        <div className="w-full max-w-4xl mx-auto mb-8">
                          <div className="bg-white p-6 rounded-lg border shadow-sm">
                            <h3 className="text-lg font-bold text-center mb-4">Feature #{feature.featureIndex} Decoder Norms</h3>
                            <div className="text-sm text-gray-600 text-center mb-4">
                              显示当前特征的decoder权重范数分布
                            </div>
                            
                            {/* 简单的柱状图显示 */}
                            <div className="flex items-end justify-center gap-1 h-64 mb-4">
                              {decoderNorms.map((norm, index) => {
                                const maxNorm = Math.max(...decoderNorms);
                                const height = Math.min((norm / maxNorm) * 200, 200);
                                const color = norm > 1.0 ? "#ef4444" : norm > 0.5 ? "#f59e0b" : "#10b981";
                                return (
                                  <div
                                    key={index}
                                    className="flex-1 min-w-[2px] bg-gray-200 relative group"
                                    style={{ height: `${height}px`, backgroundColor: color }}
                                    title={`Index ${index}: ${norm.toFixed(4)}`}
                                  >
                                    <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                                      {norm.toFixed(4)}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                            
                            {/* 统计信息 */}
                            <div className="grid grid-cols-3 gap-4 text-center text-sm">
                              <div>
                                <div className="font-bold text-gray-700">最大值</div>
                                <div className="text-lg font-bold text-red-500">
                                  {Math.max(...decoderNorms).toFixed(4)}
                                </div>
                              </div>
                              <div>
                                <div className="font-bold text-gray-700">平均值</div>
                                <div className="text-lg font-bold text-blue-500">
                                  {(decoderNorms.reduce((a, b) => a + b, 0) / decoderNorms.length).toFixed(4)}
                                </div>
                              </div>
                              <div>
                                <div className="font-bold text-gray-700">最小值</div>
                                <div className="text-lg font-bold text-green-500">
                                  {Math.min(...decoderNorms).toFixed(4)}
                                </div>
                              </div>
                            </div>
                            
                            <div className="text-xs text-gray-500 text-center mt-4">
                              * 颜色编码: <span className="text-red-500">红色</span> (norm &gt; 1.0), <span className="text-yellow-500">黄色</span> (0.5 &lt; norm ≤ 1.0), <span className="text-green-500">绿色</span> (norm ≤ 0.5)
                            </div>
                          </div>
                        </div>
                      );
                    })()}

                    {/* 全局分析折叠按钮 */}
                    <div className="w-full max-w-4xl mx-auto mb-4">
                      <div className="bg-white p-4 rounded-lg border shadow-sm">
                        <button
                          onClick={handleGlobalAnalysisToggle}
                          className="w-full flex items-center justify-between p-3 bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded-lg transition-colors"
                        >
                          <span className="text-sm font-medium text-blue-700">
                            {globalAnalysisCollapsed ? '📋 展开所有分析' : '📋 折叠所有分析'}
                          </span>
                          <span className="text-lg text-blue-600">
                            {globalAnalysisCollapsed ? '▼' : '▲'}
                          </span>
                        </button>
                        <div className="text-xs text-gray-500 text-center mt-2">
                          {globalAnalysisCollapsed ? '点击展开查看所有棋盘的详细分析信息' : '点击折叠隐藏所有棋盘的详细分析信息，只保留棋盘和FEN'}
                        </div>
                      </div>
                    </div>
                    
                    {/* 规则统计卡片 */}
                    <div className="w-full max-w-4xl mx-auto">
                      <RuleStatisticsCard maxActivationTimes={maxActivationTimes} />
                      <div className="text-xs text-gray-500 text-center mt-2">
                        * 统计基于前{maxActivationTimes}个激活次数范围内的棋盘，包含规则、物质力量和胜负概率分析
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 w-full justify-items-center">
                      {chessBoards}
                    </div>
                  </div>
                );
              }
              
              return null;
            })()}
            
            {/* Feature Card and Navigator */}
            <div className="flex gap-12 w-full">
              <Suspense fallback={<div>Loading Feature Card...</div>}>
                <FeatureCard feature={featureState.value} />
              </Suspense>
              <SectionNavigator sections={sections} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
