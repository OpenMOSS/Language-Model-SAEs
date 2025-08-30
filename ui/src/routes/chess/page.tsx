import React from "react";
import { AppNavbar } from "@/components/app/navbar";

export const ChessTestPage = () => {
  // 示例longfen数据
  const sampleLongfen = "br..q.rk.pp..pp.....p...p......p...P.P...Pn.Q.PP..P...P..RB..K..RKQ....2..20.g8g7";
  
  // 模拟后端生成的HTML（实际情况下这会从后端获取）
  const sampleChessHTML = `
    <div style="font-family: Arial, sans-serif; margin: 20px;">
        <h3 style="color: #333; margin-bottom: 10px;">国际象棋棋盘</h3>
        <div style="display: inline-block; border: 2px solid #333; margin-bottom: 15px;">
            <table style="border-collapse: collapse; font-size: 24px;">
                <tr>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;">♜</td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;">♝</td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;">♛</td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;">♜</td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;">♚</td>
                </tr>
                <tr>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;">♟</td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;">♟</td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;">♟</td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;">♟</td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                </tr>
                <tr>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;">♟</td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                </tr>
                <tr>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;">♟</td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                </tr>
                <tr>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;">♟</td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                </tr>
                <tr>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #fff; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);">♙</td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #fff; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);">♙</td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                </tr>
                <tr>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;">♞</td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #fff; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);">♕</td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #fff; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);">♙</td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #fff; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);">♙</td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                </tr>
                <tr>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #fff; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);">♙</td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #fff; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);">♙</td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #b58863; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                    <td style="width: 40px; height: 40px; background-color: #f0d9b5; text-align: center; vertical-align: middle; border: 1px solid #999; color: #000; text-shadow: none;"></td>
                </tr>
            </table>
        </div>
        <div style="margin-top: 10px; font-size: 14px; color: #666;">
            <div><strong>当前轮次:</strong> 白方走棋</div>
            <div><strong>移动次数:</strong> 2</div>
            <div><strong>半步计数:</strong> 20</div>
            <div><strong>当前思考:</strong> <code>g8g7</code></div>
            <div style="margin-top: 10px; font-size: 12px; font-family: monospace; background: #f5f5f5; padding: 8px; border-radius: 4px;">
                <strong>Longfen:</strong> br..q.rk.pp..pp.....p...p......p...P.P...Pn.Q.PP..P...P..RB..K..RKQ....2..20.g8g7
            </div>
        </div>
    </div>
  `;

  return (
    <div className="min-h-screen bg-gray-50">
      <AppNavbar />
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-8 text-center">国际象棋可视化测试</h1>
          <div className="space-y-8">
            <div className="text-center">
              <p className="text-gray-600 mb-4">
                这是一个测试页面，用于展示当model_name包含"chess"时的棋盘可视化功能。
              </p>
              <p className="text-sm text-gray-500 mb-8">
                示例Longfen: {sampleLongfen}
              </p>
            </div>
            
            <div className="border rounded-lg p-4 bg-white">
              <div dangerouslySetInnerHTML={{ __html: sampleChessHTML }} />
            </div>
            
            <div className="mt-8 p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold mb-2">功能说明:</h3>
              <ul className="text-sm space-y-1">
                <li>• 当model_name包含"chess"时，后端会自动解析longfen字符串</li>
                <li>• 后端生成HTML格式的棋盘，前端直接显示</li>
                <li>• 显示当前轮到哪方走棋</li>
                <li>• 显示吃过路兵、王车易位权限等信息</li>
                <li>• 显示当前思考的走法</li>
                <li>• 类似IPython的display(board)方式，简单高效</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 