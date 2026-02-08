"""
File: src/rwaengine/analysis/plotter.py
Description: Professional Dual-Panel Plotting (PnL + VIX Dashboard).
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
from loguru import logger
from matplotlib.offsetbox import AnchoredText

class PerformancePlotter:
    def __init__(self, style: str = 'dark_background'):
        plt.style.use(style)

    def _calculate_metrics(self, prices_series: pd.Series) -> dict:
        """计算 Sharpe, Sortino, Return"""
        daily_ret = prices_series.pct_change().dropna()
        risk_free_rate = 0.04 / 252
        total_ret = prices_series.iloc[-1] - 1.0

        mean_ret = daily_ret.mean()
        std_dev = daily_ret.std()

        if std_dev == 0:
            return {"Sharpe": "0.00", "Sortino": "0.00", "Return": f"{total_ret:.2%}"}

        sharpe = (mean_ret - risk_free_rate) / std_dev * np.sqrt(252)

        downside = daily_ret[daily_ret < 0]
        down_std = downside.std()
        sortino = (mean_ret - risk_free_rate) / down_std * np.sqrt(252) if down_std > 0 else 0

        return {
            "Sharpe": f"{sharpe:.2f}",
            "Sortino": f"{sortino:.2f}",
            "Return": f"{total_ret:.2%}"
        }

    def plot_comparison(
        self,
        df_cumulative: pd.DataFrame,
        title: str = "Comparative Returns",
        vix_series: pd.Series = None
    ):
        """
        绘制双面板专业图表：
        - 上部: 累计收益率
        - 下部: VIX 波动率仪表盘
        """
        logger.info("🎨 Generating professional dual-panel plot...")

        # 1. 设置画布与网格 (3:1 比例)
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

        # 2. 定义子图
        ax_main = plt.subplot(gs[0])
        ax_vix = plt.subplot(gs[1], sharex=ax_main) # 共享 X 轴

        # === TOP PANEL: Cumulative Returns ===
        color_map = {
            "Black-Litterman": "#0055ff",   # Electric Blue
            "Markowitz (MV)": "#ff9900",    # Orange
            "Market (SPY)": "#00cc00",      # Green
            "Equal Weight": "#ff3333"       # Red
        }
        default_colors = list(color_map.values())

        df_pct = (df_cumulative - 1) * 100
        metrics_data = []

        for i, col in enumerate(df_pct.columns):
            color = color_map.get(col, default_colors[i % len(default_colors)])
            # 主线
            ax_main.plot(df_pct.index, df_pct[col], label=col, linewidth=2, color=color, alpha=0.9)

            # 计算指标
            m = self._calculate_metrics(df_cumulative[col])
            metrics_data.append({
                "text": f"{col}:\nSharpe: {m['Sharpe']}\nReturn: {m['Return']}",
                "color": color
            })

        # Top Panel 样式
        ax_main.set_title(title, fontsize=18, color='white', pad=20, weight='bold')
        ax_main.set_ylabel("Cumulative Return (%)", fontsize=12)
        ax_main.grid(True, which='major', color='#444444', linestyle='-', linewidth=0.5, alpha=0.3)
        ax_main.legend(loc='upper left', fontsize=10, facecolor='#1a1a1a', edgecolor='gray')
        ax_main.yaxis.set_major_formatter(mtick.PercentFormatter())

        plt.setp(ax_main.get_xticklabels(), visible=False)

        start_y = 0.75
        for i, item in enumerate(metrics_data):
            ax_main.text(
                0.02, start_y - (i * 0.10),
                item["text"],
                transform=ax_main.transAxes,
                fontsize=9, color='white', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111", edgecolor=item["color"], linewidth=1.5, alpha=0.8)
            )

        if vix_series is not None:
            # 对齐数据
            vix = vix_series.reindex(df_cumulative.index).ffill().fillna(0)

            # 绘制 VIX 曲线
            ax_vix.plot(vix.index, vix, color='#aa00ff', linewidth=1.5, label='^VIX')
            ax_vix.fill_between(vix.index, vix, 0, color='#aa00ff', alpha=0.1)

            # 绘制阈值线
            ax_vix.axhline(y=25, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax_vix.text(vix.index[0], 26, 'PANIC (>25)', color='red', fontsize=8, fontweight='bold')

            ax_vix.axhline(y=15, color='green', linestyle='--', linewidth=1, alpha=0.7)
            ax_vix.text(vix.index[0], 16, 'CALM (<15)', color='green', fontsize=8, fontweight='bold')

            # 高危区域着色 (High Risk)
            ax_vix.fill_between(vix.index, vix, 25, where=(vix >= 25), color='red', alpha=0.3, interpolate=True)

            # 样式
            ax_vix.set_ylabel("VIX Index", fontsize=10)
            ax_vix.set_ylim(10, max(40, vix.max() + 5)) # 动态 Y 轴
            ax_vix.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax_vix.set_facecolor('#0f0f0f') # 稍微深一点的背景区分

        # 保存
        output_file = "comparison_result.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.success(f"📈 Professional Dashboard saved to {output_file}")

    def plot_allocation_history(self, weights_dict: dict):
        """
        绘制持仓堆叠图 (保持原有逻辑，仅优化样式)
        """
        logger.info(" Generating allocation history plots...")
        for strat_name, df_weights in weights_dict.items():
            if df_weights.empty: continue

            fig, ax = plt.subplots(figsize=(16, 8))

            # 排序：USDC 置底，其他按权重
            mean_weights = df_weights.mean().sort_values(ascending=False)
            if 'USDC' in mean_weights:
                mean_weights = mean_weights.drop('USDC')
                cols = ['USDC'] + mean_weights.index.tolist()
            else:
                cols = mean_weights.index.tolist()

            df_plot = df_weights[cols]

            colors = plt.cm.tab20.colors
            ax.stackplot(df_plot.index, df_plot.T, labels=df_plot.columns, colors=colors, alpha=0.85)

            ax.set_title(f"Portfolio Allocation: {strat_name}", fontsize=16, color='white', pad=15)
            ax.set_ylabel("Weight Exposure", fontsize=12)
            ax.set_ylim(0, 1.0)
            ax.margins(0, 0)

            # 辅助线
            ax.axhline(0.95, color='white', linestyle=':', alpha=0.5, label='Cash Buffer')
            ax.axhline(0.30, color='yellow', linestyle=':', alpha=0.3, label='Single Cap')

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, facecolor='#1a1a1a', edgecolor='gray')

            safe_name = strat_name.replace(" ", "_").replace("(", "").replace(")", "")
            output_file = f"allocation_{safe_name}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.success(f"  Allocation plot saved to {output_file}")