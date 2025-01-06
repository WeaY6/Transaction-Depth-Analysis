from binance.spot import Spot
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings


class DepthAnalyzer:
    def __init__(self, symbol="BTCUSDT", depth_levels=1000):
        self.client = Spot()
        self.symbol = symbol
        self.depth_levels = depth_levels

    def get_order_book(self) -> Dict:
        """获取当前订单簿数据"""
        return self.client.depth(self.symbol, limit=self.depth_levels)

    def calculate_total_depth(self, order_book: Dict) -> Dict[str, float]:
        """计算总深度
        返回买卖双方的总挂单量和总价值
        """
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity'], dtype=float)

        metrics = {
            'bid_volume': bids['quantity'].sum(),
            'ask_volume': asks['quantity'].sum(),
            'bid_value': (bids['price'] * bids['quantity']).sum(),
            'ask_value': (asks['price'] * asks['quantity']).sum(),
            'depth_bias': (bids['quantity'].sum() - asks['quantity'].sum()) /
                          (bids['quantity'].sum() + asks['quantity'].sum())
        }

        return metrics

    def calculate_localized_depth(self, order_book: Dict,
                                  price_ranges: List[float] = [0.1, 0.3, 0.5, 1.0]) -> Dict:
        """计算局部深度
        在不同价格范围内分析深度分布
        """
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity'], dtype=float)

        mid_price = (float(bids['price'].iloc[0]) + float(asks['price'].iloc[0])) / 2

        localized_metrics = {
            'mid_price': mid_price,
            'ranges': {}
        }

        for range_pct in price_ranges:
            price_range = mid_price * (range_pct / 100)
            lower_bound = mid_price - price_range
            upper_bound = mid_price + price_range

            bids_in_range = bids[bids['price'] >= lower_bound]
            asks_in_range = asks[asks['price'] <= upper_bound]

            range_metrics = {
                'bid_volume': bids_in_range['quantity'].sum(),
                'ask_volume': asks_in_range['quantity'].sum(),
                'bid_value': (bids_in_range['price'] * bids_in_range['quantity']).sum(),
                'ask_value': (asks_in_range['price'] * asks_in_range['quantity']).sum(),
                'depth_bias': self._calculate_depth_bias(bids_in_range, asks_in_range)
            }

            localized_metrics['ranges'][f'{range_pct}%'] = range_metrics

        return localized_metrics

    def analyze_depth_change(self, current_depth: Dict,
                             historical_depth: Dict,
                             window_size: int = 24) -> Dict:
        """分析深度变化
        比较当前深度与历史深度的差异
        """
        current_metrics = self.calculate_total_depth(current_depth)
        historical_metrics = self.calculate_total_depth(historical_depth)

        change_metrics = {
            'bid_volume_change': self._calculate_relative_change(
                current_metrics['bid_volume'],
                historical_metrics['bid_volume']
            ),
            'ask_volume_change': self._calculate_relative_change(
                current_metrics['ask_volume'],
                historical_metrics['ask_volume']
            ),
            'depth_bias_change': current_metrics['depth_bias'] - historical_metrics['depth_bias'],
            'volatility': self._calculate_depth_volatility(current_depth, historical_depth)
        }

        return change_metrics

    def monitor_order_cancellation(self,
                                   depth_snapshots: List[Dict],
                                   interval: int = 60) -> Dict:
        """监控订单撤单情况
        分析一定时间内的撤单速率和模式
        """
        cancellation_metrics = {
            'bid_cancellation_rate': [],
            'ask_cancellation_rate': [],
            'timestamps': []
        }

        for i in range(1, len(depth_snapshots)):
            prev_depth = depth_snapshots[i - 1]
            curr_depth = depth_snapshots[i]

            # 计算买卖盘的撤单率
            bid_cancel_rate = self._calculate_cancellation_rate(
                prev_depth['bids'],
                curr_depth['bids']
            )
            ask_cancel_rate = self._calculate_cancellation_rate(
                prev_depth['asks'],
                curr_depth['asks']
            )

            cancellation_metrics['bid_cancellation_rate'].append(bid_cancel_rate)
            cancellation_metrics['ask_cancellation_rate'].append(ask_cancel_rate)
            cancellation_metrics['timestamps'].append(datetime.now())

        return cancellation_metrics

    def calculate_market_impact(self,
                                order_book: Dict,
                                target_volume: float) -> Dict:
        """计算市场冲击
        模拟成交特定量对价格的影响
        """
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity'], dtype=float)

        buy_impact = self._calculate_price_impact(asks, target_volume)
        sell_impact = self._calculate_price_impact(bids, target_volume)

        return {
            'buy_impact': buy_impact,
            'sell_impact': sell_impact,
            'impact_asymmetry': buy_impact - sell_impact
        }

    def evaluate_depth_quality(self, depth_metrics: Dict) -> Dict:
        """评估深度质量
        综合多个指标评估市场深度的好坏
        """
        quality_scores = {
            'liquidity_score': self._calculate_liquidity_score(depth_metrics),
            'symmetry_score': self._calculate_symmetry_score(depth_metrics),
            'resilience_score': self._calculate_resilience_score(depth_metrics),
            'overall_score': 0  # 将由上述分数加权得出
        }

        # 计算总体评分
        weights = {'liquidity': 0.4, 'symmetry': 0.3, 'resilience': 0.3}
        quality_scores['overall_score'] = (
                weights['liquidity'] * quality_scores['liquidity_score'] +
                weights['symmetry'] * quality_scores['symmetry_score'] +
                weights['resilience'] * quality_scores['resilience_score']
        )

        return quality_scores

    def _calculate_depth_bias(self, bids: pd.DataFrame, asks: pd.DataFrame) -> float:
        """计算深度偏差"""
        total_bid = bids['quantity'].sum()
        total_ask = asks['quantity'].sum()
        return (total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0

    def _calculate_relative_change(self, current: float, past: float) -> float:
        """计算相对变化率"""
        return (current - past) / past if past != 0 else float('inf')

    def _calculate_depth_volatility(self,
                                    current_depth: Dict,
                                    historical_depth: Dict) -> float:
        """计算深度波动性"""
        current_bids = pd.DataFrame(current_depth['bids'],
                                    columns=['price', 'quantity'],
                                    dtype=float)
        historical_bids = pd.DataFrame(historical_depth['bids'],
                                       columns=['price', 'quantity'],
                                       dtype=float)

        # 计算深度分布的标准差
        current_std = current_bids['quantity'].std()
        historical_std = historical_bids['quantity'].std()

        return abs(current_std - historical_std) / historical_std if historical_std != 0 else float('inf')

    def _calculate_cancellation_rate(self,
                                     prev_orders: List,
                                     curr_orders: List) -> float:
        """计算撤单率"""
        prev_total = sum(float(order[1]) for order in prev_orders)
        curr_total = sum(float(order[1]) for order in curr_orders)

        return max(0, (prev_total - curr_total) / prev_total) if prev_total > 0 else 0

    def _calculate_price_impact(self,
                                orders: pd.DataFrame,
                                target_volume: float) -> float:
        """计算价格冲击"""
        if orders.empty:
            return float('inf')

        cumsum = orders['quantity'].cumsum()
        impact_idx = (cumsum >= target_volume).idxmax() if any(cumsum >= target_volume) else -1

        if impact_idx == -1:
            return float('inf')

        return abs(orders['price'].iloc[impact_idx] - orders['price'].iloc[0]) / orders['price'].iloc[0]

    def _calculate_liquidity_score(self, depth_metrics: Dict) -> float:
        """计算流动性评分"""
        total_volume = (depth_metrics['bid_volume'] + depth_metrics['ask_volume'])
        return min(1.0, total_volume / 1000)  # 归一化到0-1范围

    def _calculate_symmetry_score(self, depth_metrics: Dict) -> float:
        """计算对称性评分"""
        return 1 - abs(depth_metrics['depth_bias'])

    def _calculate_resilience_score(self, depth_metrics: Dict) -> float:
        """计算弹性评分"""
        if not hasattr(self, '_historical_impacts'):
            return 0.5  # 默认中等弹性

        current_impact = depth_metrics.get('market_impact', {}).get('impact_asymmetry', 0)
        historical_impact = self._historical_impacts[-1] if self._historical_impacts else 0

        return 1 / (1 + abs(current_impact - historical_impact))