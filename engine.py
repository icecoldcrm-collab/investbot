import os
import json
import random
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List

class SelfEvolvingTradingEngine:
    def __init__(self, state_filename="trading_state_strict.json"):
        self.state_filename = state_filename
        self.tickers = ["NEE", "CEG", "AES", "DUK", "SO", "XEL"]
        self.lat = 29.76  # Texas proxy for high weather volatility
        self.lon = -95.36
        self.cash_per_asset_slot = 1000.0

        # Default initial profiles (will evolve over time based on performance)
        self.default_models = {
            "Model_A_HeatwaveFocus": {"rain_buy": 5.0, "wind_buy": 15.0, "temp_buy": 31.0, "rain_sell": 1.0},
            "Model_B_StormChase":    {"rain_buy": 10.0, "wind_buy": 12.0, "temp_buy": 33.0, "rain_sell": 2.0},
            "Model_C_Aggressive":    {"rain_buy": 3.0, "wind_buy": 10.0, "temp_buy": 29.5, "rain_sell": 0.5},
            "Model_D_Conservative":  {"rain_buy": 12.0, "wind_buy": 22.0, "temp_buy": 35.0, "rain_sell": 3.0},
            "Model_E_Balanced":      {"rain_buy": 7.5, "wind_buy": 18.0, "temp_buy": 32.0, "rain_sell": 1.5}
        }

    def load_state(self) -> Dict[str, Any]:
        """Loads persistent state, ensuring backward compatibility for parameters."""
        if os.path.exists(self.state_filename):
            try:
                with open(self.state_filename, "r") as f:
                    state = json.load(f)
                    # Automatically inject missing parameters for older state files
                    for m_name, default_params in self.default_models.items():
                        if m_name in state["models"] and "parameters" not in state["models"][m_name]:
                            state["models"][m_name]["parameters"] = default_params
                    return state
            except Exception:
                pass
        
        initial_state = {
            "last_run_date": None,
            "models": {}
        }
        for m_name, params in self.default_models.items():
            initial_state["models"][m_name] = {
                "parameters": params,
                "total_trades": 0,
                "assets": {
                    t: {"cash": self.cash_per_asset_slot, "shares": 0.0, "buy_price": 0.0} 
                    for t in self.tickers
                },
                "history": []
            }
        return initial_state

    def save_state(self, state: Dict[str, Any]):
        """Saves persistent state to disk."""
        with open(self.state_filename, "w") as f:
            json.dump(state, f, indent=4)

    def fetch_recent_data(self) -> Dict[str, pd.DataFrame]:
        """Fetches actual recent market and weather data using forecast/archive blend."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        
        data_cache = {}
        for ticker in self.tickers:
            stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if stock_df.empty:
                continue
            if isinstance(stock_df.columns, pd.MultiIndex):
                stock_df.columns = stock_df.columns.get_level_values(0)

            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": self.lat, "longitude": self.lon,
                "start_date": start_date, "end_date": end_date,
                "hourly": "temperature_2m,precipitation,wind_speed_10m"
            }
            try:
                response = requests.get(weather_url, params=weather_params, timeout=10)
                w_json = response.json()
            except Exception:
                w_json = {}

            if "hourly" not in w_json:
                print(f"[WARNING] Weather API failed for {ticker}. Skipping ticker for this cycle.")
                continue

            hourly = w_json["hourly"]
            w_df = pd.DataFrame({
                "time": pd.to_datetime(hourly["time"]),
                "temperature_2m": hourly["temperature_2m"],
                "precipitation": hourly["precipitation"],
                "wind_speed": hourly["wind_speed_10m"]
            })
            w_df.set_index("time", inplace=True)
            weather_df = w_df.resample('D').mean().dropna()

            merged = pd.concat([stock_df[['Close']], weather_df], axis=1).dropna()
            if not merged.empty:
                data_cache[ticker] = merged
                
        return data_cache

    def evolve_parameters(self, state: Dict[str, Any]):
        """Genetic Algorithm step: Learns from losses by mutating parameters of underperforming models based on the winner."""
        models = state["models"]
        
        # Find the best performing model based on its latest return percentage
        best_model_name = max(
            models.keys(), 
            key=lambda m: models[m]["history"][-1]["return_pct"] if models[m]["history"] else 0.0
        )
        best_params = models[best_model_name]["parameters"]
        print(f"[EVOLUTION] Best performing model: {best_model_name}. Mutating lower performers based on its DNA.")

        # Mutate parameters for other models to explore better thresholds
        for m_name, m_data in models.items():
            if m_name == best_model_name:
                continue # Keep the champion intact
            
            mutated_params = {}
            for param, val in best_params.items():
                # Add a small random mutation (-15% to +15% variation)
                mutation_factor = random.uniform(0.85, 1.15)
                mutated_params[param] = round(val * mutation_factor, 2)
            
            # Apply evolved parameters
            m_data["parameters"] = mutated_params

    def run_daily_cycle(self):
        """Executes the daily cycle and triggers self-learning evolution."""
        state = self.load_state()
        market_data = self.fetch_recent_data()
        
        if not market_data:
            print("[ERROR] Failed to fetch recent market data. Aborting cycle.")
            return

        today_str = datetime.now().strftime("%Y-%m-%d")
        print("=" * 86)
        print(f"{f'SELF-EVOLVING PAPER TRADING ENGINE | DATE: {today_str}':^86}")
        print("=" * 86)

        leaderboard = []
        initial_total_capital = self.cash_per_asset_slot * len(self.tickers)

        for model_name, model_data in state["models"].items():
            strategy = model_data["parameters"]
            assets = model_data["assets"]
            trades_today = 0

            for ticker, df in market_data.items():
                if ticker not in assets or df.empty:
                    continue
                
                latest_row = df.iloc[-1]
                close_price = float(latest_row['Close'])
                precip = float(latest_row.get('precipitation', 0.0))
                wind = float(latest_row.get('wind_speed', 10.0))
                temp = float(latest_row.get('temperature_2m', 25.0))

                is_buy_signal = (precip > strategy["rain_buy"]) or (wind > strategy["wind_buy"]) or (temp > strategy["temp_buy"])
                is_sell_signal = (precip < strategy["rain_sell"]) and (temp < (strategy["temp_buy"] - 3))

                asset_slot = assets[ticker]

                if is_buy_signal and asset_slot["cash"] >= close_price and asset_slot["shares"] == 0:
                    asset_slot["shares"] = asset_slot["cash"] / close_price
                    asset_slot["buy_price"] = close_price
                    asset_slot["cash"] = 0.0
                    trades_today += 1
                    model_data["total_trades"] += 1
                elif is_sell_signal and asset_slot["shares"] > 0:
                    asset_slot["cash"] = asset_slot["shares"] * close_price
                    asset_slot["shares"] = 0.0
                    asset_slot["buy_price"] = 0.0
                    trades_today += 1
                    model_data["total_trades"] += 1

            total_portfolio_value = 0.0
            for ticker, asset_slot in assets.items():
                slot_value = asset_slot["cash"]
                if asset_slot["shares"] > 0 and ticker in market_data:
                    current_price = float(market_data[ticker].iloc[-1]['Close'])
                    slot_value = asset_slot["shares"] * current_price
                total_portfolio_value += slot_value

            net_profit = total_portfolio_value - initial_total_capital
            net_return_pct = (net_profit / initial_total_capital) * 100

            model_data["history"].append({
                "date": today_str,
                "portfolio_value": round(total_portfolio_value, 2),
                "return_pct": round(net_return_pct, 2)
            })

            leaderboard.append({
                "Model": model_name,
                "Value": f"${total_portfolio_value:.2f}",
                "Net Yield": f"{net_return_pct:+.2f}%",
                "Trades Today": trades_today,
                "Total Trades": model_data["total_trades"]
            })

        # Trigger learning/evolution after evaluating today's performance
        self.evolve_parameters(state)

        state["last_run_date"] = today_str
        self.save_state(state)

        lb_df = pd.DataFrame(leaderboard)
        print(lb_df.to_string(index=False))
        print("=" * 86)
        print("Adaptive learning cycle completed and state saved.")
        print("=" * 86)

if __name__ == "__main__":
    engine = SelfEvolvingTradingEngine()
    engine.run_daily_cycle()
