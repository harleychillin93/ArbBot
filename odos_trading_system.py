import requests
from web3 import Web3
from web3.middleware import PythonicMiddleware
from decimal import Decimal
import os
from dotenv import load_dotenv
import time
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from hdwallet import BIP44HDWallet
from hdwallet.cryptocurrencies import EthereumMainnet
from hdwallet.derivations import BIP44Derivation
from eth_account import Account
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio
import re 


class BalanceManager:
    def __init__(self, trader, token_info: Dict[str, str]):
        self.trader = trader
        self.token_info = token_info
        self.rebalance_history = []
        self.last_rebalance = 0
        self.total_gas_spent = Decimal('0')    
    async def check_decimal_conversion(self, amount: Decimal, token_address: str) -> Decimal:
        """Safely convert amounts considering token decimals"""
        token_info = await self.trader.get_token_info(token_address)
        decimals = token_info['decimals']
        
        # Ensure the amount is properly scaled for the token's decimals
        if amount > 1e20:  # Sanity check for unusually large amounts
            amount = amount / Decimal(str(10 ** decimals))
        
        return amount       
    async def should_rebalance(self, current_ratios: Dict[str, float], 
                             target_ratios: Dict[str, float],
                             min_interval: int = 14400,  # 4 hours
                             deviation_threshold: float = 0.15,  # 15%
                             max_gas_gwei: int = 50) -> tuple[bool, str]:
        """Determine if rebalancing is needed"""
        current_time = time.time()
        
        # Check time interval
        if current_time - self.last_rebalance < min_interval:
            return False, "Too soon since last rebalance"
            
        # Check gas price
        current_gas = self.trader.w3.eth.gas_price / 1e9
        if current_gas > max_gas_gwei:
            return False, f"Gas price too high ({current_gas:.1f} gwei)"
            
        # Check deviations
        max_deviation = 0
        for token, current in current_ratios.items():
            target = target_ratios.get(token, 0)
            deviation = abs(current - target)
            max_deviation = max(max_deviation, deviation)
            
        if max_deviation > deviation_threshold:
            return True, f"Maximum deviation {max_deviation:.1%} exceeds threshold"
            
        return False, "Balances within acceptable range"      
    async def convert_all_to_usdc(self, usdc_address: str) -> Dict:
            """Convert all assets to USDC"""
            results = []
            total_gas_cost = Decimal('0')
            
            for symbol, address in self.token_info.items():
                if address == usdc_address:
                    continue
                    
                try:
                    token_info = await self.trader.get_token_info(address)
                    if token_info['balance'] > 0:
                        # Execute trade to USDC
                        trade_result = await self.trader.execute_trade(
                            input_token=address,  # Fix: Specify parameter names
                            output_token=usdc_address,
                            amount_in_human=token_info['balance'],
                            slippage=1.0  # Higher slippage for emergency conversion
                        )
                        
                        results.append({
                            'token': symbol,
                            'amount': float(token_info['balance']),
                            'success': trade_result['status'] == 'success',
                            'gas_cost': trade_result['gas_cost_usd']
                        })
                        
                        total_gas_cost += Decimal(str(trade_result['gas_cost_usd']))
                        
                except Exception as e:
                    results.append({
                        'token': symbol,
                        'error': str(e)
                    })
                    
            return {
                'conversions': results,
                'total_gas_cost': float(total_gas_cost)
            }        
    async def track_rebalance(self, gas_cost: Decimal, tokens_traded: List[str]) -> None:
        """Track rebalancing costs and frequency"""
        self.total_gas_spent += gas_cost
        self.last_rebalance = time.time()
        
        self.rebalance_history.append({
            'timestamp': time.time(),
            'gas_cost': float(gas_cost),
            'tokens': tokens_traded,
            'cumulative_gas': float(self.total_gas_spent)
        })
        
        # Save history to file
        try:
            with open('rebalance_history.json', 'w') as f:
                json.dump({
                    'total_gas_spent': float(self.total_gas_spent),
                    'history': self.rebalance_history
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving rebalance history: {e}")         
    def get_rebalance_stats(self) -> Dict:
        """Get rebalancing statistics"""
        if not self.rebalance_history:
            return {"status": "No rebalancing history"}
            
        first_rebalance = self.rebalance_history[0]['timestamp']
        days_running = (time.time() - first_rebalance) / (24 * 3600)
        
        return {
            'total_rebalances': len(self.rebalance_history),
            'total_gas_spent': float(self.total_gas_spent),
            'average_gas_per_rebalance': float(self.total_gas_spent) / len(self.rebalance_history),
            'rebalances_per_day': len(self.rebalance_history) / days_running,
            'days_running': days_running
        }
class ArbitrageTracker:
    def __init__(self):
        self.opportunities = []
        self.executed_trades = []
        self.total_profit = Decimal('0')
        self.current_cycle = None
        self.last_rebalance = time.time()
        self.rebalance_settings = {
            'min_interval': 3600,  # Minimum time between rebalances (1 hour)
            'threshold': 0.15,     # 15% deviation triggers rebalance
            'min_gas_price': 50,   # Maximum gas price (gwei) for rebalancing
            'min_liquidity': 1000  # Minimum liquidity required in USD
        }

    def calculate_rebalance_need(self, token_balances: Dict) -> tuple[bool, str]:
        """
        Determine if rebalancing is needed based on multiple factors
        Returns: (needs_rebalance, reason)
        """
        current_time = time.time()
        time_since_last = current_time - self.last_rebalance
        
        # Check minimum time interval
        if time_since_last < self.rebalance_settings['min_interval']:
            return False, "Too soon since last rebalance"
            
        # Calculate balance ratios
        total_value_usd = sum(balance['usd_value'] for balance in token_balances.values())
        if total_value_usd == 0:
            return False, "No balance to rebalance"
            
        ratios = {
            token: balance['usd_value'] / total_value_usd 
            for token, balance in token_balances.items()
        }
        
        # Check for significant imbalance
        target_ratio = 1.0 / len(token_balances)
        max_deviation = max(abs(ratio - target_ratio) for ratio in ratios.values())
        
        if max_deviation > self.rebalance_settings['threshold']:
            # Check liquidity requirements
            all_liquid = all(
                balance['usd_value'] >= self.rebalance_settings['min_liquidity']
                for balance in token_balances.values()
            )
            if not all_liquid:
                return False, "Insufficient liquidity for safe rebalancing"
                
            return True, f"Balance deviation {max_deviation:.1%} exceeds threshold"
            
        return False, "Balances within acceptable range"

    def complete_cycle(self, reverse_trade: Dict) -> Dict:
        """Complete the current arbitrage cycle and calculate true profit"""
        if not self.current_cycle:
            raise ValueError("No active cycle to complete")
        
        cycle = self.current_cycle
        cycle['reverse_tx'] = reverse_trade['tx_hash']
        cycle['reverse_gas'] = reverse_trade['gas_cost_usd']
        
        # Calculate actual amounts received/sent
        initial_amount = Decimal(str(cycle['initial_amount']))
        final_amount = Decimal(str(reverse_trade['received_amount']))
        
        # True profit calculation
        profit_amount = final_amount - initial_amount
        profit_usd = float(profit_amount)  # Assuming amounts are in USD stablecoin
        
        # Calculate total gas cost
        total_gas_cost = Decimal(str(cycle['forward_gas'])) + Decimal(str(reverse_trade['gas_cost_usd']))
        net_profit_usd = profit_usd - float(total_gas_cost)
        
        cycle.update({
            'final_amount': float(final_amount),
            'gross_profit_usd': profit_usd,
            'total_gas_cost': float(total_gas_cost),
            'net_profit_usd': net_profit_usd,
            'status': 'completed',
            'completion_timestamp': time.time()
        })
        
        # Save completed cycle
        self.executed_trades.append(cycle)
        self.total_profit += Decimal(str(net_profit_usd))
        
        # Save trade info with corrected profit calculation
        trade_info = {
            'timestamp': cycle['timestamp'],
            'pair': cycle['pair'],
            'amount': float(initial_amount),
            'profit_usd': profit_usd,
            'gas_cost_usd': float(total_gas_cost),
            'gas_price_gwei': reverse_trade.get('gas_price_gwei', 0),
            'forward_tx': cycle['forward_tx'],
            'reverse_tx': cycle['reverse_tx']
        }
        self.save_trade_history(trade_info)
        
        # Clear current cycle
        result = cycle.copy()
        self.current_cycle = None
        
        return result

    def get_rebalance_ratios(self, current_ratios: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate optimal rebalancing ratios considering gas costs and market conditions
        """
        token_count = len(current_ratios)
        target_ratio = 1.0 / token_count
        
        # Calculate required shifts
        shifts = {
            token: target_ratio - ratio
            for token, ratio in current_ratios.items()
        }
        
        # Only adjust if deviation is significant enough to justify gas costs
        return {
            token: target_ratio if abs(shift) > self.rebalance_settings['threshold'] else ratio
            for token, (shift, ratio) in zip(shifts.keys(), zip(shifts.values(), current_ratios.values()))
        }

    def estimate_rebalance_cost(self, required_trades: List[Dict]) -> float:
        """Estimate gas cost for rebalancing trades"""
        # Assume average gas cost per trade
        avg_gas_per_trade = 150000  # Conservative estimate
        current_gas_price = self.w3.eth.gas_price / 1e9  # Convert to gwei
        
        total_gas = avg_gas_per_trade * len(required_trades)
        return total_gas * current_gas_price

    def should_execute_rebalance(self, estimated_cost: float, total_value: float) -> bool:
        """Determine if rebalancing is cost-effective"""
        # Don't rebalance if gas price is too high
        current_gas_price = self.w3.eth.gas_price / 1e9
        if current_gas_price > self.rebalance_settings['min_gas_price']:
            return False
            
        # Cost should not exceed 0.1% of total portfolio value
        max_acceptable_cost = total_value * 0.001
        return estimated_cost <= max_acceptable_cost
class RequestRateLimiter:
    def __init__(self, max_requests: int = 1000, time_window: int = 300):
        """
        Initialize rate limiter
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds (300 = 5 minutes)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def can_make_request(self) -> bool:
        """Check if we can make a new request"""
        current_time = time.time()
        # Remove old requests
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.time_window]
        return len(self.requests) < self.max_requests
    
    def add_request(self):
        """Record a new request"""
        self.requests.append(time.time())
        
    def get_remaining_requests(self) -> int:
        """Get number of remaining requests in current window"""
        self.can_make_request()  # Clean up old requests
        return self.max_requests - len(self.requests)
class ProfitThresholds:
    def __init__(self, 
                 min_profit_usd: float = 5.0,
                 min_profit_percentage: float = 0.1,  # 0.1%
                 dynamic_gas_multiplier: float = 1.5):
        self.min_profit_usd = min_profit_usd
        self.min_profit_percentage = min_profit_percentage
        self.dynamic_gas_multiplier = dynamic_gas_multiplier
        self.highest_profit_seen = Decimal('0')
        self.average_profit = Decimal('0')
        self.profit_samples = []
        
    def update_stats(self, profit_usd: Decimal):
        """Update profit statistics"""
        self.profit_samples.append(profit_usd)
        if len(self.profit_samples) > 100:  # Keep last 100 samples
            self.profit_samples.pop(0)
        self.average_profit = sum(self.profit_samples) / len(self.profit_samples)
        if profit_usd > self.highest_profit_seen:
            self.highest_profit_seen = profit_usd
            
    def is_profitable(self, 
                     profit_usd: Decimal, 
                     profit_percentage: Decimal, 
                     gas_cost_usd: Decimal) -> tuple[bool, str]:
        """
        Check if opportunity meets profit thresholds
        Returns: (is_profitable, reason)
        """
        min_gas_profit = gas_cost_usd * self.dynamic_gas_multiplier
        
        if profit_usd < self.min_profit_usd:
            return False, f"Profit (${float(profit_usd):.2f}) below minimum (${self.min_profit_usd:.2f})"
            
        if profit_percentage < self.min_profit_percentage:
            return False, f"Profit percentage ({float(profit_percentage):.3f}%) below minimum ({self.min_profit_percentage:.3f}%)"
            
        if profit_usd < min_gas_profit:
            return False, f"Profit (${float(profit_usd):.2f}) below gas cost threshold (${float(min_gas_profit):.2f})"
            
        return True, "Profitable opportunity found"
class DEXQuoteHandler:
    def __init__(self):
        self.cache = {}
        self.cache_duration = 3  # Cache duration in seconds

    def _get_cache_key(self, dex: str, token_in: str, token_out: str, amount_in_wei: int) -> str:
        return f"{dex}_{token_in}_{token_out}_{amount_in_wei}_{int(time.time() / self.cache_duration)}"

    def _is_cache_valid(self, timestamp: float) -> bool:
        return (time.time() - timestamp) < self.cache_duration


class MarketConditions:
    def __init__(self, initial_threshold: float = 1.0):
        self.base_threshold = initial_threshold
        self.volatility_windows = {
            'short': [],   # Last 5 minutes
            'medium': [],  # Last 30 minutes
            'long': []     # Last 2 hours
        }
        self.gas_history = []
        self.volume_history = []
        self.last_update = time.time()
        
    def update_volatility(self, price: Decimal, timestamp: float = None):
        """Track price volatility across different time windows"""
        if timestamp is None:
            timestamp = time.time()
            
        # Add new price point
        price_point = {'price': price, 'timestamp': timestamp}
        
        # Update windows
        self.volatility_windows['short'].append(price_point)
        self.volatility_windows['medium'].append(price_point)
        self.volatility_windows['long'].append(price_point)
        
        # Clean old data
        current_time = time.time()
        self.volatility_windows['short'] = [p for p in self.volatility_windows['short'] 
                                          if current_time - p['timestamp'] <= 300]  # 5 min
        self.volatility_windows['medium'] = [p for p in self.volatility_windows['medium'] 
                                           if current_time - p['timestamp'] <= 1800]  # 30 min
        self.volatility_windows['long'] = [p for p in self.volatility_windows['long'] 
                                         if current_time - p['timestamp'] <= 7200]  # 2 hours
        
    def calculate_dynamic_threshold(self, current_gas_price: float) -> float:
        """
        Calculate dynamic profit threshold based on market conditions
        """
        try:
            # 1. Volatility Impact
            short_vol = self._calculate_volatility('short')
            medium_vol = self._calculate_volatility('medium')
            long_vol = self._calculate_volatility('long')
            
            # Higher volatility = higher threshold needed
            volatility_factor = (short_vol * 0.5 + medium_vol * 0.3 + long_vol * 0.2)
            
            # 2. Gas Price Impact
            gas_impact = current_gas_price / 50  # Normalize to baseline of 50 gwei
            
            # 3. Volume Impact (if data available)
            volume_impact = 1.0
            if self.volume_history:
                avg_volume = sum(self.volume_history) / len(self.volume_history)
                current_volume = self.volume_history[-1]
                volume_impact = current_volume / avg_volume
            
            # Combine factors
            dynamic_threshold = self.base_threshold * (
                1 + volatility_factor * 0.5 +  # Volatility has 50% weight
                (gas_impact - 1) * 0.3 +      # Gas price has 30% weight
                (volume_impact - 1) * 0.2      # Volume has 20% weight
            )
            
            return max(self.base_threshold, min(dynamic_threshold, self.base_threshold * 3))
            
        except Exception:
            return self.base_threshold  # Fallback to base threshold
            
    def _calculate_volatility(self, window: str) -> float:
        """Calculate price volatility for a given window"""
        prices = [p['price'] for p in self.volatility_windows[window]]
        if len(prices) < 2:
            return 0
        
        returns = [(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]
        return float(Decimal(str(statistics.stdev(returns))) if returns else Decimal('0'))
class OdosTrader:
    def __init__(self, log_dir: str = "logs"):
        # Existing initialization code remains the same until token_decimals_cache
        self.token_decimals_cache = {}
        self.tracker = ArbitrageTracker()
        
        self.rate_limiter = RequestRateLimiter()
        
        # Add MATIC/WMATIC addresses as class constants
        self.WMATIC_ADDRESS = "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270"
        self.matic_price_cache = {'price': None, 'timestamp': 0}
        self.PRICE_CACHE_DURATION = 60  # Cache MATIC price for 60 seconds
        
        
        # Load environment variables
        load_dotenv()
        
        # Initialize logging
        self.setup_logging(log_dir)
        
        # API endpoints
        self.BASE_URL = "https://api.odos.xyz"
        self.QUOTE_URL = f"{self.BASE_URL}/sor/quote/v2"
        self.ASSEMBLE_URL = f"{self.BASE_URL}/sor/assemble"
        self.PRICE_URL = f"{self.BASE_URL}/pricing/token"
        
        # Initialize Web3
        self.rpc_url = os.getenv('POLYGON_RPC_URL')
        if not self.rpc_url:
            raise ValueError("POLYGON_RPC_URL not set in environment variables")
            
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        self.w3.middleware_onion.inject(PythonicMiddleware, layer=0)
        
        if os.getenv("TRADING_WALLET_PRIVATE_KEY") is not None:
            private_key = os.getenv("TRADING_WALLET_PRIVATE_KEY")
    
            if private_key:
                # Strip any '0x' prefix if present
                private_key = private_key.replace('0x', '')
                
                # Validate private key format
                if not re.match(r'^[0-9a-fA-F]{64}$', private_key):
                    raise ValueError("Invalid private key format")
                
                self.private_key = private_key
                
                # Create account from private key to get address
                account = Account.from_key(private_key)
                self.wallet_address = account.address
                
            else:
                # If no private key, try mnemonic
                mnemonic = os.getenv('TRADING_WALLET_MNEMONIC_KEY')
                ### End Testing FROM har
                # Initialize wallet from mnemonic
                try:
                    # Create wallet from mnemonic
                    self.wallet = BIP44HDWallet(cryptocurrency=EthereumMainnet)
                    
                    # Clean mnemonic (remove any extra spaces and convert commas to spaces)
                    clean_mnemonic = mnemonic.replace(',', ' ').strip()
                    
                    # Import mnemonic to wallet
                    self.wallet.from_mnemonic(
                        mnemonic=clean_mnemonic,
                        language="english",
                        passphrase=None
                    )
                    
                    # Derive Polygon account (same as Ethereum, since Polygon is EVM-compatible)
                    # BIP44 derivation path for Polygon: m/44'/60'/0'/0/0
                    self.wallet.clean_derivation()
                    derivation = BIP44Derivation(
                        cryptocurrency=EthereumMainnet,
                        account=0,
                        change=False,
                        address=0
                    )
                    self.wallet.from_path(path=derivation)
                    
                    # Get private key and address
                    self.private_key = self.wallet.private_key()
                    self.wallet_address = self.wallet.address()
                    
                    # Verify the derived address matches the environment variable
                    env_wallet_address = os.getenv('POLYGON_TRADING_WALLET')
                    if env_wallet_address and self.wallet_address.lower() != env_wallet_address.lower():
                        self.logger.warning(
                            f"Derived wallet address ({self.wallet_address}) does not match "
                            f"environment variable ({env_wallet_address})"
                        )
                        
                except Exception as e:
                    raise ValueError(f"Error deriving wallet from mnemonic: {str(e)}")
        
        # Initialize network settings
        self.chain_id = 137  # Polygon
        
        # Get Odos router address
        self.router_address = self.get_odos_router_address()
        self.logger.info(f"Using Odos router address: {self.router_address}")
        
        self.logger.info(
            f"Initialized OdosTrader for wallet: {self.wallet_address}\n"
            f"Derived from mnemonic: {self.wallet_address}"
        )
        
        # Token info cache
        self.token_decimals_cache = {}
        # Add after existing initialization
        self.tokens = {
            "USDC": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
            "WETH": "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619",
            "USDT": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
            "WBTC": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6"
        }
        
        # Initialize BalanceManager
        self.balance_manager = BalanceManager(self, self.tokens)
        
        # Add rebalancing configuration
        self.rebalance_config = {
            'min_interval': 14400,  # 4 hours
            'deviation_threshold': 0.15,  # 15%
            'max_gas_gwei': 50,
            'min_liquidity': 1000
        }
    
    async def show_swap_menu(self):
        """Enhanced swap menu with dedicated MATIC options"""
        while True:
            print("\nSwap Menu:")
            print("1. Regular Token Swap")
            print("2. Buy MATIC (Quick Options)")
            print("3. Unwrap Existing WMATIC")
            print("4. Back to Main Menu")
            
            choice = input("\nSelect option: ")
            
            try:
                if choice == "1":
                    await self.regular_swap()
                elif choice == "2":
                    await self.buy_matic_menu()
                elif choice == "3":
                    await self.show_unwrap_menu()
                elif choice == "4":
                    return
            except Exception as e:
                print(f"Error: {e}")
    async def buy_matic_menu(self):
        """Dedicated menu for buying MATIC"""
        try:
            # Get current MATIC status
            gas_status = await self.check_and_add_gas()
            current_matic = gas_status['current_balance_matic']
            current_gas_price = gas_status['gas_price_gwei']
            
            print("\nCurrent MATIC Status:")
            print(f"Balance: {current_matic:.6f} MATIC (${gas_status['current_balance_usd']:.2f})")
            print(f"Gas Price: {current_gas_price:.1f} gwei")
            
            # Show available tokens for MATIC purchase
            print("\nSelect token to buy MATIC with:")
            available_tokens = {
                "1": {"symbol": "USDC", "address": self.tokens["USDC"]},
                "2": {"symbol": "USDT", "address": self.tokens["USDT"]},
                "3": {"symbol": "WETH", "address": self.tokens["WETH"]},
                "4": {"symbol": "WBTC", "address": self.tokens["WBTC"]},
            }
            
            for key, token in available_tokens.items():
                token_info = await self.get_token_info(token["address"])
                token_price = await self.get_token_price(token["address"])
                usd_value = float(token_info['balance']) * (token_price or 0)
                print(f"{key}. {token['symbol']}: {token_info['balance']:.6f} (${usd_value:.2f})")
            
            print("5. Back to Swap Menu")
            
            token_choice = input("\nSelect token number: ")
            
            if token_choice == "5" or token_choice not in available_tokens:
                return
                
            selected_token = available_tokens[token_choice]
            token_info = await self.get_token_info(selected_token["address"])
            
            # Get MATIC price in selected token's terms
            matic_price = await self.get_token_price(self.WMATIC_ADDRESS)
            source_price = await self.get_token_price(selected_token["address"])
            
            if matic_price and source_price:
                matic_token_price = matic_price / source_price
            else:
                print("Error getting price information")
                return
                
            # Calculate suggested amounts
            gas_needed = gas_status['estimated_gas_needed_matic']
            suggested_amounts = [
                gas_needed * 2,  # Minimum suggested (2x estimated need)
                gas_needed * 4,  # Medium amount
                gas_needed * 8   # Large amount
            ]
            
            print("\nSuggested MATIC purchase amounts:")
            for i, amount in enumerate(suggested_amounts, 1):
                token_cost = amount * matic_token_price
                usd_cost = amount * matic_price
                print(f"{i}. {amount:.4f} MATIC (Cost: {token_cost:.6f} {selected_token['symbol']}, ${usd_cost:.2f})")
            
            print("4. Custom amount")
            print("5. Back")
            
            amount_choice = input("\nSelect amount option: ")
            
            if amount_choice == "5":
                return
            
            if amount_choice == "4":
                custom_amount = float(input(f"\nEnter MATIC amount to buy: "))
                matic_amount = custom_amount
            elif amount_choice in ["1", "2", "3"]:
                matic_amount = suggested_amounts[int(amount_choice) - 1]
            else:
                print("Invalid choice")
                return
                
            # Calculate token amount needed
            token_amount = matic_amount * matic_token_price
            
            if token_amount > token_info['balance']:
                print(f"\nInsufficient {selected_token['symbol']} balance")
                print(f"Required: {token_amount:.6f}")
                print(f"Available: {token_info['balance']:.6f}")
                return
                
            # Confirm purchase
            print(f"\nPurchase Summary:")
            print(f"Buy: {matic_amount:.6f} MATIC")
            print(f"Cost: {token_amount:.6f} {selected_token['symbol']}")
            print(f"USD Value: ${(matic_amount * matic_price):.2f}")
            
            confirm = input("\nConfirm purchase? (y/n): ")
            
            if confirm.lower() == 'y':
                print("\nExecuting MATIC purchase...")
                try:
                    result = await self.buy_matic(
                        input_token=selected_token["address"],
                        matic_amount=Decimal(str(token_amount)),
                        input_symbol=selected_token["symbol"]
                    )
                    
                    if result['status'] == 'success':
                        print(f"\nSuccessfully purchased and unwrapped MATIC")
                        print(f"Buy Transaction: {result['buy_tx']}")
                        print(f"Unwrap Transaction: {result['unwrap_tx']}")
                        print(f"\nTotal Gas Cost: ${result['total_gas_cost_usd']:.4f}")
                        print(f"New MATIC Balance: {result['new_matic_balance']:.6f}")
                    else:
                        print("\nFailed to complete MATIC purchase")
                        
                except Exception as e:
                    print(f"Error executing MATIC purchase: {e}")
                    self.logger.error(f"MATIC purchase error: {e}")
            
        except Exception as e:
            print(f"Error in MATIC purchase menu: {e}")
            self.logger.error(f"MATIC menu error: {e}")
    async def regular_swap(self):
        """Regular token swap functionality"""
        try:
            # Show all token balances
            print("\nCurrent Balances:")
            balances = {}
            for symbol, address in self.tokens.items():
                token_info = await self.get_token_info(address)
                balances[symbol] = token_info
                print(f"{symbol}: {token_info['balance']:.6f} {token_info['symbol']}")

            # Let user select input and output tokens
            print("\nSelect input token:")
            for i, symbol in enumerate(self.tokens.keys(), 1):
                print(f"{i}. {symbol}")
            input_choice = int(input("Enter number: ")) - 1
            input_symbol = list(self.tokens.keys())[input_choice]
            input_address = self.tokens[input_symbol]

            print("\nSelect output token:")
            available_outputs = [s for s in self.tokens.keys() if s != input_symbol]
            for i, symbol in enumerate(available_outputs, 1):
                print(f"{i}. {symbol}")
            output_choice = int(input("Enter number: ")) - 1
            output_symbol = available_outputs[output_choice]
            output_address = self.tokens[output_symbol]

            # Get trade amount
            max_amount = float(balances[input_symbol]['balance'])
            amount = float(input(f"\nEnter amount to trade (max {max_amount:.6f} {input_symbol}): "))
            
            if amount > max_amount:
                print(f"Amount exceeds available balance of {max_amount:.6f} {input_symbol}")
                return

            # Execute trade
            result = await self.execute_trade(
                input_token=input_address,
                output_token=output_address,
                amount_in_human=Decimal(str(amount)),
                slippage=0.5
            )
            
            # Show results
            print("\nTrade Result:")
            print(f"Status: {result['status']}")
            print(f"Transaction Hash: {result['transaction_hash']}")
            print(f"Gas Used: {result['gas_used']}")
            print(f"Gas Cost (USD): ${result['gas_cost_usd']:.4f}")
            print(f"Price Impact: {result['price_impact']}%")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
    async def unwrap_wmatic(self, amount: Decimal) -> Dict:
        """Unwrap WMATIC to native MATIC with improved gas handling"""
        try:
            wmatic_abi = [
                {
                    "constant": False,
                    "inputs": [{"name": "wad", "type": "uint256"}],
                    "name": "withdraw",
                    "outputs": [],
                    "payable": False,
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [{"name": "owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function"
                }
            ]
            
            wmatic_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.WMATIC_ADDRESS),
                abi=wmatic_abi
            )
            
            # Verify WMATIC balance first
            wmatic_balance = wmatic_contract.functions.balanceOf(self.wallet_address).call()
            amount_wei = int(amount * Decimal('1e18'))
            
            if wmatic_balance < amount_wei:
                raise ValueError(f"Insufficient WMATIC balance. Have: {wmatic_balance/1e18}, Need: {amount}")
            
            # Get current gas price and increase it slightly for faster confirmation
            base_gas_price = self.w3.eth.gas_price
            gas_price = int(base_gas_price * 1.1)  # 10% higher than base
            
            # Build transaction with higher gas price
            transaction = wmatic_contract.functions.withdraw(amount_wei).build_transaction({
                'from': self.wallet_address,
                'gas': 100000,  # Increased gas limit
                'gasPrice': gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address),
                'chainId': self.chain_id
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt with longer timeout
            receipt = self.w3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=180,  # 3 minutes timeout
                poll_latency=2  # Check every 2 seconds
            )
            
            if receipt['status'] != 1:
                raise Exception("Unwrap transaction failed")
            
            # Verify the unwrap was successful by checking MATIC balance change
            matic_balance_after = self.w3.eth.get_balance(self.wallet_address)
            gas_cost_usd = await self.calculate_gas_cost_usd(receipt['gasUsed'])
            
            result = {
                'status': 'success',
                'transaction_hash': receipt['transactionHash'].hex(),
                'gas_used': receipt['gasUsed'],
                'gas_cost_usd': gas_cost_usd,
                'amount_unwrapped': float(amount),
                'new_matic_balance': matic_balance_after / 1e18
            }
            
            self.logger.info(
                f"Successfully unwrapped {amount} WMATIC to MATIC\n"
                f"Gas used: {receipt['gasUsed']}\n"
                f"Gas cost: ${gas_cost_usd:.4f}"
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Error unwrapping WMATIC: {e}")
            raise
    async def buy_matic(self, input_token: str, matic_amount: Decimal, input_symbol: str) -> Dict:
        """Buy and unwrap MATIC with better error handling"""
        try:
            # First buy WMATIC
            print("\nStep 1: Buying WMATIC...")
            trade_result = await self.execute_trade(
                input_token=input_token,
                output_token=self.WMATIC_ADDRESS,
                amount_in_human=matic_amount,
                slippage=1.0
            )
            
            if trade_result['status'] != 'success':
                raise Exception("Failed to buy WMATIC")
            
            # Get actual WMATIC received
            wmatic_received = Decimal(trade_result['received_amount'])
            print(f"\nReceived {wmatic_received:.6f} WMATIC")
            
            # Wait a bit for the WMATIC balance to be updated
            print("Waiting for balance update...")
            await asyncio.sleep(5)
            
            # Unwrap WMATIC to MATIC
            print("\nStep 2: Unwrapping WMATIC to MATIC...")
            unwrap_result = await self.unwrap_wmatic(wmatic_received)
            
            if unwrap_result['status'] != 'success':
                raise Exception("Failed to unwrap WMATIC")
            
            # Calculate total gas cost
            total_gas_cost = trade_result['gas_cost_usd'] + unwrap_result['gas_cost_usd']
            
            result = {
                'status': 'success',
                'buy_tx': trade_result['transaction_hash'],
                'unwrap_tx': unwrap_result['transaction_hash'],
                'amount_bought': float(wmatic_received),
                'total_gas_cost_usd': total_gas_cost,
                'new_matic_balance': unwrap_result['new_matic_balance']
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in MATIC purchase process: {e}")
            
            # Try to provide more context about failure
            if 'trade_result' in locals():
                print("\nWMATIC purchase was successful but unwrapping failed.")
                print(f"WMATIC transaction: {trade_result['transaction_hash']}")
                print("\nYou can try unwrapping the WMATIC later using the unwrap function.")
            
            raise
    async def check_and_add_gas(self) -> Dict:
        """Check MATIC balance and estimate if more is needed"""
        try:
            # Get current MATIC balance - Web3.py functions are synchronous
            balance_wei = self.w3.eth.get_balance(self.wallet_address)
            balance_matic = balance_wei / 1e18
            
            # Get MATIC price - this is already async
            matic_price = await self.get_matic_price()
            balance_usd = balance_matic * matic_price if matic_price else 0
            
            # Get current gas price - synchronous
            current_gas_price = self.w3.eth.gas_price / 1e9  # Convert to gwei
            
            # Estimate gas needed for common operations
            gas_limit = 150000  # Standard gas limit for trades
            operations_count = 10
            
            estimated_gas_cost_matic = (gas_limit * current_gas_price * operations_count) / 1e9
            estimated_gas_cost_usd = estimated_gas_cost_matic * matic_price if matic_price else 0
            
            self.logger.info(
                f"\nGas Status:"
                f"\nMATIC Balance: {balance_matic:.6f} (${balance_usd:.2f})"
                f"\nGas Price: {current_gas_price:.1f} gwei"
                f"\nEstimated need: {estimated_gas_cost_matic:.6f} MATIC"
            )
            
            result = {
                'current_balance_matic': balance_matic,
                'current_balance_usd': balance_usd,
                'estimated_gas_needed_matic': estimated_gas_cost_matic,
                'estimated_gas_needed_usd': estimated_gas_cost_usd,
                'gas_price_gwei': current_gas_price,
                'matic_price': matic_price,
                'needs_more_gas': balance_matic < estimated_gas_cost_matic * 1.5  # 50% buffer
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking gas balance: {e}")
            raise
    async def show_gas_menu(self):
        """Menu for managing gas (MATIC) balance"""
        try:
            gas_status = await self.check_and_add_gas()
            
            print("\nGas (MATIC) Status:")
            print(f"Current Balance: {gas_status['current_balance_matic']:.6f} MATIC (${gas_status['current_balance_usd']:.2f})")
            print(f"Current Gas Price: {gas_status['gas_price_gwei']:.1f} gwei")
            print(f"Estimated needed for 10 operations: {gas_status['estimated_gas_needed_matic']:.6f} MATIC")
            print(f"Estimated cost: ${gas_status['estimated_gas_needed_usd']:.2f}")
            
            if gas_status['needs_more_gas']:
                print("\nWARNING: Current gas balance may be insufficient")
                print("\nOptions:")
                print("1. Buy MATIC with USDC")
                print("2. Back to menu")
                
                choice = input("\nSelect option: ")
                
                if choice == "1":
                    # Get USDC balance
                    usdc_info = await self.get_token_info(self.tokens["USDC"])
                    
                    # Calculate how much MATIC to buy
                    suggested_matic = gas_status['estimated_gas_needed_matic'] * 2  # 100% buffer
                    suggested_usdc = suggested_matic * gas_status['matic_price']
                    
                    print(f"\nSuggested MATIC purchase: {suggested_matic:.6f} MATIC (${suggested_usdc:.2f})")
                    print(f"Available USDC: ${float(usdc_info['balance']):.2f}")
                    
                    if usdc_info['balance'] < suggested_usdc:
                        print(f"Warning: Insufficient USDC balance.")
                        max_possible = float(usdc_info['balance'])
                        if max_possible > 0:
                            print(f"Maximum possible purchase: ${max_possible:.2f}")
                            proceed = input("Would you like to proceed with maximum possible amount? (y/n): ")
                            if proceed.lower() == 'y':
                                suggested_usdc = max_possible
                            else:
                                return
                        else:
                            print("No USDC available for MATIC purchase")
                            return
                    
                    confirm = input(f"Proceed with {suggested_usdc:.2f} USDC to MATIC purchase? (y/n): ")
                    if confirm.lower() == 'y':
                        try:
                            print("\nExecuting MATIC purchase...")
                            # Execute trade from USDC to MATIC
                            result = await self.execute_trade(
                                input_token=self.tokens["USDC"],
                                output_token=self.WMATIC_ADDRESS,
                                amount_in_human=Decimal(str(suggested_usdc)),
                                slippage=1.0
                            )
                            
                            if result['status'] == 'success':
                                print(f"\nSuccessfully purchased MATIC")
                                print(f"Transaction hash: {result['transaction_hash']}")
                                
                                # Show updated balance
                                updated_status = await self.check_and_add_gas()
                                print(f"\nUpdated MATIC balance: {updated_status['current_balance_matic']:.6f} MATIC")
                            else:
                                print("\nFailed to purchase MATIC")
                        except Exception as e:
                            print(f"Error executing MATIC purchase: {e}")
            else:
                print("\nGas balance is sufficient for operations")
                input("\nPress Enter to continue...")
                
        except Exception as e:
            print(f"Error in gas menu: {e}")
            self.logger.error(f"Gas menu error: {e}")
    def get_odos_router_address(self) -> str:
        """Get the Odos router address for the current chain"""
        try:
            # Get router address from Odos API
            url = f"{self.BASE_URL}/info/router/v2/{self.chain_id}"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            router_address = data.get('address')
            
            if not router_address:
                raise ValueError("Router address not found in response")
                
            return Web3.to_checksum_address(router_address)
            
        except Exception as e:
            self.logger.error(f"Error getting router address: {str(e)}")
            # Fallback to known router address for Polygon
            return Web3.to_checksum_address("0xa32ee1c40594249eb3183c10792b4648758a7e7b")
    async def show_unwrap_menu(self):
        """Menu for unwrapping existing WMATIC"""
        try:
            # Get WMATIC balance
            wmatic_info = await self.get_token_info(self.WMATIC_ADDRESS)
            wmatic_balance = wmatic_info['balance']
            
            print(f"\nCurrent WMATIC Balance: {wmatic_balance:.6f}")
            
            if wmatic_balance <= 0:
                print("No WMATIC balance to unwrap")
                return
                
            amount = float(input(f"Enter amount to unwrap (max {wmatic_balance:.6f}): "))
            if amount > float(wmatic_balance):
                print("Amount exceeds balance")
                return
                
            print(f"\nUnwrapping {amount:.6f} WMATIC to MATIC...")
            result = await self.unwrap_wmatic(Decimal(str(amount)))
            
            if result['status'] == 'success':
                print(f"\nSuccessfully unwrapped WMATIC to MATIC")
                print(f"Transaction: {result['transaction_hash']}")
                print(f"Gas Used: {result['gas_used']}")
                print(f"Gas Cost: ${result['gas_cost_usd']:.4f}")
                print(f"New MATIC Balance: {result['new_matic_balance']:.6f}")
            
        except Exception as e:
            print(f"Error unwrapping WMATIC: {e}")
    async def get_token_info(self, token_address: str) -> Dict:
        """Get token decimals and balance"""
        # Basic ERC20 ABI
        erc20_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [
                    {"name": "_owner", "type": "address"},
                    {"name": "_spender", "type": "address"}
                ],
                "name": "allowance",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": False,
                "inputs": [
                    {"name": "_spender", "type": "address"},
                    {"name": "_value", "type": "uint256"}
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            }
        ]

        token_address = Web3.to_checksum_address(token_address)
        
        if token_address in self.token_decimals_cache:
            decimals = self.token_decimals_cache[token_address]
        else:
            contract = self.w3.eth.contract(address=token_address, abi=erc20_abi)
            decimals = contract.functions.decimals().call()
            self.token_decimals_cache[token_address] = decimals
            
        # Get token info
        contract = self.w3.eth.contract(address=token_address, abi=erc20_abi)
        
        try:
            symbol = contract.functions.symbol().call()
        except:
            symbol = token_address[:8]
            
        balance_wei = contract.functions.balanceOf(self.wallet_address).call()
        balance = Decimal(balance_wei) / Decimal(10 ** decimals)
        
        # Check allowance
        allowance = contract.functions.allowance(
            self.wallet_address,
            self.router_address
        ).call()
        
        return {
            'decimals': decimals,
            'balance': balance,
            'balance_wei': balance_wei,
            'symbol': symbol,
            'allowance': allowance,
            'contract': contract  # Return contract for reuse
        }
    async def check_token_approval_status(self, token_address: str) -> Dict:
        """Check token approval status"""
        try:
            token_info = await self.get_token_info(token_address)
            
            return {
                'token': token_info['symbol'],
                'allowance': token_info['allowance'],
                'has_approval': token_info['allowance'] > 0,
                'max_approval': token_info['allowance'] >= (2**256 - 1),
            }
            
        except Exception as e:
            self.logger.error(f"Error checking approval status: {str(e)}")
            raise
    async def revoke_token_approval(self, token_address: str) -> str:
        """Revoke token approval by setting it to 0"""
        try:
            return await self.approve_token(token_address, 0)
        except Exception as e:
            self.logger.error(f"Error revoking approval: {str(e)}")
            raise
    async def approve_token(self, token_address: str, amount: int = None) -> str:
        """Approve token spending"""
        try:
            token_info = await self.get_token_info(token_address)
            contract = token_info['contract']
            
            # If no amount specified, use max uint256
            if amount is None:
                amount = 2**256 - 1
            
            # Build approval transaction
            approve_txn = contract.functions.approve(
                self.router_address,
                amount
            ).build_transaction({
                'from': self.wallet_address,
                'gas': 100000,  # Standard gas limit for approvals
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address),
                'chainId': self.chain_id
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(approve_txn, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt['status'] == 1:
                self.logger.info(f"Approved {token_info['symbol']} for trading")
                return receipt['transactionHash'].hex()
            else:
                raise Exception("Approval transaction failed")
                
        except Exception as e:
            self.logger.error(f"Error approving token: {str(e)}")
            raise
    async def check_current_prices():
        try:
            # Addresses
            wmatic_address = "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270"  # WMATIC
            usdc_address = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"   # USDC

            # Get MATIC price in USDC
            matic_price = await trader.get_token_price(wmatic_address)
            
            if matic_price:
                print(f"\nCurrent MATIC Price: ${matic_price:.4f}")
                return matic_price
            else:
                print("Error getting MATIC price")
                return None

        except Exception as e:
            print(f"Error checking prices: {str(e)}")
            return None
    async def check_and_approve_token(self, token_address: str, amount_in_wei: int) -> None:
        """Check if token needs approval and approve if necessary"""
        token_info = await self.get_token_info(token_address)
        
        if token_info['allowance'] < amount_in_wei:
            self.logger.info(f"Insufficient allowance for {token_info['symbol']}, approving...")
            await self.approve_token(token_address)
            # Wait a bit for approval to be processed
            time.sleep(2)
    async def get_matic_price(self) -> Optional[float]:
            """Get current MATIC price with caching"""
            try:
                current_time = time.time()
                
                # Return cached price if still valid
                if (self.matic_price_cache['price'] is not None and 
                    current_time - self.matic_price_cache['timestamp'] < self.PRICE_CACHE_DURATION):
                    return self.matic_price_cache['price']
                
                # Get fresh MATIC price
                price = await self.get_token_price(self.WMATIC_ADDRESS)
                if price:
                    self.matic_price_cache = {
                        'price': price,
                        'timestamp': current_time
                    }
                    return price
                
                return None
                
            except Exception as e:
                self.logger.error(f"Error getting MATIC price: {str(e)}")
                return None
    async def get_token_price(self, token_address: str, currency_id: str = "USD") -> Optional[float]:
        """Get token price from Odos pricing API"""
        try:
            url = f"{self.PRICE_URL}/{self.chain_id}/{token_address}"
            params = {"currencyId": currency_id} if currency_id != "USD" else {}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("price")
            
        except Exception as e:
            self.logger.error(f"Error getting price for {token_address}: {str(e)}")
            return None
    async def calculate_gas_cost_usd(self, gas_amount: int) -> float:
        """Calculate gas cost in USD using current MATIC price"""
        try:
            matic_price = await self.get_matic_price()
            if not matic_price:
                raise ValueError("Could not get MATIC price for gas calculation")
                
            gas_price_wei = self.w3.eth.gas_price
            gas_cost_matic = (gas_amount * gas_price_wei) / 1e18
            gas_cost_usd = gas_cost_matic * matic_price
            
            return gas_cost_usd
            
        except Exception as e:
            self.logger.error(f"Error calculating gas cost: {str(e)}")
            raise
    async def get_uniswap_quote(self, token_in: str, token_out: str, amount_in_wei: int, decimals_in: int, decimals_out: int) -> Dict:
        """Get quote from Uniswap V3"""
        try:
            # Uniswap V3 Quoter address on Polygon
            quoter_address = "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6"
            
            # Convert addresses to checksum format
            token_in = Web3.to_checksum_address(token_in)
            token_out = Web3.to_checksum_address(token_out)
            quoter_address = Web3.to_checksum_address(quoter_address)
            
            # Uniswap V3 Quoter ABI
            quoter_abi = [
                {
                    "inputs": [
                        {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                        {"internalType": "address[]", "name": "path", "type": "address[]"}
                    ],
                    "name": "getAmountsOut",
                    "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
            
            # Create contract instance
            quoter = self.w3.eth.contract(address=quoter_address, abi=quoter_abi)
            
            # Get quote
            amounts = quoter.functions.getAmountsOut(
                amount_in_wei,
                [token_in, token_out]
            ).call()
            
            # Get price
            amount_in_decimal = Decimal(amount_in_wei) / Decimal(10 ** decimals_in)
            amount_out_decimal = Decimal(amounts[1]) / Decimal(10 ** decimals_out)
            price = amount_out_decimal / amount_in_decimal if amount_in_decimal else 0
            
            return {
                'price': float(price),
                'output_amount': amounts[1],
                'gas_estimate': 150000  # Estimated gas for Uniswap swap
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Uniswap quote: {str(e)}")
            return None
    async def get_sushiswap_quote(self, token_in: str, token_out: str, amount_in_wei: int, decimals_in: int, decimals_out: int) -> Dict:
        """Get quote from SushiSwap with improved error handling"""
        try:
            router_address = Web3.to_checksum_address("0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506")
            token_in = Web3.to_checksum_address(token_in)
            token_out = Web3.to_checksum_address(token_out)
            
            router = self.w3.eth.contract(address=router_address, abi=[{
                "inputs": [
                    {"type": "uint256", "name": "amountIn"},
                    {"type": "address[]", "name": "path"}
                ],
                "name": "getAmountsOut",
                "outputs": [{"type": "uint256[]", "name": "amounts"}],
                "stateMutability": "view",
                "type": "function"
            }])
            
            amounts = router.functions.getAmountsOut(
                amount_in_wei,
                [token_in, token_out]
            ).call()
            
            amount_in_decimal = Decimal(str(amount_in_wei)) / Decimal(str(10 ** decimals_in))
            amount_out_decimal = Decimal(str(amounts[1])) / Decimal(str(10 ** decimals_out))
            price = amount_out_decimal / amount_in_decimal if amount_in_decimal else 0
            
            return {
                'price': float(price),
                'output_amount': amounts[1],
                'gas_estimate': 130000,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.debug(f"SushiSwap quote error: {str(e)}")
            return None
    async def get_quickswap_quote(self, token_in: str, token_out: str, amount_in_wei: int, decimals_in: int, decimals_out: int) -> Dict:
        """Get quote from QuickSwap"""
        try:
            # QuickSwap Router address on Polygon
            router_address = "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff"
            
            # Convert addresses to checksum format
            token_in = Web3.to_checksum_address(token_in)
            token_out = Web3.to_checksum_address(token_out)
            router_address = Web3.to_checksum_address(router_address)
            
            # Router ABI
            router_abi = [
                {
                    "inputs": [
                        {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                        {"internalType": "address[]", "name": "path", "type": "address[]"}
                    ],
                    "name": "getAmountsOut",
                    "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
            
            # Create contract instance
            router = self.w3.eth.contract(address=router_address, abi=router_abi)
            
            # Get quote
            amounts = router.functions.getAmountsOut(
                amount_in_wei,
                [token_in, token_out]
            ).call()
            
            # Get price
            amount_in_decimal = Decimal(amount_in_wei) / Decimal(10 ** decimals_in)
            amount_out_decimal = Decimal(amounts[1]) / Decimal(10 ** decimals_out)
            price = amount_out_decimal / amount_in_decimal if amount_in_decimal else 0
            
            return {
                'price': float(price),
                'output_amount': amounts[1],
                'gas_estimate': 130000  # Estimated gas for QuickSwap swap
            }
            
        except Exception as e:
            self.logger.error(f"Error getting QuickSwap quote: {str(e)}")
            return None
    async def execute_trade_on_dex(self, token_in: str, token_out: str, amount_in: Decimal, dex: str) -> Dict:
        """Execute trade on a specific DEX"""
        # This method needs to be implemented with the actual trade execution logic
        # for each DEX. For now, it raises a NotImplementedError
        raise NotImplementedError("DEX-specific trade execution not implemented yet")
    async def get_dex_quote(
        self,
        dex: str,
        token_in: str,
        token_out: str,
        amount_in_wei: int,
        decimals_in: int,
        decimals_out: int
    ) -> Optional[Dict]:
        """Wrapper for getting DEX quotes with proper error handling"""
        try:
            if dex == "uniswap":
                # Skip Uniswap for now as it's having issues
                return None
            
            quote_func = getattr(self, f'get_{dex}_quote', None)
            if quote_func:
                return await quote_func(
                    token_in,
                    token_out,
                    amount_in_wei,
                    decimals_in,
                    decimals_out
                )
            return None
        except Exception as e:
            self.logger.debug(f"Error getting {dex} quote: {str(e)}")  # Reduced to debug level
            return None
    def calculate_potential_profit(trade_amount: float, spread: float = 0.001, gas_cost: float = 0.5):
        """
        Calculate potential profit for different trade sizes
        
        Args:
            trade_amount: Amount in USDC
            spread: Price difference between DEXs (0.1% = 0.001)
            gas_cost: Fixed gas cost in USD
        """
        gross_profit = trade_amount * spread
        net_profit = gross_profit - gas_cost
        return {
            'trade_amount': trade_amount,
            'gross_profit': gross_profit,
            'gas_cost': gas_cost,
            'net_profit': net_profit,
            'profitable': net_profit > 0
        }
    async def check_arbitrage_opportunity(
        self,
        token_in: str,
        token_out: str,
        amount_in_human: Decimal,
        dexes: List[str] = ['sushiswap', 'quickswap']
    ) -> Dict:
        """Check for arbitrage opportunities between DEXes with accurate gas costs"""
        try:
            input_token_info = await self.get_token_info(token_in)
            output_token_info = await self.get_token_info(token_out)
            
            amount_in_wei = int(amount_in_human * (10 ** input_token_info['decimals']))
            
            dex_quotes = {}
            opportunities = []
            
            # Get current MATIC price once for all calculations
            matic_price = await self.get_matic_price()
            if not matic_price:
                raise ValueError("Could not get MATIC price for calculations")

            self.logger.info(f"\nChecking arbitrage opportunities:")
            self.logger.info(f"Trading Pair: {input_token_info['symbol']}-{output_token_info['symbol']}")
            self.logger.info(f"Amount: {float(amount_in_human):.2f} {input_token_info['symbol']}")
            self.logger.info(f"Current MATIC Price: ${matic_price:.4f}")
            
            # Get quotes from each DEX
            for dex in dexes:
                try:
                    quote = await getattr(self, f'get_{dex}_quote')(
                        token_in,
                        token_out,
                        amount_in_wei,
                        input_token_info['decimals'],
                        output_token_info['decimals']
                    )
                    if quote:
                        dex_quotes[dex] = quote
                        self.logger.info(
                            f"\n{dex.upper()} Quote:"
                            f"\nOutput: {quote['output_amount'] / (10 ** output_token_info['decimals']):.6f} {output_token_info['symbol']}"
                            f"\nRate: 1 {input_token_info['symbol']} = {quote['price']:.6f} {output_token_info['symbol']}"
                        )
                except Exception as e:
                    self.logger.debug(f"Error getting {dex} quote: {str(e)}")
                    continue

            # Compare prices between DEXes
            if len(dex_quotes) >= 2:
                self.logger.info("\nComparing DEX prices:")
                
                for buy_dex, buy_quote in dex_quotes.items():
                    for sell_dex, sell_quote in dex_quotes.items():
                        if sell_dex != buy_dex:
                            try:
                                # Calculate potential profit
                                buy_price = buy_quote['price']
                                sell_price = sell_quote['price']
                                price_diff = ((sell_price - buy_price) / buy_price) * 100
                                
                                # Calculate gas costs
                                total_gas = buy_quote['gas_estimate'] + sell_quote['gas_estimate']
                                gas_cost_matic = (total_gas * self.w3.eth.gas_price) / 1e18
                                gas_cost_usd = gas_cost_matic * matic_price
                                
                                # Calculate profits
                                trade_amount_in_usd = float(amount_in_human)
                                estimated_profit_usd = (price_diff/100) * trade_amount_in_usd
                                net_profit_usd = estimated_profit_usd - gas_cost_usd
                                
                                self.logger.info(
                                    f"\nArbitrage Path: {buy_dex.upper()} → {sell_dex.upper()}"
                                    f"\nBuy at:  {buy_price:.6f} {output_token_info['symbol']}/{input_token_info['symbol']}"
                                    f"\nSell at: {sell_price:.6f} {output_token_info['symbol']}/{input_token_info['symbol']}"
                                    f"\nPrice difference: {price_diff:.2f}%"
                                    f"\nEstimated profit: ${estimated_profit_usd:.4f}"
                                    f"\nGas cost: {gas_cost_matic:.6f} MATIC (${gas_cost_usd:.4f})"
                                    f"\nNet profit: ${net_profit_usd:.4f}"
                                )

                                opportunities.append({
                                    'buy_dex': buy_dex,
                                    'sell_dex': sell_dex,
                                    'profit_percent': price_diff,
                                    'estimated_profit_usd': estimated_profit_usd,
                                    'gas_cost_matic': gas_cost_matic,
                                    'gas_cost_usd': gas_cost_usd,
                                    'net_profit_usd': net_profit_usd,
                                    'gas_price_gwei': self.w3.eth.gas_price / 1e9,
                                    'matic_price_usd': matic_price,
                                    'buy_price': buy_price,
                                    'sell_price': sell_price
                                })
                                
                            except Exception as e:
                                self.logger.debug(f"Error calculating profit for {buy_dex}->{sell_dex}: {str(e)}")
                                continue

                # Sort opportunities by net profit
                opportunities.sort(key=lambda x: x['net_profit_usd'], reverse=True)
                
                if opportunities:
                    best_opp = opportunities[0]
                    self.logger.info(
                        f"\nBest Opportunity:"
                        f"\nPath: {best_opp['buy_dex'].upper()} → {best_opp['sell_dex'].upper()}"
                        f"\nPrice difference: {best_opp['profit_percent']:.2f}%"
                        f"\nNet profit: ${best_opp['net_profit_usd']:.4f}"
                    )
                else:
                    self.logger.info("\nNo profitable opportunities found")

            return {
                'token_in': input_token_info['symbol'],
                'token_out': output_token_info['symbol'],
                'amount': amount_in_human,
                'opportunities': opportunities,
                'dex_quotes': dex_quotes,
                'matic_price': matic_price
            }

        except Exception as e:
            self.logger.error(f"Error checking arbitrage opportunities: {str(e)}")
            raise   
    async def safe_check_opportunity(
        self,
        input_token_info: Dict,
        output_token_info: Dict,
        forward_quote: Dict,
        reverse_quote: Dict,
        trade_amount: Decimal
    ) -> Optional[Dict]:
        """Safely check if an opportunity is profitable"""
        try:
            # Calculate amounts
            out_amount = Decimal(forward_quote["outAmounts"][0])
            out_amount_human = out_amount / Decimal(str(10**output_token_info['decimals']))
            
            # Verify we have enough balance for the reverse trade
            if out_amount_human > output_token_info['balance']:
                return None
                
            final_amount = Decimal(reverse_quote["outAmounts"][0]) / Decimal(str(10**input_token_info['decimals']))
            profit_amount = final_amount - trade_amount
            
            # Get token price
            token_price = await self.get_token_price(input_token_info['address'])
            if not token_price:
                return None
                
            # Calculate profits
            profit_usd = float(profit_amount * Decimal(str(token_price)))
            gas_cost_usd = float(forward_quote.get("gasEstimateValue", 0) + 
                            reverse_quote.get("gasEstimateValue", 0))
            net_profit_usd = profit_usd - gas_cost_usd
            
            return {
                'profit_amount': profit_amount,
                'profit_usd': profit_usd,
                'gas_cost_usd': gas_cost_usd,
                'net_profit_usd': net_profit_usd,
                'final_amount': final_amount
            }
            
        except Exception as e:
            self.logger.debug(f"Error checking opportunity: {str(e)}")
            return None
    async def execute_arbitrage(
        self,
        token_in: str,
        token_out: str,
        amount_in_human: Decimal,
        buy_dex: str,
        sell_dex: str,
        min_profit_usd: float = 5.0  # Minimum profit threshold in USD
    ) -> Dict:
        """Execute an arbitrage trade across two DEXes"""
        try:
            # First check if the opportunity is profitable
            arb_check = await self.check_arbitrage_opportunity(
                token_in,
                token_out,
                amount_in_human,
                dexes=[buy_dex, sell_dex]
            )

            best_opportunity = next(
                (opp for opp in arb_check['opportunities'] 
                if opp['buy_dex'] == buy_dex and opp['sell_dex'] == sell_dex),
                None
            )

            if not best_opportunity:
                raise ValueError("No viable arbitrage opportunity found")

            if best_opportunity['net_profit_usd'] < min_profit_usd:
                raise ValueError(
                    f"Profit ({best_opportunity['net_profit_usd']:.2f} USD) below minimum threshold "
                    f"({min_profit_usd} USD)"
                )

            # Execute the trades
            # 1. Buy on first DEX
            buy_result = await self.execute_trade_on_dex(
                token_in,
                token_out,
                amount_in_human,
                buy_dex
            )

            # Get the received amount
            received_amount = Decimal(buy_result['received_amount'])

            # 2. Sell on second DEX
            sell_result = await self.execute_trade_on_dex(
                token_out,
                token_in,
                received_amount,
                sell_dex
            )

            # Calculate actual profit
            actual_profit = Decimal(sell_result['received_amount']) - amount_in_human
            total_gas_cost = (
                buy_result['gas_cost_usd'] + 
                sell_result['gas_cost_usd']
            )

            trade_result = {
                'status': 'success',
                'buy_transaction': buy_result['transaction_hash'],
                'sell_transaction': sell_result['transaction_hash'],
                'input_amount': str(amount_in_human),
                'output_amount': str(sell_result['received_amount']),
                'profit_amount': str(actual_profit),
                'total_gas_cost_usd': total_gas_cost,
                'net_profit_usd': float(actual_profit) * float(arb_check['dex_quotes'][sell_dex]['price']) - total_gas_cost,
                'buy_dex': buy_dex,
                'sell_dex': sell_dex
            }

            # Send notification
            self.send_arbitrage_notification(trade_result)

            return trade_result

        except Exception as e:
            self.logger.error(f"Error executing arbitrage: {str(e)}")
            raise
    async def balance_portfolio(self, token_pair: Dict, target_ratio: float = 0.5) -> None:
        """
        Balance portfolio to maintain target ratio between token pairs

        Args:
            token_pair: Dictionary containing input and output token addresses
            target_ratio: Target ratio for token_in (0.5 means 50/50 split)
        """
        try:
            # Get token info
            token_in_info = await self.get_token_info(token_pair["in"])
            token_out_info = await self.get_token_info(token_pair["out"])

            # Get current prices
            token_in_price = await self.get_token_price(token_pair["in"])
            token_out_price = await self.get_token_price(token_pair["out"])

            if not all([token_in_price, token_out_price]):
                self.logger.error("Could not get token prices for portfolio balancing")
                return

            # Calculate current values in USD
            token_in_value = token_in_info['balance'] * Decimal(str(token_in_price))
            token_out_value = token_out_info['balance'] * Decimal(str(token_out_price))
            total_value = token_in_value + token_out_value

            if total_value == 0:
                self.logger.info("No assets to balance")
                return

            current_ratio = float(token_in_value / total_value)

            # Check if rebalancing is needed (5% threshold)
            if abs(current_ratio - target_ratio) < 0.05:
                self.logger.info(f"Portfolio already balanced (Current ratio: {current_ratio:.2%})")
                return

            self.logger.info(
                f"Rebalancing portfolio:\n"
                f"Current ratio: {current_ratio:.2%}\n"
                f"Target ratio: {target_ratio:.2%}"
            )

            # Initialize variables for logging
            trade_result = None

            # Store balances before the trade for profit calculation
            token_in_balance_before = token_in_info['balance']
            token_out_balance_before = token_out_info['balance']

            # Calculate required trade
            if current_ratio > target_ratio:
                # Need to sell token_in for token_out
                trade_value = float(token_in_value) - (float(total_value) * target_ratio)
                trade_amount = Decimal(str(trade_value / token_in_price))

                if trade_amount > 0:
                    trade_result = await self.execute_trade(
                        token_pair["in"],
                        token_pair["out"],
                        trade_amount,
                        slippage=0.5
                    )
            else:
                # Need to sell token_out for token_in
                trade_value = float(token_out_value) - (float(total_value) * (1 - target_ratio))
                trade_amount = Decimal(str(trade_value / token_out_price))

                if trade_amount > 0:
                    trade_result = await self.execute_trade(
                        token_pair["out"],
                        token_pair["in"],
                        trade_amount,
                        slippage=0.5
                    )

            self.logger.info("Portfolio rebalancing complete")

            if trade_result and trade_result['status'] == 'success':
                # Get updated token info after the trade
                token_in_info_after = await self.get_token_info(token_pair["in"])
                token_out_info_after = await self.get_token_info(token_pair["out"])

                # Calculate balances after the trade
                token_in_balance_after = token_in_info_after['balance']
                token_out_balance_after = token_out_info_after['balance']

                # Calculate the received amount
                if current_ratio > target_ratio:
                    # Sold token_in for token_out
                    amount_in_symbol = token_in_info['symbol']
                    amount_out_symbol = token_out_info['symbol']
                    amount_in = trade_amount
                    amount_out = token_out_balance_after - token_out_balance_before
                else:
                    # Sold token_out for token_in
                    amount_in_symbol = token_out_info['symbol']
                    amount_out_symbol = token_in_info['symbol']
                    amount_in = trade_amount
                    amount_out = token_in_balance_after - token_in_balance_before

                # Recalculate token values after trade
                token_in_value_after = token_in_info_after['balance'] * Decimal(str(token_in_price))
                token_out_value_after = token_out_info_after['balance'] * Decimal(str(token_out_price))
                total_value_after = token_in_value_after + token_out_value_after

                # Calculate profit as change in total portfolio value
                profit_usd = float(total_value_after - total_value)
                gas_cost_usd = trade_result.get('gas_cost_usd', 0)
                net_profit_usd = profit_usd - gas_cost_usd

                # Log trade information
                trade_info = {
                    'timestamp': time.time(),
                    'trade_type': 'rebalancing',
                    'pair': f"{amount_in_symbol}-{amount_out_symbol}",
                    'amount_in': float(amount_in),
                    'amount_in_symbol': amount_in_symbol,
                    'amount_out': float(amount_out),
                    'amount_out_symbol': amount_out_symbol,
                    'profit_usd': profit_usd,
                    'gas_cost_usd': float(gas_cost_usd),
                    'net_profit_usd': net_profit_usd,
                    'gas_price_gwei': self.w3.eth.gas_price / 1e9,
                    'transaction_hash': trade_result.get('transaction_hash', '')
                }
                self.save_rebalance_trade(trade_info)

                # Optionally, log the trade info for debugging
                self.logger.info(f"Rebalance trade info: {trade_info}")

            else:
                self.logger.error("Trade execution failed during rebalancing")

        except Exception as e:
            self.logger.error(f"Error balancing portfolio: {str(e)}")   
    async def monitor_and_execute_arbitrage(
        self,
        token_pairs: List[Dict[str, str]], 
        profit_thresholds: Optional[ProfitThresholds] = None,
        check_interval: float = 2.0,
        max_gas_gwei: int = 50,
        amount_percentage: float = 0.95,
        min_auto_execute_profit: float = 0.1,
        enable_balancing: bool = False,
        target_balance_ratio: float = 0.5,
        rebalance_threshold: float = 0.05,
        initial_allocation: Dict[str, float] = None,  # Add this
        trade_settings: Dict = None                   # Add this
    ) -> None:
        if profit_thresholds is None:
            profit_thresholds = ProfitThresholds()

        execution_stats = {
            'attempts': 0,
            'successes': 0,
            'total_profit': Decimal('0'),
            'last_execution': None,
            'opportunities_found': 0,
            'errors': 0,
            'start_time': time.time(),
            'trades': []
        }
        # Add right after execution_stats initialization
        last_balance_check = time.time()
        last_rebalance_check = time.time()
        # Setup enhanced logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        # Add logging to file
        log_file = log_dir / f"arbitrage_{time.strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

        self.logger.info("\nStarting Arbitrage Monitor & Executor"
                    f"\nMin Execute Profit: ${min_auto_execute_profit}"
                    f"\nMax Gas Price: {max_gas_gwei} gwei"
                    f"\nTrade Size: {amount_percentage*100}% of balance"
                    f"\nCheck Interval: {check_interval}s"
                    "\nPress Ctrl+C to stop")
        # At the start of the function, after existing initialization
        self.logger.info("\nInitializing arbitrage monitoring with configuration:")
        self.logger.info(f"Rebalance interval: {self.rebalance_config['min_interval']/3600:.1f} hours")
        self.logger.info(f"Deviation threshold: {self.rebalance_config['deviation_threshold']*100:.1f}%")
        self.logger.info(f"Max gas price: {self.rebalance_config['max_gas_gwei']} gwei")
        last_rebalance_check = time.time()
        while True:
            try:
                current_gas_price = self.w3.eth.gas_price / 1e9
                if current_gas_price > max_gas_gwei:
                    self.logger.info(f"Gas price too high: {current_gas_price:.1f} gwei > {max_gas_gwei} gwei")
                    await asyncio.sleep(check_interval)
                    continue
                # Add after current_gas_price check
                current_time = time.time()

                # Balance check every 30 minutes
                if current_time - last_balance_check >= 1800:
                    await self.monitor_balances({
                        token: float(amount_in_human) 
                        for token, amount_in_human in initial_allocation.items()
                    })
                    last_balance_check = current_time

                # Rebalance check every 4 hours
                if enable_balancing and current_time - last_rebalance_check >= 14400:
                    has_rebalanced = await self.periodic_rebalance(
                        token_pairs, 
                        target_balance_ratio,
                        rebalance_threshold
                    )
                    if has_rebalanced:
                        last_rebalance_check = current_time
                        await asyncio.sleep(check_interval * 2)  # Brief pause after rebalancing
                        continue

                for pair in token_pairs:
                    try:
                        if "is_triangular" in pair and pair["is_triangular"]:
                            # Get initial token info and calculate trade amount
                            input_token_info = await self.get_token_info(pair["full_path"][0])
                            trade_amount = min(
                                input_token_info['balance'] * Decimal(str(amount_percentage)),
                                Decimal(str(input_token_info['balance'])) * Decimal('0.4')  # Cap at 40% of balance
                            )                            
                            # Handle triangular arbitrage
                            triangle_result = await self.check_triangular_opportunity(
                                pair["full_path"],
                                trade_amount
                            )
                            
                            if triangle_result:
                                net_profit_usd = triangle_result['net_profit_usd']
                                if net_profit_usd > min_auto_execute_profit:
                                    try:
                                        execution_stats['attempts'] += 1
                                        trade_result = await self.execute_triangular_trade(
                                            triangle_result,
                                            amount_percentage,
                                            slippage=0.5
                                        )
                                        
                                        if trade_result['net_profit_usd'] > 0:
                                            execution_stats['successes'] += 1
                                            execution_stats['total_profit'] += Decimal(str(trade_result['net_profit_usd']))
                                            execution_stats['last_execution'] = time.time()
                                            execution_stats['trades'].append(trade_result)
                                        
                                    except Exception as e:
                                        self.logger.error(f"Error executing triangular trade: {str(e)}")
                                        execution_stats['errors'] += 1
                                else:
                                    self.logger.info(
                                        f"Not executing: Net profit (${net_profit_usd:.4f}) "
                                        f"below minimum (${min_auto_execute_profit})"
                                    )
                            continue

                        # Check both directions for each pair
                        input_token_info = await self.get_token_info(pair["in"])
                        output_token_info = await self.get_token_info(pair["out"])

                        directions = [
                            {"in": pair["in"], "out": pair["out"]},  # USDC -> WETH
                            {"in": pair["out"], "out": pair["in"]}   # WETH -> USDC
                        ]
                        for direction in directions:
                            input_token_info = await self.get_token_info(direction["in"])
                            output_token_info = await self.get_token_info(direction["out"])
                            
                            # Skip if insufficient balance
                            if input_token_info['balance'] <= 0:
                                continue

                            trade_amount = min(
                                input_token_info['balance'] * Decimal(str(amount_percentage)),
                                Decimal(str(input_token_info['balance'])) * Decimal('0.4')  # Cap at 40% of balance
                            )
                            amount_in_wei = int(trade_amount * (10 ** input_token_info['decimals']))

                            self.logger.info(f"\n{'='*50}")
                            self.logger.info("Starting new arbitrage check cycle")
                            self.logger.info(
                                f"\nAnalyzing {input_token_info['symbol']}-{output_token_info['symbol']}"
                                f"\nBalance: {float(input_token_info['balance']):.6f} {input_token_info['symbol']}"
                                f"\nTrade Amount: {float(trade_amount):.6f} {input_token_info['symbol']}"
                                f"\nGas Price: {current_gas_price:.1f} gwei"
                            )

                            # Get forward quote
                            forward_quote = await self.get_quote(
                                pair["in"],
                                pair["out"],
                                trade_amount,
                                0.5
                            )
                            if not forward_quote:
                                self.logger.info("Failed to get forward quote")
                                continue
                            out_amount = Decimal(forward_quote["outAmounts"][0])
                            out_amount_human = out_amount / Decimal(str(10**output_token_info['decimals']))
                            
                            self.logger.info(f"Forward Quote: {float(out_amount_human):.8f} {output_token_info['symbol']}")
                            self.logger.info(f"Price Impact: {forward_quote.get('priceImpact', 0)}%")
                            
                            # Check if we have enough balance for the reverse trade
                            #if out_amount_human > output_token_info['balance']:
                            #    self.logger.info(
                            #        f"Insufficient balance for reverse trade. Need: {float(out_amount_human):.8f} "
                            #        f"Have: {float(output_token_info['balance']):.8f} {output_token_info['symbol']}"
                            #    )
                            #    continue
                            # Get reverse quote
                            self.logger.info("\nGetting reverse quote...")
                            reverse_quote = await self.get_quote(
                                pair["out"],
                                pair["in"],
                                out_amount_human,
                                0.5
                            )
                            if not reverse_quote:
                                self.logger.info("Failed to get reverse quote")
                                continue
                            # Calculate profits
                            final_amount = Decimal(reverse_quote["outAmounts"][0]) / Decimal(str(10**input_token_info['decimals']))
                            profit_amount = final_amount - trade_amount
                            
                            self.logger.info(f"\nProfit calculation:")
                            self.logger.info(f"Input amount: {float(trade_amount):.6f} {input_token_info['symbol']}")
                            self.logger.info(f"Final amount: {float(final_amount):.6f} {input_token_info['symbol']}")
                            self.logger.info(f"Raw profit: {float(profit_amount):.6f} {input_token_info['symbol']}")
                            
                            # Get token price and calculate USD values
                            token_price = await self.get_token_price(pair["in"])
                            if not token_price:
                                self.logger.info("Failed to get token price")
                                continue
                                
                            profit_usd = float(profit_amount * Decimal(str(token_price)))
                            
                            # Calculate gas costs
                            total_gas_estimate = forward_quote.get("gasEstimate", 150000) + reverse_quote.get("gasEstimate", 150000)
                            self.logger.info(f"Total gas estimate: {total_gas_estimate}")
                            
                            gas_cost_usd = await self.calculate_gas_cost_usd(total_gas_estimate)
                            net_profit_usd = profit_usd - gas_cost_usd

                            self.logger.info(
                                f"\nPotential Arbitrage Found:"
                                f"\nToken Price: ${token_price:.4f}"
                                f"\nProfit Amount: {float(profit_amount):.6f} {input_token_info['symbol']}"
                                f"\nProfit USD: ${profit_usd:.4f}"
                                f"\nGas Cost USD: ${gas_cost_usd:.4f}"
                                f"\nNet Profit USD: ${net_profit_usd:.4f}"
                                f"\nMin Required Profit: ${min_auto_execute_profit}"
                            )

                            # Check execution criteria
                            if net_profit_usd < min_auto_execute_profit:
                                self.logger.info(
                                    f"Not executing: Net profit (${net_profit_usd:.4f}) below minimum (${min_auto_execute_profit})"
                                )
                                continue

                            if profit_amount <= 0:
                                self.logger.info(
                                    f"Not executing: No positive profit. Profit amount: {float(profit_amount):.6f}"
                                )
                                continue

                            price_impact = forward_quote.get("priceImpact", 0)
                            if price_impact > 5:
                                self.logger.info(f"Not executing: Price impact too high ({price_impact}%)")
                                continue

                            if (execution_stats['last_execution'] and 
                                time.time() - execution_stats['last_execution'] < 60):
                                self.logger.info("Not executing: Too soon since last trade (cooling period)")
                                continue
                            # Check execution criteria
                            if net_profit_usd < min_auto_execute_profit:
                                self.logger.info(
                                    f"Not executing: Net profit (${net_profit_usd:.4f}) below minimum (${min_auto_execute_profit})"
                                )
                                continue

                            if profit_amount <= 0:
                                self.logger.info("Not executing: No positive profit")
                                continue

                            price_impact = forward_quote.get("priceImpact", 0)
                            if price_impact > 5:
                                self.logger.info(f"Not executing: Price impact too high ({price_impact}%)")
                                continue

                            # Check time since last execution
                            if (execution_stats['last_execution'] and 
                                time.time() - execution_stats['last_execution'] < 60):
                                self.logger.info("Not executing: Too soon since last trade (cooling period)")
                                continue

                            # Execute the trades
                            # Execute the trades
                            self.logger.info("\n🚀 Profitable opportunity found! Executing trades...")
                            execution_stats['attempts'] += 1

                            try:
                                # Check if we're in a cycle
                                if self.tracker.is_cycle_active():
                                    intermediate_token = self.tracker.get_current_cycle_token()
                                    initial_token = self.tracker.current_cycle['pair'].split('-')[0]
                                    
                                    # Get token info for reverse trade
                                    token_info = await self.get_token_info(intermediate_token)
                                    reverse_amount = token_info['balance'] * Decimal(str(amount_percentage))
                                    
                                    # Execute reverse trade
                                    reverse_result = await self.execute_trade(
                                        intermediate_token,
                                        initial_token,
                                        reverse_amount,
                                        0.5
                                    )
                                    
                                    if reverse_result['status'] == 'success':
                                        cycle_result = self.tracker.complete_cycle(reverse_result)
                                        # Update statistics
                                        execution_stats['successes'] += 1
                                        execution_stats['total_profit'] += Decimal(str(cycle_result['net_profit_usd']))
                                        execution_stats['last_execution'] = time.time()
                                        continue
                                
                                # Start new cycle with forward trade
                                forward_result = await self.execute_trade(
                                    pair["in"],
                                    pair["out"],
                                    trade_amount,
                                    0.5
                                )

                                if forward_result['status'] != 'success':
                                    self.logger.error("Forward trade failed")
                                    continue

                                # Start tracking new cycle
                                self.tracker.start_cycle({
                                    'pair': f"{input_token_info['symbol']}-{output_token_info['symbol']}",
                                    'amount': float(trade_amount),
                                    'tx_hash': forward_result['transaction_hash'],
                                    'gas_cost_usd': forward_result['gas_cost_usd'],
                                    'received_amount': forward_result['received_amount'],
                                    'output_token': pair['out']
                                })

                                # Update statistics for forward trade
                                execution_stats['successes'] += 1
                                execution_stats['last_execution'] = time.time()

                                self.logger.info(f"\n✅ Forward trade executed successfully!")
                                # Update output token info (WETH balance)
                                output_token_info = await self.get_token_info(pair["out"])

                                # Get actual output amount received from the forward trade
                                actual_out_amount = Decimal(forward_result['received_amount'])

                                # Get actual output amount
                                actual_out_amount = Decimal(forward_result['received_amount'])
                                
                                # Execute reverse trade
                                reverse_result = await self.execute_trade(
                                    pair["out"],
                                    pair["in"],
                                    actual_out_amount,
                                    0.5
                                )

                                if reverse_result['status'] != 'success':
                                    self.logger.error("Reverse trade failed")
                                    continue
                                
                                # Calculate actual final amount in input token (USDC)
                                actual_final_amount = Decimal(reverse_result['received_amount']) / Decimal(str(10**input_token_info['decimals']))

                                # Calculate actual profits
                                actual_profit_amount = actual_final_amount - trade_amount
                                actual_profit_usd = float(actual_profit_amount * Decimal(str(token_price)))
                                total_gas_cost = forward_result['gas_cost_usd'] + reverse_result['gas_cost_usd']
                                actual_net_profit = actual_profit_usd - total_gas_cost

                                # Update statistics
                                execution_stats['successes'] += 1
                                execution_stats['total_profit'] += Decimal(str(actual_net_profit))
                                execution_stats['last_execution'] = time.time()

                                # Record trade details
                                trade_info = {
                                    'timestamp': time.time(),
                                    'pair': f"{input_token_info['symbol']}-{output_token_info['symbol']}",
                                    'amount': float(trade_amount),
                                    'profit_usd': actual_net_profit,
                                    'gas_cost_usd': total_gas_cost,
                                    'gas_price_gwei': current_gas_price,
                                    'forward_tx': forward_result['transaction_hash'],
                                    'reverse_tx': reverse_result['transaction_hash']
                                }
                                execution_stats['trades'].append(trade_info)

                                # Save trade to history
                                self.save_trade_history(trade_info)

                                # Send notifications
                                await self.send_notification(
                                    f"Arbitrage Complete!\n"
                                    f"Net Profit: ${actual_net_profit:.2f}\n"
                                    f"Total Profit: ${float(execution_stats['total_profit']):.2f}"
                                )

                                self.logger.info(
                                    f"\n✅ Arbitrage executed successfully!"
                                    f"\nActual Net Profit: ${actual_net_profit:.2f}"
                                    f"\nTotal Gas Cost: ${total_gas_cost:.2f}"
                                    f"\nCumulative Profit: ${float(execution_stats['total_profit']):.2f}"
                                    f"\nSuccess Rate: {execution_stats['successes']}/{execution_stats['attempts']}"
                                )
                                #if actual_net_profit > min_auto_execute_profit:
                                    #await self.send_status_email(
                                    #    "Arbitrage Trade Success",
                                    #    f"Pair: {input_token_info['symbol']}-{output_token_info['symbol']}<br>" +
                                    #    f"Net Profit: ${actual_net_profit:.4f}<br>" +
                                    #    f"Gas Cost: ${total_gas_cost:.4f}<br>" +
                                    #    f"Total Profit: ${float(execution_stats['total_profit']):.2f}",
                                    #    urgent=False
                                    #)
                                # Add near your existing logging
                                if execution_stats['successes'] > 0:
                                    avg_profit = float(execution_stats['total_profit']) / execution_stats['successes']
                                    self.logger.info(
                                        f"\nTrading Statistics:"
                                        f"\nAverage Profit/Trade: ${avg_profit:.4f}"
                                        f"\nSuccess Rate: {(execution_stats['successes']/execution_stats['attempts'])*100:.1f}%"
                                        f"\nTotal Profit: ${float(execution_stats['total_profit']):.2f}"
                                    )

                            except Exception as e:
                                self.logger.error(f"Error executing trades: {str(e)}")
                                execution_stats['errors'] += 1
                                continue

                    except Exception as e:
                        self.logger.error(f"Error processing pair: {str(e)}")
                        execution_stats['errors'] += 1
                        continue
                if enable_balancing:
                    for pair in token_pairs:
                        await self.balance_portfolio(pair, target_balance_ratio)

                await asyncio.sleep(check_interval)
                self.logger.info(f"API Calls Made: {self.get_api_call_count()}")
                self.logger.info(f"Remaining API Calls: {self.get_remaining_api_calls()}")

            except KeyboardInterrupt:
                runtime = time.time() - execution_stats['start_time']
                opps_per_hour = (execution_stats['opportunities_found'] / runtime) * 3600
                
                self.logger.info(
                    f"\n{'='*50}"
                    f"\nSession Summary"
                    f"\nRuntime: {runtime/3600:.1f} hours"
                    f"\nOpportunities Found: {execution_stats['opportunities_found']}"
                    f"\nOpportunities/Hour: {opps_per_hour:.1f}"
                    f"\nExecution Attempts: {execution_stats['attempts']}"
                    f"\nSuccessful Trades: {execution_stats['successes']}"
                    f"\nErrors Encountered: {execution_stats['errors']}"
                    f"\nTotal Profit: ${float(execution_stats['total_profit']):.2f}"
                    f"\nAverage Profit/Trade: ${float(execution_stats['total_profit']/execution_stats['successes']):.2f}" if execution_stats['successes'] > 0 else "\nNo successful trades"
                    f"\n{'='*50}"
                )
                return
            except Exception as e:
                self.logger.error(f"Monitor error: {str(e)}")
                execution_stats['errors'] += 1
                await asyncio.sleep(check_interval)    
    async def check_triangular_opportunity(
        self,
        path: List[str],
        amount_in_human: Decimal
    ) -> Optional[Dict]:
        """Check for triangular arbitrage opportunities"""
        try:
            # Get initial token info
            initial_token_info = await self.get_token_info(path[0])
            self.logger.info(f"\n{'='*50}")
            self.logger.info("Starting triangular arbitrage check")
            self.logger.info(f"Path: {' -> '.join([initial_token_info['symbol']])}")
            
            # Track amounts through the path
            current_amount = amount_in_human
            quotes = []
            total_gas = 0
            steps = []
            
            # Get quotes for each step in the path
            for i in range(len(path)):
                token_in = path[i]
                token_out = path[(i + 1) % len(path)]  # Use modulo to loop back to first token
                
                token_in_info = await self.get_token_info(token_in)
                token_out_info = await self.get_token_info(token_out)
                
                self.logger.info(
                    f"\nStep {i+1}: {token_in_info['symbol']} -> {token_out_info['symbol']}"
                    f"\nAmount: {float(current_amount):.6f} {token_in_info['symbol']}"
                )
                
                quote = await self.get_quote(
                    token_in,
                    token_out,
                    current_amount,
                    0.5
                )
                
                if not quote:
                    self.logger.info(f"Failed to get quote for step {i+1}")
                    return None
                    
                out_amount = Decimal(quote["outAmounts"][0])
                out_amount_human = out_amount / Decimal(str(10 ** token_out_info['decimals']))
                
                steps.append({
                    'token_in': token_in,
                    'token_out': token_out,
                    'amount_in': float(current_amount),
                    'amount_out': float(out_amount_human),
                    'quote': quote
                })
                
                self.logger.info(f"Quote received: {float(out_amount_human):.8f} {token_out_info['symbol']}")
                self.logger.info(f"Price Impact: {quote.get('priceImpact', 0)}%")
                
                quotes.append(quote)
                current_amount = out_amount_human
                total_gas += quote.get("gasEstimate", 150000)
            
            # Calculate profit
            final_amount = current_amount
            profit_amount = final_amount - amount_in_human
            
            # Get USD value
            token_price = await self.get_token_price(path[0])
            if not token_price:
                self.logger.info("Failed to get token price")
                return None
                
            profit_usd = float(profit_amount * Decimal(str(token_price)))
            gas_cost_usd = await self.calculate_gas_cost_usd(total_gas)
            net_profit_usd = profit_usd - gas_cost_usd
            
            self.logger.info(f"\nTriangular Arbitrage Summary:")
            self.logger.info(f"Initial Amount: {float(amount_in_human):.6f} {initial_token_info['symbol']}")
            self.logger.info(f"Final Amount: {float(final_amount):.6f} {initial_token_info['symbol']}")
            self.logger.info(f"Profit Amount: {float(profit_amount):.6f} {initial_token_info['symbol']}")
            self.logger.info(f"Profit USD: ${profit_usd:.4f}")
            self.logger.info(f"Gas Cost USD: ${gas_cost_usd:.4f}")
            self.logger.info(f"Net Profit USD: ${net_profit_usd:.4f}")
            
            return {
                'profit_amount': profit_amount,
                'profit_usd': profit_usd,
                'gas_cost_usd': gas_cost_usd,
                'net_profit_usd': net_profit_usd,
                'path': path,
                'quotes': quotes,
                'steps': steps,
                'total_gas': total_gas,
                'initial_token': initial_token_info['symbol'],
                'initial_amount': amount_in_human
            }
            
        except Exception as e:
            self.logger.error(f"Error checking triangular opportunity: {str(e)}")
            return None
    async def execute_triangular_trade(
        self,
        opportunity: Dict,
        amount_percentage: float = 0.95,
        slippage: float = 0.5
    ) -> Dict:
        """Execute a triangular arbitrage trade"""
        try:
            self.logger.info("\n🔺 Executing triangular arbitrage...")
            path = opportunity['path']
            steps = opportunity['steps']
            
            results = []
            current_amount = opportunity['initial_amount']
            
            for i, step in enumerate(steps):
                token_in = step['token_in']
                token_out = step['token_out']
                
                self.logger.info(
                    f"\nExecuting step {i+1}: {step['amount_in']:.6f} "
                    f"{(await self.get_token_info(token_in))['symbol']} -> "
                    f"{(await self.get_token_info(token_out))['symbol']}"
                )

                # Execute trade for this step
                trade_result = await self.execute_trade(
                    token_in,
                    token_out,
                    Decimal(str(current_amount)),
                    slippage
                )

                if trade_result['status'] != 'success':
                    raise Exception(f"Trade failed at step {i+1}")

                results.append(trade_result)
                current_amount = Decimal(trade_result['received_amount'])
                
                # Small delay between trades to ensure transaction confirmations
                await asyncio.sleep(2)

            # Calculate actual results
            total_gas_cost = sum(r['gas_cost_usd'] for r in results)
            actual_final_amount = Decimal(results[-1]['received_amount'])
            actual_profit_amount = actual_final_amount - opportunity['initial_amount']
            
            # Get current token price for accurate profit calculation
            token_price = await self.get_token_price(path[0])
            actual_profit_usd = float(actual_profit_amount * Decimal(str(token_price)))
            actual_net_profit = actual_profit_usd - total_gas_cost

            trade_info = {
                'timestamp': time.time(),
                'type': 'triangular',
                'path': [
                    (await self.get_token_info(p))['symbol']
                    for p in path
                ],
                'initial_amount': float(opportunity['initial_amount']),
                'final_amount': float(actual_final_amount),
                'profit_usd': actual_profit_usd,
                'gas_cost_usd': total_gas_cost,
                'net_profit_usd': actual_net_profit,
                'transactions': [r['transaction_hash'] for r in results],
            }

            # Save trade info
            self.save_trade_history(trade_info)

            self.logger.info(
                f"\n✅ Triangular arbitrage executed successfully!"
                f"\nActual profit amount: {float(actual_profit_amount):.6f}"
                f"\nActual profit USD: ${actual_profit_usd:.4f}"
                f"\nTotal gas cost: ${total_gas_cost:.4f}"
                f"\nNet profit: ${actual_net_profit:.4f}"
            )

            # Send notification
            await self.send_notification(
                f"🔺 Triangular Arbitrage Complete!\n"
                f"Path: {' -> '.join(trade_info['path'])}\n"
                f"Net Profit: ${actual_net_profit:.2f}"
            )

            return trade_info

        except Exception as e:
            self.logger.error(f"Error executing triangular trade: {str(e)}")
            raise
    def save_rebalance_trade(self, trade_info: Dict) -> None:
        """Save rebalancing trade details to a file for analysis."""
        try:
            history_file = "rebalance_history.json"
            
            # Load existing history
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except FileNotFoundError:
                history = []
                
            # Add new trade
            history.append(trade_info)
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving rebalance trade history: {str(e)}")
    def save_trade_history(self, trade_info: Dict) -> None:
        """Save trade details to a file for historical analysis"""
        try:
            history_file = "trade_history.json"
            
            # Load existing history
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except FileNotFoundError:
                history = []
                
            # Add new trade
            history.append(trade_info)
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving trade history: {str(e)}")
    def setup_logging(self, log_dir: str):
        """Enhanced logging setup"""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a logs subdirectory
        logs_subdir = log_dir / "logs"
        logs_subdir.mkdir(exist_ok=True)
        
        # Create log filename with date
        log_file = logs_subdir / f"odos_trades_{time.strftime('%Y%m%d')}.log"
        
        # Configure logging
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # Set up file handler with daily rotation
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        self.logger.handlers = []
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Add log file to .gitignore if not already present
        gitignore_path = Path(".gitignore")
        if not gitignore_path.exists():
            gitignore_path.write_text("logs/\n")
        else:
            gitignore_content = gitignore_path.read_text()
            if "logs/" not in gitignore_content:
                with gitignore_path.open("a") as f:
                    f.write("\n# Log files\nlogs/\n")
    def analyze_bot_performance(self, days_back: int = 365) -> Dict:
        """
        Analyze bot's trading performance over the specified period
        
        Args:
            days_back: Number of days to analyze (default: 365)
        """
        try:
            # Load trade history
            with open("trade_history.json", 'r') as f:
                history = json.load(f)
                
            # Calculate time threshold
            cutoff_time = time.time() - (days_back * 24 * 60 * 60)
            
            # Filter relevant trades
            relevant_trades = [
                trade for trade in history 
                if trade['timestamp'] >= cutoff_time
            ]
            
            if not relevant_trades:
                return {
                    'status': 'No trades found in specified period',
                    'days_analyzed': days_back
                }
                
            # Calculate statistics
            total_profit = sum(trade['profit_usd'] for trade in relevant_trades)
            total_gas = sum(trade['gas_cost_usd'] for trade in relevant_trades)
            trade_count = len(relevant_trades)
            
            # Calculate daily/monthly averages
            days_covered = min(days_back, (time.time() - relevant_trades[0]['timestamp']) / (24 * 60 * 60))
            
            results = {
                'period_days': days_back,
                'actual_days': round(days_covered, 1),
                'total_trades': trade_count,
                'total_profit_usd': round(total_profit, 2),
                'total_gas_cost_usd': round(total_gas, 2),
                'net_profit_usd': round(total_profit - total_gas, 2),
                'average_per_trade_usd': round((total_profit - total_gas) / trade_count, 2),
                'average_per_day_usd': round((total_profit - total_gas) / days_covered, 2),
                'average_per_month_usd': round((total_profit - total_gas) / days_covered * 30, 2),
                'trades_per_day': round(trade_count / days_covered, 1),
                'success_rate': 1.0,  # Since we only record successful trades
                'gas_efficiency': round(total_profit / total_gas, 2) if total_gas > 0 else float('inf'),
            }
            
            return results
            
        except FileNotFoundError:
            return {
                'status': 'No trade history found',
                'days_analyzed': days_back
            }
        except Exception as e:
            return {
                'status': f'Error analyzing performance: {str(e)}',
                'days_analyzed': days_back
            }       
    def send_arbitrage_notification(self, trade_result: Dict) -> None:
        """Send detailed email notification about the arbitrage trade"""
        try:
            email_user = os.getenv('EMAIL_USER')
            email_password = os.getenv('EMAIL_PASSWORD')
            email_host = os.getenv('EMAIL_HOST')
            email_port = int(os.getenv('EMAIL_PORT', '587'))

            if not all([email_user, email_password, email_host]):
                self.logger.error("Email credentials not properly configured")
                return

            # Enhanced trade details
            input_amount = Decimal(trade_result['input_amount'])
            output_amount = Decimal(trade_result['output_amount'])
            percentage_gain = ((output_amount / input_amount) - 1) * 100

            msg = MIMEMultipart('alternative')
            msg['From'] = f"Odos Arbitrage Bot <{email_user}>"
            msg['To'] = email_user
            msg['Subject'] = f"🚀 Arbitrage Success: ${trade_result['net_profit_usd']:.2f} Profit ({percentage_gain:.2f}%)"

            html_content = f"""
            <html>
                <head>
                    <style>
                        .container {{ font-family: Arial, sans-serif; padding: 20px; }}
                        .header {{ background-color: #4CAF50; color: white; padding: 10px; }}
                        .profit {{ color: green; font-weight: bold; font-size: 1.2em; }}
                        .loss {{ color: red; font-weight: bold; }}
                        .details {{ margin: 20px 0; }}
                        .transaction {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; }}
                        .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h2>Arbitrage Trade Complete</h2>
                            <p>Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                        </div>
                        <div class="details">
                            <h3>Trade Path:</h3>
                            <p>{trade_result['buy_dex'].upper()} ➜ {trade_result['sell_dex'].upper()}</p>
                            
                            <div class="metrics">
                                <div>
                                    <h4>Input:</h4>
                                    <p>{trade_result['input_amount']} {trade_result.get('input_symbol', '')}</p>
                                </div>
                                <div>
                                    <h4>Output:</h4>
                                    <p>{trade_result['output_amount']} {trade_result.get('output_symbol', '')}</p>
                                </div>
                            </div>
                            
                            <h3>Profit Analysis:</h3>
                            <p class="profit">Net Profit: ${trade_result['net_profit_usd']:.4f}</p>
                            <p>Return: {percentage_gain:.2f}%</p>
                            <p>Gas Cost: ${trade_result['total_gas_cost_usd']:.4f}</p>
                            
                            <h3>Market Conditions:</h3>
                            <p>Gas Price: {trade_result.get('gas_price_gwei', 'N/A')} gwei</p>
                            <p>MATIC Price: ${trade_result.get('matic_price_usd', 0):.4f}</p>
                        </div>
                        <div class="transaction">
                            <h3>Transaction Details:</h3>
                            <p>Buy Transaction: 
                                <a href="https://polygonscan.com/tx/{trade_result['buy_transaction']}">
                                    View on Polygonscan
                                </a>
                            </p>
                            <p>Sell Transaction: 
                                <a href="https://polygonscan.com/tx/{trade_result['sell_transaction']}">
                                    View on Polygonscan
                                </a>
                            </p>
                        </div>
                    </div>
                </body>
            </html>
            """

            msg.attach(MIMEText(html_content, 'html'))

            with smtplib.SMTP(email_host, email_port) as server:
                server.starttls()
                server.login(email_user, email_password)
                server.send_message(msg)

            self.logger.info("Detailed arbitrage notification email sent successfully")
            
        except Exception as e:
            self.logger.error(f"Error sending arbitrage notification: {str(e)}")
    def send_trade_notification(self, trade_details: Dict) -> None:
        """Send email notification about the trade"""
        try:
            # Get email credentials from environment
            email_user = os.getenv('EMAIL_USER')
            email_password = os.getenv('EMAIL_PASSWORD')
            email_host = os.getenv('EMAIL_HOST')
            email_port = int(os.getenv('EMAIL_PORT', '587'))

            if not all([email_user, email_password, email_host]):
                self.logger.error("Email credentials not properly configured")
                return

            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_user
            msg['To'] = email_user  # Sending to self
            msg['Subject'] = 'Odos Trade Execution Notification'

            # Create email body
            body = f"""
                Trade Execution Summary:
                -----------------------
                Status: {trade_details['status']}
                Transaction Hash: {trade_details['transaction_hash']}
                Input Amount: {trade_details['input_amount']} {trade_details.get('input_symbol', '')}
                Expected Output: {trade_details['expected_output']} {trade_details.get('output_symbol', '')}
                Price Impact: {trade_details['price_impact']}%
                Gas Used: {trade_details['gas_used']}
                Gas Cost (USD): ${trade_details['gas_cost_usd']:.4f}
                Block Number: {trade_details['block_number']}

                Transaction Explorer:
                https://polygonscan.com/tx/{trade_details['transaction_hash']}

                Trade Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

                Note: This is an automated notification from your Odos Trading Bot.
                """
            msg.attach(MIMEText(body, 'plain'))

            # Setup SMTP server
            server = smtplib.SMTP(email_host, email_port)
            server.starttls()
            server.login(email_user, email_password)

            # Send email
            text = msg.as_string()
            server.sendmail(email_user, email_user, text)
            server.quit()

            self.logger.info("Trade notification email sent successfully")
            
        except Exception as e:
            self.logger.error(f"Error sending trade notification email: {str(e)}")
    async def send_status_email(self, subject: str, content: str, urgent: bool = False) -> None:
        """Send status email with formatted content"""
        try:
            email_user = os.getenv('EMAIL_USER')
            email_password = os.getenv('EMAIL_PASSWORD')
            email_host = os.getenv('EMAIL_HOST')
            email_port = int(os.getenv('EMAIL_PORT', '587'))

            msg = MIMEMultipart('alternative')
            msg['From'] = f"Odos Bot <{email_user}>"
            msg['To'] = email_user
            msg['Subject'] = f"{'🚨 ' if urgent else ''}Bot Alert: {subject}"

            # Create HTML content
            html_content = f"""
            <html>
                <body style="font-family: Arial, sans-serif;">
                    <div style="padding: 20px;">
                        <h2 style="color: {'red' if urgent else 'black'};">{subject}</h2>
                        <div style="margin: 20px 0;">
                            {content}
                        </div>
                        <div style="color: gray; font-size: 12px;">
                            Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}
                        </div>
                    </div>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html_content, 'html'))
            
            with smtplib.SMTP(email_host, email_port) as server:
                server.starttls()
                server.login(email_user, email_password)
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"Failed to send status email: {str(e)}")
    def setup_telegram_notifications(self):
        """Setup Telegram notifications"""
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if self.telegram_token and self.telegram_chat_id:
            import telegram
            self.telegram_bot = telegram.Bot(token=self.telegram_token)
        else:
            self.telegram_bot = None
    async def send_notification(self, message: str, is_error: bool = False) -> None:
        """Send notification via Telegram"""
        try:
            if self.telegram_bot:
                # Add emoji based on message type
                prefix = "🚨 " if is_error else "💰 "
                await self.telegram_bot.send_message(
                    chat_id=self.telegram_chat_id,
                    text=f"{prefix}{message}"
                )
        except Exception as e:
            self.logger.error(f"Failed to send notification: {str(e)}")
    def get_api_call_count(self) -> int:
        """Get the total number of API calls made in the current window."""
        return len(self.rate_limiter.requests)
    def get_remaining_api_calls(self) -> int:
        """Get the number of remaining API calls allowed in the current window."""
        return self.rate_limiter.get_remaining_requests()
    async def get_quote(
            self,
            input_token: str,
            output_token: str,
            amount_in_human: Decimal,
            slippage: float = 0.5
        ) -> Dict:
            """Get quote from Odos API"""
            try:
                # Get token info and convert amount to wei
                token_info = await self.get_token_info(input_token)
                amount_in_wei = int(amount_in_human * (10 ** token_info['decimals']))
                
                # Verify the amount using balance manager
                amount_in_wei = int(await self.balance_manager.check_decimal_conversion(
                    Decimal(str(amount_in_wei)),
                    input_token  # Fix: Use input_token instead of token_in
                ))
                
                # Check balance
                if amount_in_wei > token_info['balance_wei']:
                    raise ValueError(
                        f"Insufficient balance. Have: {token_info['balance']} {token_info['symbol']}, "
                        f"Need: {amount_in_human} {token_info['symbol']}"
                    )
                
                request_body = {
                    "chainId": self.chain_id,
                    "inputTokens": [{
                        "tokenAddress": input_token,
                        "amount": str(amount_in_wei),
                    }],
                    "outputTokens": [{
                        "tokenAddress": output_token,
                        "proportion": 1
                    }],
                    "slippageLimitPercent": slippage,
                    "userAddr": self.wallet_address,
                    "referralCode": 0,
                    "compact": True,
                    "disableRFQs": True,
                }
                
                # Check if we can make a request
                if not self.rate_limiter.can_make_request():
                    self.logger.error("Rate limit exceeded. Cannot make more API requests at this time.")
                    raise Exception("Rate limit exceeded")

                # Record the API request
                self.rate_limiter.add_request()
                
                self.logger.info(f"Requesting quote: {amount_in_human} {token_info['symbol']} -> {output_token}")
                response = requests.post(self.QUOTE_URL, json=request_body)
                self.logger.info(f"API Call to {response.url} with status {response.status_code}")
                response.raise_for_status()
                
                quote = response.json()
                self.logger.info(f"Quote received. Path ID: {quote.get('pathId')}")
                return quote
                
            except Exception as e:
                self.logger.error(f"Error getting quote: {str(e)}")
                raise    
    async def show_advanced_menu(self):
        while True:
            print("\nAdvanced Settings:")
            print("1. Configure Rebalancing")
            print("2. View Rebalancing Stats")
            print("3. Emergency Conversion to USDC")
            print("4. Configure Trading Parameters")
            print("5. Check/Add Gas (MATIC)")
            print("6. Back to Main Menu")
            
            choice = input("\nSelect option: ")
            
            try:
                if choice == "1":
                    # Existing rebalancing configuration code...
                    pass
                elif choice == "2":
                    # Existing stats code...
                    pass
                elif choice == "3":
                    confirm = input("\nWARNING: This will convert all assets to USDC. Continue? (y/n): ")
                    if confirm.lower() == 'y':
                        # Check gas balance first
                        gas_status = await self.check_and_add_gas()
                        if gas_status['needs_more_gas']:
                            print("\nWARNING: Low gas balance. Please add more MATIC first.")
                            await self.show_gas_menu()
                            continue
                            
                        print("\nConverting all assets to USDC...")
                        result = await self.balance_manager.convert_all_to_usdc(self.tokens["USDC"])
                        print("\nConversion Results:")
                        for conv in result['conversions']:
                            if 'error' in conv:
                                print(f"{conv['token']}: Failed - {conv['error']}")
                            else:
                                print(f"{conv['token']}: {conv['amount']:.6f} converted")
                                print(f"Gas cost: ${conv['gas_cost']:.2f}")
                        print(f"\nTotal Gas Cost: ${result['total_gas_cost']:.2f}")
                elif choice == "4":
                    # Existing trading parameters code...
                    pass
                elif choice == "5":
                    await self.show_gas_menu()
                elif choice == "6":
                    break
                    
            except Exception as e:
                print(f"Error: {str(e)}")
                self.logger.error(f"Advanced menu error: {str(e)}") 
    async def execute_trade(
            self,
            input_token: str,
            output_token: str,
            amount_in_human: Decimal,
            slippage: float = 0.5
        ) -> Dict:
            """Execute a trade through Odos with accurate gas costs"""
            try:
                input_token_info = await self.get_token_info(input_token)
                output_token_info = await self.get_token_info(output_token)
                amount_in_wei = int(amount_in_human * (10 ** input_token_info['decimals']))

                if amount_in_wei > input_token_info['balance_wei']:
                    raise ValueError(
                        f"Insufficient balance. Have: {input_token_info['balance']} {input_token_info['symbol']}, "
                        f"Need: {amount_in_human} {input_token_info['symbol']}"
                    )

                await self.check_and_approve_token(input_token, amount_in_wei)
                quote = await self.get_quote(input_token, output_token, amount_in_human, slippage)

                price_impact = quote.get('priceImpact')
                if price_impact and price_impact > 5:
                    raise ValueError(f"Price impact too high: {price_impact}%")

                assembled = await self.assemble_transaction(quote['pathId'])

                if not assembled.get('simulation', {}).get('isSuccess'):
                    raise ValueError(f"Transaction simulation failed: {assembled.get('simulation', {}).get('simulationError')}")

                # Get current MATIC price for accurate gas cost calculation
                matic_price = await self.get_matic_price()
                if not matic_price:
                    raise ValueError("Could not get MATIC price for gas calculation")

                transaction = assembled['transaction']
                transaction['value'] = int(transaction['value'])
                transaction['chainId'] = self.chain_id

                # Get output token balance before the trade
                output_balance_before = output_token_info['balance']

                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(transaction, self.private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

                # Update output token info after the trade
                output_token_info_after = await self.get_token_info(output_token)
                output_balance_after = output_token_info_after['balance']

                # Calculate received amount
                received_amount = output_balance_after - output_balance_before

                # Calculate actual gas cost in USD
                gas_cost_usd = await self.calculate_gas_cost_usd(receipt['gasUsed'])

                trade_result = {
                    'status': 'success' if receipt['status'] == 1 else 'failed',
                    'transaction_hash': receipt['transactionHash'].hex(),
                    'gas_used': receipt['gasUsed'],
                    'block_number': receipt['blockNumber'],
                    'input_amount': str(amount_in_human),
                    'input_symbol': input_token_info['symbol'],
                    'expected_output': quote['outAmounts'][0],
                    'output_symbol': output_token_info['symbol'],
                    'price_impact': price_impact,
                    'gas_cost_usd': gas_cost_usd,
                    'matic_price_usd': matic_price,
                    'received_amount': str(received_amount)
                }

                self.logger.info(f"Trade executed successfully: {trade_result}")
                self.send_trade_notification(trade_result)

                return trade_result

            except Exception as e:
                self.logger.error(f"Error executing trade: {str(e)}")
                raise    
    async def monitor_balanc5es(self, initial_allocation: Dict[str, float]) -> None:
        """Monitor token balances and portfolio value"""
        try:
            total_value_usd = 0
            current_allocation = {}
            alerts = []

            for token, address in self.tokens.items():
                token_info = await self.get_token_info(address)
                token_price = await self.get_token_price(address)
                
                if not token_price:
                    continue
                    
                value_usd = float(token_info['balance']) * token_price
                total_value_usd += value_usd
                current_allocation[token] = value_usd

                # Check if balance is too low
                if value_usd < trade_settings['min_liquidity']:
                    alerts.append(f"Low {token} balance: ${value_usd:.2f}")

            # Check allocation deviations
            for token, value in current_allocation.items():
                target = initial_allocation[token]
                current_pct = (value / total_value_usd) * 100
                target_pct = (target / sum(initial_allocation.values())) * 100
                
                if abs(current_pct - target_pct) > 15:  # 15% deviation threshold
                    alerts.append(f"{token} allocation off target: {current_pct:.1f}% vs {target_pct:.1f}%")

            if alerts:
                alert_content = "<br>".join(alerts)
                await self.send_status_email(
                    "Balance Alert",
                    f"Portfolio Value: ${total_value_usd:.2f}<br><br>Alerts:<br>{alert_content}",
                    urgent=True
                )

            # Send daily summary
            if int(time.time()) % 86400 < 300:  # Send around midnight
                await self.send_daily_summary(total_value_usd, current_allocation)
                
        except Exception as e:
            self.logger.error(f"Error monitoring balances: {str(e)}")
    async def periodic_rebalance(self, token_pairs: List[Dict], initial_allocation: Dict[str, float]) -> None:
        """Periodic portfolio rebalancing check"""
        try:
            # Get current token values and total portfolio value
            portfolio = {}
            total_value = 0
            has_traded = False  # Track if we've made any trades this cycle

            for token, target in initial_allocation.items():
                token_info = await self.get_token_info(tokens[token])
                token_price = await self.get_token_price(tokens[token])
                
                if not token_price:
                    continue
                    
                value = float(token_info['balance']) * token_price
                portfolio[token] = {
                    'value': value,
                    'balance': token_info['balance'],
                    'price': token_price
                }
                total_value += value

            # Calculate deviations
            rebalance_trades = []
            for token, data in portfolio.items():
                current_pct = (data['value'] / total_value) * 100
                target_pct = (initial_allocation[token] / sum(initial_allocation.values())) * 100
                deviation = abs(current_pct - target_pct)

                if deviation > 10:  # If >10% off target
                    # Determine trade direction and amount
                    if current_pct > target_pct:
                        # Need to reduce this token
                        excess_value = data['value'] - (total_value * (target_pct / 100))
                        trade_amount = min(excess_value / data['price'], data['balance'] * 0.8)  # Don't use more than 80%
                        
                        # Find best token to trade into
                        for other_token, other_data in portfolio.items():
                            if other_token != token:
                                other_current_pct = (other_data['value'] / total_value) * 100
                                other_target_pct = (initial_allocation[other_token] / sum(initial_allocation.values())) * 100
                                
                                if other_current_pct < other_target_pct:
                                    rebalance_trades.append({
                                        'from_token': token,
                                        'to_token': other_token,
                                        'amount': trade_amount,
                                        'deviation': deviation
                                    })

            # Sort trades by deviation (largest first)
            rebalance_trades.sort(key=lambda x: x['deviation'], reverse=True)

            # Execute rebalancing trades if gas is reasonable
            current_gas = self.w3.eth.gas_price / 1e9
            if current_gas <= trade_settings['gas_limit']:
                for trade in rebalance_trades:
                    if has_traded:  # If we've already made a trade this cycle, wait
                        break
                        
                    try:
                        result = await self.execute_trade(
                            tokens[trade['from_token']],
                            tokens[trade['to_token']],
                            Decimal(str(trade['amount'])),
                            slippage=0.5
                        )
                        
                        if result['status'] == 'success':
                            has_traded = True
                            
                            # Send rebalance notification
                            await self.send_status_email(
                                "Rebalance Trade Executed",
                                f"Rebalanced {trade['from_token']} to {trade['to_token']}<br>" +
                                f"Amount: {trade['amount']:.6f}<br>" +
                                f"Gas Cost: ${result['gas_cost_usd']:.2f}",
                                urgent=False
                            )
                            
                    except Exception as e:
                        self.logger.error(f"Rebalance trade failed: {str(e)}")
                        continue

            return has_traded

        except Exception as e:
            self.logger.error(f"Error in periodic rebalance: {str(e)}")
            return False
    async def monitor_balances(self, initial_allocation: Dict[str, float]) -> None:
        """Monitor token balances and portfolio value with alerts"""
        try:
            total_value_usd = 0
            current_allocation = {}
            alerts = []
            previous_value_usd = 0
            try:
                with open('last_portfolio_value.json', 'r') as f:
                    previous_value_usd = json.load(f)['value']
            except:
                pass
            
            # Calculate total portfolio value and current allocations
            for token, target_value in initial_allocation.items():
                try:
                    token_info = await self.get_token_info(self.tokens[token])
                    token_price = await self.get_token_price(self.tokens[token])
                    
                    if token_price is None:
                        self.logger.warning(f"Could not get price for {token}")
                        continue
                    # Add inside the token loop, after getting token_price
                    
                    value_usd = float(token_info['balance']) * token_price
                    total_value_usd += value_usd
                    if token_price is not None:
                        # SAFETY CHECK 2: Price deviation check
                        secondary_price = await self.get_token_price(self.tokens[token])  # Second price check
                        if secondary_price is not None:
                            price_deviation = abs(token_price - secondary_price) / token_price * 100
                            if price_deviation > 5:  # 5% deviation threshold
                                alerts.append(f"⚠️ Price discrepancy for {token}: ${token_price:.2f} vs ${secondary_price:.2f}")
                                token_price = min(token_price, secondary_price)  # Use conservative price

                        # SAFETY CHECK 3: Balance anomaly detection
                        if value_usd > target_value * 2:
                            alerts.append(f"🚨 Unusually high balance for {token}: ${value_usd:.2f}")

                        # SAFETY CHECK 4: Sudden value changes
                        if previous_value_usd > 0:
                            value_change_pct = abs(value_usd - previous_value_usd) / previous_value_usd * 100
                            if value_change_pct > 20:  # 20% change threshold
                                alerts.append(f"🚨 Large value change in {token}: {value_change_pct:.1f}%")
                    
                    # Add after calculating total_value_usd
                    # SAFETY CHECK 5: Minimum portfolio value
                    min_portfolio_value = sum(initial_allocation.values()) * 0.5  # 50% of initial allocation
                    if total_value_usd < min_portfolio_value:
                        alerts.append(f"🚨 Portfolio value (${total_value_usd:.2f}) below minimum threshold (${min_portfolio_value:.2f})")
                    
                    
                    current_allocation[token] = {
                        'value_usd': value_usd,
                        'balance': float(token_info['balance']),
                        'price': token_price
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error getting info for {token}: {str(e)}")
                    continue

            if total_value_usd == 0:
                self.logger.error("Could not calculate portfolio value")
                return
            # Add after initial variables
           

            # SAFETY CHECK 1: Verify blockchain connection
            try:
                block = await self.w3.eth.get_block('latest')
                if time.time() - block.timestamp > 600:  # 10 minutes
                    raise Exception("Blockchain data might be stale")
            except Exception as e:
                self.logger.error(f"Blockchain connection issue: {str(e)}")
                return
            
            # Check allocations and generate alerts
            allocation_details = []
            for token, data in current_allocation.items():
                target_value = initial_allocation[token]
                current_pct = (data['value_usd'] / total_value_usd) * 100
                target_pct = (target_value / sum(initial_allocation.values())) * 100
                deviation = abs(current_pct - target_pct)
                
                status_emoji = "🟢" if deviation < 5 else "🟡" if deviation < 15 else "🔴"
                
                allocation_details.append(
                    f"{status_emoji} {token}: ${data['value_usd']:.2f} "
                    f"({current_pct:.1f}% vs target {target_pct:.1f}%)"
                )

                # Alert if significant deviation
                if deviation > 15:  # 15% deviation threshold
                    alerts.append(
                        f"⚠️ {token} allocation off target: {current_pct:.1f}% vs {target_pct:.1f}%"
                    )

                # Alert if balance too low
                min_value = target_value * 0.1  # 10% of target as minimum
                if data['value_usd'] < min_value:
                    alerts.append(
                        f"🚨 Low {token} balance: ${data['value_usd']:.2f} (below ${min_value:.2f})"
                    )
            # Add before sending notifications
            # Save current value for future reference
            try:
                with open('last_portfolio_value.json', 'w') as f:
                    json.dump({'value': total_value_usd, 'timestamp': time.time()}, f)
            except Exception as e:
                self.logger.error(f"Error saving portfolio value: {str(e)}")
            # Prepare notification content
            content = f"""
            <h3>Portfolio Value: ${total_value_usd:.2f}</h3>
            
            <h4>Current Allocations:</h4>
            <ul>
            {''.join(f'<li>{detail}</li>' for detail in allocation_details)}
            </ul>
            """

            # Add alerts if any
            if alerts:
                content += f"""
                <h4>⚠️ Alerts:</h4>
                <ul style="color: {'red' if any('🚨' in alert for alert in alerts) else 'orange'}">
                {''.join(f'<li>{alert}</li>' for alert in alerts)}
                </ul>
                """

            # Daily summary at midnight (within 5 minutes)
            current_hour = datetime.now().hour
            if current_hour == 0:
                # Send full report
                await self.send_status_email(
                    "Daily Portfolio Summary",
                    content,
                    urgent=bool(alerts)
                )
            
            elif alerts:
                # Send alert-only report
                await self.send_status_email(
                    f"Portfolio Alert - {len(alerts)} issue{'s' if len(alerts) > 1 else ''}",
                    content,
                    urgent=True
                )

            # Log summary
            self.logger.info(
                f"\nPortfolio Summary:"
                f"\nTotal Value: ${total_value_usd:.2f}"
                f"\nAlerts: {len(alerts)}"
            )

            for alert in alerts:
                self.logger.warning(alert)

        except Exception as e:
            self.logger.error(f"Error monitoring balances: {str(e)}")
            # Try to send error notification
            try:
                await self.send_status_email(
                    "Balance Monitoring Error",
                    f"Error monitoring portfolio: {str(e)}",
                    urgent=True
                )
            except:
                pass  # Prevent notification errors from causing additional issues
    async def assemble_transaction(self, path_id: str) -> Dict:
        """Assemble transaction from quote"""
        try:
            request_body = {
                "userAddr": self.wallet_address,
                "pathId": path_id,
                "simulate": True
            }
            # Check if we can make a request
            if not self.rate_limiter.can_make_request():
                self.logger.error("Rate limit exceeded. Cannot make more API requests at this time.")
                raise Exception("Rate limit exceeded")

            # Record the API request
            self.rate_limiter.add_request()
            self.logger.info(f"Assembling transaction for path ID: {path_id}")
            response = requests.post(self.ASSEMBLE_URL, json=request_body)
            self.logger.info(f"API Call to {response.url} with status {response.status_code}")
            response.raise_for_status()
            
            assembled = response.json()
            self.logger.info(f"Transaction assembled. Gas estimate: {assembled.get('gasEstimate')}")
            return assembled
            
        except Exception as e:
            self.logger.error(f"Error assembling transaction: {str(e)}")
            raise
            
async def main():
    # Initialize trader
    trader = OdosTrader()
    # Create tracker instance
    tracker = ArbitrageTracker()
    tracker.rebalance_settings.update({
        'min_interval': 3600,  # Adjust time between rebalances
        'threshold': 0.15,     # Adjust balance deviation threshold
        'min_gas_price': 50,   # Set maximum gas price for rebalancing
        'min_liquidity': 1000  # Set minimum liquidity requirement
    })

    # Token addresses
    usdc_address = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"  # USDC on Polygon
    weth_address = "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619"  # WETH on Polygon
    tokens = {
        "USDC": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",  # USDC on Polygon
        "WETH": "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619",  # WETH on Polygon
        "USDT": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",  # USDT on Polygon
        "WBTC": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6"   # WBTC on Polygon   # POL on Polygon
    }
    # Define trading pairs for direct arbitrage
    token_pairs = [
        {"in": tokens["USDC"], "out": tokens["WETH"]},  # Primary pair
        {"in": tokens["USDC"], "out": tokens["WBTC"]},  # Secondary pair
        {"in": tokens["WETH"], "out": tokens["WBTC"]}   # Tertiary pair
    ]
    initial_allocation = {
        "USDC": 800,    # ~$800 (47%) - Main trading token
        "WETH": 600,    # ~$600 (35%) - Primary trade pair
        "WBTC": 300     # ~$300 (18%) - Secondary pair
    }
    # Define triangular arbitrage paths
    triangular_paths = [
        # USDC -> WETH -> WBTC -> USDC
        [tokens["USDC"], tokens["WETH"], tokens["WBTC"]],
        # USDT -> WETH -> WBTC -> USDT
        [tokens["USDT"], tokens["WETH"], tokens["WBTC"]],
        # USDC -> USDT -> WBTC -> USDC
        [tokens["USDC"], tokens["USDT"], tokens["WBTC"]]
    ]
    # Create custom profit thresholds
    thresholds = ProfitThresholds(
        min_profit_usd=0.15,
        min_profit_percentage=0.05,  # 0.2%
        dynamic_gas_multiplier=1.2
    )
    # Trading parameters
    trade_settings = {
        'min_trade_size': 100,         # Lower minimum trade size
        'max_trade_size': 600,         # ~35% of stablecoin balance
        'gas_limit': 80,               # Higher gas limit for more opportunities
        'slippage_tolerance': 0.5,     # 0.5% slippage allowed
        'min_liquidity': 50            # Minimum balance per token
    }
    while True:
        try:
            print("\nOdos Trader Bot Menu:")
            print("1. Swap Tokens")
            print("2. Scan Arbitrage Opportunity")
            print("3. Start Arbitrage Bot")
            print("4. View Wallet Balances")
            print("5. Exit")
            print("6. Performance Analysis")
            print("7. Test Arbitrage Strategies")  # New option for testing
            print("8. Check Token Info")  # New option for checking token info
            print("9. Advanced Settings")
            print("10. Harley's Sandbox")
            answer = input("Select an option: ")

            if answer == "1":
                await trader.show_swap_menu()       
            elif answer == "2":
                try:
                    # Get available balance
                    usdc_info = await trader.get_token_info(usdc_address)
                    max_amount = float(usdc_info['balance'])
                    amount = float(input(f"\nEnter amount to check (max {max_amount:.2f} USDC): "))
                    
                    if amount > max_amount:
                        print(f"Amount exceeds available balance of {max_amount:.2f} USDC")
                        continue
                    
                    # Check for arbitrage opportunities
                    opportunities = await trader.check_arbitrage_opportunity(
                        usdc_address,
                        weth_address,
                        Decimal(str(amount))
                    )
                    
                    print("\nArbitrage Opportunities:")
                    for opp in opportunities['opportunities']:
                        print(f"\nBuy on {opp['buy_dex']}, Sell on {opp['sell_dex']}")
                        print(f"Potential Profit: {opp['profit_percent']:.2f}%")
                        print(f"Estimated Profit (USD): ${opp['estimated_profit_usd']:.2f}")
                        print(f"Gas Cost (USD): ${opp['gas_cost_usd']:.2f}")
                        print(f"Net Profit (USD): ${opp['net_profit_usd']:.2f}")
                        print(f"MATIC Price: ${opp['matic_price_usd']:.4f}")
                
                except Exception as e:
                    print(f"\nError: {str(e)}")
                    
            elif answer == "3":
                print("\n1. Monitor Only (No Execution)")
                print("2. Monitor and Auto-Execute")
                mode = input("Select mode: ")

                print("\nSelect monitoring strategy:")
                print("1. Direct pairs only")
                print("2. Triangular arbitrage only")
                print("3. Both strategies")
                strategy = input("Select strategy: ")

                # Get auto-balancing preferences (keep existing code)
                print("\nAuto-balancing Configuration:")
                enable_balancing = input("Enable auto-balancing? (y/n): ").lower() == 'y'
                
                balance_config = {}
                if enable_balancing:
                    try:
                        ratio = float(input("Enter target balance ratio (0.5 = 50/50 split): ") or 0.5)
                        threshold = float(input("Enter rebalance threshold (0.05 = 5% deviation): ") or 0.05)
                        balance_config = {
                            "target_balance_ratio": max(0.1, min(0.9, ratio)),
                            "rebalance_threshold": max(0.01, min(0.2, threshold))
                        }
                    except ValueError:
                        print("Invalid input, using default values (50/50 split, 5% threshold)")
                        balance_config = {
                            "target_balance_ratio": 0.5,
                            "rebalance_threshold": 0.05
                        }

                # Determine which pairs to monitor based on strategy
                monitoring_pairs = []
                if strategy in ["1", "3"]:
                    monitoring_pairs.extend(token_pairs)
                if strategy in ["2", "3"]:
                    # Add triangular paths
                    for path in triangular_paths:
                        for i in range(len(path)):
                            monitoring_pairs.append({
                                "in": path[i],
                                "out": path[(i + 1) % len(path)],
                                "is_triangular": True,
                                "full_path": path
                            })

                if mode == "1":
                    print("\nStarting monitor only mode... Press Ctrl+C to stop")
                    await trader.monitor_arbitrage_opportunities(
                        token_pairs=monitoring_pairs,
                        profit_thresholds=thresholds,
                        check_interval=1.5,
                        max_gas_gwei=50,
                        amount_percentage=0.95,
                        **balance_config
                    )
                elif mode == "2":
                    print("\nStarting monitor and execute mode... Press Ctrl+C to stop")
                    print("Auto-balancing:", "Enabled" if enable_balancing else "Disabled")
                    if enable_balancing:
                        print(f"Target ratio: {balance_config['target_balance_ratio']:.1%}")
                        print(f"Rebalance threshold: {balance_config['rebalance_threshold']:.1%}")
                    
                    await trader.monitor_and_execute_arbitrage(
                        token_pairs=monitoring_pairs,
                        profit_thresholds=thresholds,
                        check_interval=2,
                        max_gas_gwei=80,
                        amount_percentage=0.95,
                        min_auto_execute_profit=0.1,
                        enable_balancing=enable_balancing,
                        initial_allocation=initial_allocation,  # Add this
                        trade_settings=trade_settings,         # Add this
                        **balance_config
                    )         
            elif answer == "4":
                # Show current balances and prices
                usdc_info = await trader.get_token_info(usdc_address)
                weth_info = await trader.get_token_info(weth_address)
                usdt_info = await trader.get_token_info(tokens["USDT"])
                wbtc_info = await trader.get_token_info(tokens["WBTC"])
                matic_price = await trader.get_matic_price()
                
                print(f"\nCurrent Balances and Prices:")
                print(f"USDC: {usdc_info['balance']:.2f} {usdc_info['symbol']}")
                print(f"WETH: {weth_info['balance']:.6f} {weth_info['symbol']}")
                print(f"USDT: {usdt_info['balance']:.2f} {usdt_info['symbol']}")
                print(f"WBTC: {wbtc_info['balance']:.8f} {wbtc_info['symbol']}")
                print(f"Current MATIC Price: ${matic_price:.4f}")
                print(f"Current Gas Price: {trader.w3.eth.gas_price / 1e9:.1f} gwei")
                
                # Calculate sample gas cost
                gas_cost = await trader.calculate_gas_cost_usd(100000)  # Sample gas amount
                print(f"Sample Gas Cost (100k gas): ${gas_cost:.4f}")
            elif answer == "5":
                print("\nExiting...")
                break       
            elif answer == "6":
                print("\nBot Performance Analysis")
                print("1. Last 24 hours")
                print("2. Last 7 days")
                print("3. Last 30 days")
                print("4. Last year")
                print("5. Custom period")
                
                period = input("Select period: ")
                
                if period == "1":
                    days = 1
                elif period == "2":
                    days = 7
                elif period == "3":
                    days = 30
                elif period == "4":
                    days = 365
                elif period == "5":
                    days = int(input("Enter number of days to analyze: "))
                else:
                    print("Invalid selection")
                    continue
                    
                results = trader.analyze_bot_performance(days)
                
                if 'status' in results:
                    print(f"\n{results['status']}")
                else:
                    print(f"\nPerformance Analysis ({results['actual_days']} days)")
                    print(f"Total Trades: {results['total_trades']}")
                    print(f"Total Profit: ${results['total_profit_usd']}")
                    print(f"Total Gas Cost: ${results['total_gas_cost_usd']}")
                    print(f"Net Profit: ${results['net_profit_usd']}")
                    print(f"Average Profit/Trade: ${results['average_per_trade_usd']}")
                    print(f"Average Profit/Day: ${results['average_per_day_usd']}")
                    print(f"Average Profit/Month: ${results['average_per_month_usd']}")
                    print(f"Trades per Day: {results['trades_per_day']}")
                    print(f"Gas Efficiency: {results['gas_efficiency']}x")           
            elif answer == "7":
                print("\nTriangular Arbitrage Test Menu:")
                print("1. Check single triangular path")
                print("2. Monitor all triangular paths")
                
                test_mode = input("Select test mode: ")
                
                if test_mode == "1":
                    print("\nAvailable paths:")
                    for i, path in enumerate(triangular_paths, 1):
                        path_symbols = [
                            (await trader.get_token_info(addr))['symbol']
                            for addr in path
                        ]
                        print(f"{i}. {' -> '.join(path_symbols)}")
                    
                    path_choice = int(input("Select path to test: ")) - 1
                    test_path = triangular_paths[path_choice]
                    
                    # Get initial token info
                    initial_token = await trader.get_token_info(test_path[0])
                    max_amount = float(initial_token['balance'])
                    
                    print(f"\nCurrent balance: {max_amount:.6f} {initial_token['symbol']}")
                    amount = float(input(f"Enter amount to test (max {max_amount:.6f}): "))
                    
                    if amount > max_amount:
                        print(f"Amount exceeds available balance")
                        continue
                    
                    # Test opportunity
                    result = await trader.check_triangular_opportunity(
                        test_path,
                        Decimal(str(amount))
                    )
                    
                    if result and result['net_profit_usd'] > 0:
                        print("\nProfitable opportunity found!")
                        execute = input("Execute trade? (y/n): ").lower()
                        if execute == 'y':
                            await trader.execute_triangular_trade(
                                result,
                                amount_percentage=0.95,
                                slippage=0.5
                            )
                
                elif test_mode == "2":
                    print("\nStarting triangular arbitrage monitor...")
                    monitoring_pairs = []
                    for path in triangular_paths:
                        for i in range(len(path)):
                            monitoring_pairs.append({
                                "in": path[i],
                                "out": path[(i + 1) % len(path)],
                                "is_triangular": True,
                                "full_path": path
                            })
                    
                    await trader.monitor_and_execute_arbitrage(
                        token_pairs=monitoring_pairs,
                        profit_thresholds=thresholds,
                        check_interval=2,
                        max_gas_gwei=80,
                        amount_percentage=0.95,
                        min_auto_execute_profit=0.1,
                        enable_balancing=enable_balancing,
                        initial_allocation=initial_allocation,  # Add this
                        trade_settings=trade_settings,         # Add this
                        **balance_config
                    )           
            elif answer == "8":
                print("\nAvailable tokens:")
                for i, (symbol, address) in enumerate(tokens.items(), 1):
                    print(f"{i}. {symbol}")
                
                token_choice = int(input("Select a token to check info: ")) - 1
                token_symbol = list(tokens.keys())[token_choice]
                token_address = tokens[token_symbol]
                
                # Fetch and display token info
                token_info = await trader.get_token_info(token_address)
                
                print(f"\nToken Information for {token_symbol} ({token_address}):")
                print(f"Decimals: {token_info['decimals']}")
                print(f"Balance: {token_info['balance']} {token_info['symbol']}")
                print(f"Balance (Wei): {token_info['balance_wei']}")
                print(f"Allowance: {token_info['allowance']}")           
            elif answer == "9":
                await trader.show_advanced_menu()
            elif answer == "10":
                print("\nHarley's Sandbox:")
                print("1. Option 1 ")
                print("2. Option 2 ")
                print("3. Exit")
                
                sandbox_choice = input("Select option: ")
                
                if sandbox_choice == "1":
                    try:
                        # Harley can put bot shit here
                        pass
                    
                    except Exception as e:
                        print(f"\nError: {str(e)}")
                        
                elif sandbox_choice == "2":
                    try:
                        # Homies shit
                        pass
                    except Exception as e:
                        print(f"\nError: {str(e)}")

                elif sandbox_choice == "3":
                    break
            else:
                print("\nInvalid option selected")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            continue
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    asyncio.run(main())