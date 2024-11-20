from web3 import Web3 
from eth_account import Account
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime
from ArbBotFuncs import get_trade_log_file_name
from ArbBotFuncs import init_trade_log_file
from ArbBotFuncs import log_trade_event
from ArbBotFuncs import save_trade_log
from ArbBotFuncs import init_port_log_file
from ArbBotFuncs import log_port_event

load_dotenv()

# Define the trade log file directory
Log_Dir = "trade_logs"
# Ensure the log directory exists
os.makedirs(Log_Dir, exist_ok=True)
# Generate today's log file name based on the date
log_df = init_trade_log_file(Log_Dir)
# Save Note
log_df = log_trade_event(log_df, 
                   'Script Started')
# Function to save the log DataFrame to a CSV file
save_trade_log(Log_Dir, log_df)

# Define the portfolio log file directory
Port_Dir = "portfolio_logs"
# Ensure the log directory exists
os.makedirs(Port_Dir, exist_ok=True)
# Generate today's log file name based on the date
port_df = init_port_log_file(Port_Dir)


# Security
prikey = os.getenv("MM_PriKey")
pubaddress = os.getenv("MM_PubAddr")
account = Account.from_key(prikey)

# Token 1 info
tkn1_ca_poly = os.getenv("poly_WETH_ca")
tkn1_abi_poly = os.getenv("poly_WETH_abi")
tkn1_dec_poly = int(os.getenv("poly_WETH_dec"))

# Token 2 info
tkn2_ca_poly = os.getenv("poly_USDCe_ca")
tkn2_abi_poly = os.getenv("poly_USDCe_abi")
tkn2_dec_poly = int(os.getenv("poly_USDCe_dec"))

# Define RPC
polyRPCURL = os.getenv("polyRPCURL")
# Connect RPC
poly_web3 = Web3(Web3.HTTPProvider(polyRPCURL))

# Save Portfolio at Startup
port_df = log_port_event(Port_Dir, port_df, pubaddress, polyRPCURL, event ='Script Started')



# Create token contract instance
token1_contract = poly_web3.eth.contract(address=tkn1_ca_poly, abi=tkn1_abi_poly)
# Use contract instance to get balance of wallet in token's smallest unit (like "wei" for ETH)
balance = token1_contract.functions.balanceOf(pubaddress).call()
# Get the number of decimals to convert balance to a human-readable format
token1_balance = balance / (10 ** tkn1_dec_poly)
print(f"Balance of token 1 at {tkn1_ca_poly} in wallet {pubaddress}: {token1_balance}")


# At this point you can check a balance, now whats the usdc value of that balance, use odos
import json
import requests
import math
from time import sleep
from hexbytes import HexBytes
from ArbBotFuncs import SOR_Quote
from ArbBotFuncs import SOR_Assemble
from ArbBotFuncs import SOR_SendTxn
from ArbBotFuncs import SOR_GetGas4Quote

quote_info = SOR_Quote(tkn1_ca_poly, tkn1_dec_poly, token1_balance, tkn2_ca_poly, tkn2_dec_poly, pubaddress)

odos_quote = quote_info[0]
human_quote = quote_info[1]


DolVal_1 = float(human_quote)
print("Dollar Value of is" , DolVal_1)

## Youve got the dollar balance of 1, get $ balance of 2
# Create token contract instance
token2_contract = poly_web3.eth.contract(address=tkn2_ca_poly, abi=tkn2_abi_poly)
# Use contract instance to get balance of wallet in token's smallest unit (like "wei" for ETH)
balance = token2_contract.functions.balanceOf(pubaddress).call()
# Get the number of decimals to convert balance to a human-readable format
token2_balance = balance / (10 ** tkn2_dec_poly)
print(f"Balance of token at {tkn2_ca_poly} in wallet {pubaddress}: {token2_balance}")
print("Dollar Value of token 2" , token2_balance)

# So now you know what avaialble liquidty you have. You never want to trade with more than 50% of the lowest balance per pair
lowest_liq = min([DolVal_1, token2_balance])
trade_size = math.ceil(lowest_liq / 3)

arb_op = 0
i=1
txn1_list = []
txn2_list = []

while i <= 1:

  while arb_op <= 0.012:
    sleep(4)
    # Now we know the trade size. Lets get a buy side quote and a sell side quote

    # Want to buy exactly trade size, want to sell the exact ammount of other token to return trade_size
    quote = SOR_Quote(tkn2_ca_poly, tkn2_dec_poly, trade_size, tkn1_ca_poly, tkn1_dec_poly, pubaddress)
    buy_quote = quote[0]
    token1_ToRecieve = int(buy_quote['outAmounts'][0]) / (10 ** tkn1_dec_poly)

    quote = SOR_Quote(tkn1_ca_poly, tkn1_dec_poly, token1_ToRecieve, tkn2_ca_poly, tkn2_dec_poly, pubaddress)
    sell_quote = quote[0]
    token2_ToRecieve = int(sell_quote['outAmounts'][0]) / (10 ** tkn2_dec_poly)

    arb_op = (token2_ToRecieve-trade_size)/trade_size
    print(arb_op)

  # If code makes it to here theres and arb opportunity so use the same buy_quote and sell quote we just made

  quote = SOR_Quote(tkn2_ca_poly, tkn2_dec_poly, trade_size, tkn1_ca_poly, tkn1_dec_poly, pubaddress)
  price1 = quote[1]
  buy_quote = quote[0]
  Assembled_Buy = SOR_Assemble(buy_quote, pubaddress)
  nonce = Assembled_Buy['transaction']['nonce']
  buy_txn_hash = SOR_SendTxn(Assembled_Buy, polyRPCURL, prikey)
  print(buy_txn_hash)
  
  quote = SOR_Quote(tkn1_ca_poly, tkn1_dec_poly, token1_ToRecieve, tkn2_ca_poly, tkn2_dec_poly, pubaddress)
  price2 = quote[1]
  sell_quote = quote[0]
  Assembled_Sell = SOR_Assemble(sell_quote, pubaddress)
  Assembled_Sell['transaction']['nonce'] = nonce+1
  sell_txn_hash = SOR_SendTxn(Assembled_Sell, polyRPCURL, prikey)
  print(sell_txn_hash)

  network = 'poly'; # could be any supported network
  key = os.getenv("Owlracle_Key"); # fill your api key here
  res = requests.get('https://api.owlracle.info/v4/{}/gas?apikey={}'.format(network, key))
  data = res.json()
  print(data)


  log_df = log_trade_event(log_df, 'Buy Side Arb', buy_txn_hash, "USDCe", trade_size, "WETH", price1, poly_web3.eth.gas_price, SOR_GetGas4Quote(buy_quote, polyRPCURL, pubaddress))
  log_df = log_trade_event(log_df, 'Sell Side Arb', sell_txn_hash, "WETH", token1_ToRecieve, "USDCe", price2, poly_web3.eth.gas_price, SOR_GetGas4Quote(sell_quote, polyRPCURL, pubaddress))
  # Function to save the log DataFrame to a CSV file
  save_trade_log(Log_Dir, log_df)

  arb_op = 0
  i = i + 1
  sleep(30)
  port_df = log_port_event(Port_Dir, port_df, pubaddress, polyRPCURL, event ='Arb Attempted')



