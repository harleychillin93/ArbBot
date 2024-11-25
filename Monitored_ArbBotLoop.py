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

network_symbol = "poly"
token1_symbol = "WETH"
token2_symbol = "USDCe"

chainID = os.getenv(network_symbol + "_" + "chainID")

# Token 1 info
tkn1_ca = os.getenv(network_symbol + "_" + token1_symbol + "_" + "ca")
tkn1_abi = os.getenv(network_symbol + "_" + token1_symbol + "_" + "abi")
tkn1_dec = int(os.getenv(network_symbol + "_" + token1_symbol + "_" + "dec"))

# Token 2 info
tkn2_ca = os.getenv(network_symbol + "_" + token2_symbol + "_" + "ca")
tkn2_abi = os.getenv(network_symbol + "_" + token2_symbol + "_" + "abi")
tkn2_dec = int(os.getenv(network_symbol + "_" + token2_symbol + "_" + "dec"))

# Define RPC
RPCURL = os.getenv(network_symbol + "RPCURL")
# Connect RPC
web3 = Web3(Web3.HTTPProvider(RPCURL))

# Save Portfolio at Startup
port_df = log_port_event(Port_Dir, port_df, pubaddress, RPCURL, chainID, event ='Script Started')



# Create token contract instance
token1_contract = web3.eth.contract(address=tkn1_ca, abi=tkn1_abi)
# Use contract instance to get balance of wallet in token's smallest unit (like "wei" for ETH)
balance = token1_contract.functions.balanceOf(pubaddress).call()
# Get the number of decimals to convert balance to a human-readable format
token1_balance = balance / (10 ** tkn1_dec)
print(f"Balance of token 1 at {tkn1_ca} in wallet {pubaddress}: {token1_balance}")


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

quote_info = SOR_Quote(tkn1_ca, tkn1_dec, token1_balance, tkn2_ca, tkn2_dec, chainID, pubaddress)

odos_quote = quote_info[0]
human_quote = quote_info[1]


DolVal_1 = float(human_quote)
print("Dollar Value of is" , DolVal_1)

## Youve got the dollar balance of 1, get $ balance of 2
# Create token contract instance
token2_contract = web3.eth.contract(address=tkn2_ca, abi=tkn2_abi)
# Use contract instance to get balance of wallet in token's smallest unit (like "wei" for ETH)
balance = token2_contract.functions.balanceOf(pubaddress).call()
# Get the number of decimals to convert balance to a human-readable format
token2_balance = balance / (10 ** tkn2_dec)
print(f"Balance of token at {tkn2_ca} in wallet {pubaddress}: {token2_balance}")
print("Dollar Value of token 2" , token2_balance)

# So now you know what avaialble liquidty you have. You never want to trade with more than 50% of the lowest balance per pair
lowest_liq = min([DolVal_1, token2_balance])
trade_size = math.ceil(lowest_liq / 2)

i=1
txn1_list = []
txn2_list = []

while i <= 1:
  price_of_gas = 1000
  arb_op = 0
  while arb_op <= 0.018:
    sleep(3)

    temp = int(web3.eth.gas_price * 12 // 10)
    price_of_gas = web3.from_wei(temp, 'gwei') 
    
    if price_of_gas > 160:
      print(f"Gas price of {price_of_gas} Gwei is too high")
    else:
      print(f"Gas price of {price_of_gas} Gwei is appropriate")

      # Want to buy exactly trade size, want to sell the exact ammount of other token to return trade_size
      try: 
        quote = SOR_Quote(tkn2_ca, tkn2_dec, trade_size, tkn1_ca, tkn1_dec, chainID, pubaddress)
        buy_quote = quote[0]
        price1 = quote[1]
        token1_ToRecieve = int(buy_quote['outAmounts'][0]) / (10 ** tkn1_dec)
      except:
        print("Arb Quote 1 not returned")
        print(ValueError)  
      

      try: 
        quote = SOR_Quote(tkn1_ca, tkn1_dec, token1_ToRecieve, tkn2_ca, tkn2_dec, chainID, pubaddress)
        sell_quote = quote[0]
        price2= quote[1]
        token2_ToRecieve = int(sell_quote['outAmounts'][0]) / (10 ** tkn2_dec)
      except:
        print("Arb Quote 2 not returned")
        print(ValueError) 
      
      arb_op = (token2_ToRecieve-trade_size)/trade_size
      print(f"Current Arb_op is {arb_op} percent")


      # If code makes it to here theres and arb opportunity and its good gas prices rn so make fresh quote and buy/sell

  
  Assembled_Buy = SOR_Assemble(buy_quote, pubaddress)
  nonce = Assembled_Buy['transaction']['nonce']
  buy_txn_hash = SOR_SendTxn(Assembled_Buy, RPCURL, prikey)
  print(buy_txn_hash)
  
  
  Assembled_Sell = SOR_Assemble(sell_quote, pubaddress)
  Assembled_Sell['transaction']['nonce'] = nonce+1
  sell_txn_hash = SOR_SendTxn(Assembled_Sell, RPCURL, prikey)
  print(sell_txn_hash)

  log_df = log_trade_event(log_df, 'Buy Side Arb', buy_txn_hash, "USDCe", trade_size, "WETH", price1, web3.eth.gas_price, SOR_GetGas4Quote(buy_quote, RPCURL, pubaddress))
  log_df = log_trade_event(log_df, 'Sell Side Arb', sell_txn_hash, "WETH", token1_ToRecieve, "USDCe", price2, web3.eth.gas_price, SOR_GetGas4Quote(sell_quote, RPCURL, pubaddress))
  # Function to save the log DataFrame to a CSV file
  save_trade_log(Log_Dir, log_df)

  arb_op = 0
  i = i + 1
  sleep(30)
  port_df = log_port_event(Port_Dir, port_df, pubaddress, RPCURL, chainID, event ='Arb Attempted')



