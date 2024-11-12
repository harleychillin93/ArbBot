from web3 import Web3 
from eth_account import Account
from dotenv import load_dotenv
import os

load_dotenv()

# Security
prikey = os.getenv("MM_PriKey")
pubaddress = os.getenv("MM_PubAddr")
account = Account.from_key(prikey)

# Token 1 info
WETH_ca_poly = os.getenv("poly_WETH_ca")
WETH_abi_poly = os.getenv("poly_WETH_abi")
WETH_dec_poly = int(os.getenv("poly_WETH_dec"))

# Token 2 info
USDC_ca_poly = os.getenv("poly_USDC_ca")
USDC_abi_poly = os.getenv("poly_USDC_abi")
USDC_dec_poly = int(os.getenv("poly_USDC_dec"))

# Define RPC
polyRPCURL = os.getenv("polyRPCURL")
# Connect RPC
poly_web3 = Web3(Web3.HTTPProvider(polyRPCURL))
# Create token contract instance
token1_contract = poly_web3.eth.contract(address=WETH_ca_poly, abi=WETH_abi_poly)
# Use contract instance to get balance of wallet in token's smallest unit (like "wei" for ETH)
balance = token1_contract.functions.balanceOf(pubaddress).call()
# Get the number of decimals to convert balance to a human-readable format
token1_balance = balance / (10 ** WETH_dec_poly)
print(f"Balance of token at {WETH_ca_poly} in wallet {pubaddress}: {token1_balance}")


# At this point you can check a balance, now whats the usdc value of that balance, use odos
import json
import requests
import math
from time import sleep
from hexbytes import HexBytes


quote_url = "https://api.odos.xyz/sor/quote/v2"

quote1_request_body = {
    "chainId": 137, # Replace with desired chainId
    "inputTokens": [
        {
            "tokenAddress": WETH_ca_poly, # checksummed input token address
            "amount": str(int(token1_balance * (10 ** WETH_dec_poly))) , # input amount as a string in fixed integer precision
        }
    ],
"outputTokens": [
        {
            "tokenAddress": USDC_ca_poly, # checksummed output token address
            "proportion": 1
        }
    ],
    "slippageLimitPercent": 0.7, # set your slippage limit percentage (1 = 1%)
    "userAddr": pubaddress, # checksummed user address
    "referralCode": 0, # referral code (recommended)
    "disableRFQs": True,
    "compact": True,
}

response1 = requests.post(
  quote_url,
  headers={"Content-Type": "application/json"},
  json=quote1_request_body
)

if response1.status_code == 200:
  quote = response1.json()
  # handle quote response data
else:
  print(f"Error in Quote: {response1.json()}")
  # handle quote failure cases

DolVal_1 = int(quote['outAmounts'][0]) / (10 ** USDC_dec_poly)
print("Dollar Value of is" , DolVal_1)

## Youve got the dollar balance of 1, get $ balance of 2
# Create token contract instance
token2_contract = poly_web3.eth.contract(address=USDC_ca_poly, abi=USDC_abi_poly)
# Use contract instance to get balance of wallet in token's smallest unit (like "wei" for ETH)
balance = token2_contract.functions.balanceOf(pubaddress).call()
# Get the number of decimals to convert balance to a human-readable format
token2_balance = balance / (10 ** USDC_dec_poly)
print(f"Balance of token at {USDC_ca_poly} in wallet {pubaddress}: {token2_balance}")
print("Dollar Value of token 2" , token2_balance)

# So now you know what avaialble liquidty you have. You never want to trade with more than 50% of the lowest balance per pair
lowest_liq = min([DolVal_1, token2_balance])
trade_size = math.ceil(lowest_liq / 2)

arb_op = 0



while arb_op <= 0.007:
  sleep(5)
  # Now we know the trade size. Lets get a buy side quote and a sell side quote

  # Want to buy exactly trade size, want to sell the exact ammount of other token to return trade_size

  quote2_request_body = {
      "chainId": 137, # Replace with desired chainId
      "inputTokens": [
          {
              "tokenAddress": USDC_ca_poly, # checksummed input token address
              "amount": str(int(trade_size * (10 ** USDC_dec_poly))) , # input amount as a string in fixed integer precision
          }
      ],
      "outputTokens": [
          {
              "tokenAddress": WETH_ca_poly, # checksummed output token address
              "proportion": 1
          }
      ],
      "slippageLimitPercent": 0.7, # set your slippage limit percentage (1 = 1%)
      "userAddr": pubaddress, # checksummed user address
      "referralCode": 0, # referral code (recommended)
      "disableRFQs": True,
      "compact": True,
  }

  response2 = requests.post(
    quote_url,
    headers={"Content-Type": "application/json"},
    json=quote2_request_body
  )

  if response2.status_code == 200:
    buy_quote = response2.json()
    # handle quote response data
  else:
    print(f"Error in Quote: {response2.json()}")
    # handle quote failure cases

  token1_quoted = int(buy_quote['outAmounts'][0]) / (10 ** WETH_dec_poly)

  
  quote3_request_body = {
      "chainId": 137, # Replace with desired chainId
      "inputTokens": [
          {
              "tokenAddress": WETH_ca_poly, # checksummed input token address
              "amount": str(int(token1_quoted * (10 ** WETH_dec_poly))) , # input amount as a string in fixed integer precision
          }
      ],
      "outputTokens": [
          {
              "tokenAddress": USDC_ca_poly, # checksummed output token address
              "proportion": 1
          }
      ],
      "slippageLimitPercent": 0.7, # set your slippage limit percentage (1 = 1%)
      "userAddr": pubaddress, # checksummed user address
      "referralCode": 0, # referral code (recommended)
      "disableRFQs": True,
      "compact": True,
  }

  response3 = requests.post(
    quote_url,
    headers={"Content-Type": "application/json"},
    json=quote3_request_body
  )

  if response3.status_code == 200:
    sell_quote = response3.json()
    # handle quote response data
  else:
    print(f"Error in Quote: {response3.json()}")
    # handle quote failure cases

  token2_quoted = int(sell_quote['outAmounts'][0]) / (10 ** USDC_dec_poly)

  arb_op = (token2_quoted-trade_size)/trade_size
  print(arb_op)

# If code makes it to here theres and arb opportunity so use the same buy_quote and sell quote we just made



assemble_url = "https://api.odos.xyz/sor/assemble"

# Buy side
assemble_request_body1 = {
    "userAddr": pubaddress, # the checksummed address used to generate the quote
    "pathId": buy_quote["pathId"], # Replace with the pathId from quote response in step 1
    "simulate": False, # this can be set to true if the user isn't doing their own estimate gas call for the transaction
}

response1 = requests.post(
  assemble_url,
  headers={"Content-Type": "application/json"},
  json=assemble_request_body1
)

if response1.status_code == 200:
  assembled_transaction1 = response1.json()
  # handle Transaction Assembly response data
else:
  print(f"Error in Transaction Assembly: {response1.json()}")
  # handle Transaction Assembly failure cases



# Send TXN 1
transaction1 = assembled_transaction1["transaction"]
# web3py requires the value to be an integer
transaction1["value"] = int(transaction1["value"])
signed_tx = poly_web3.eth.account.sign_transaction(transaction1, prikey)
# 4. Send the signed transaction
tx_hash1 = poly_web3.eth.send_raw_transaction(signed_tx.raw_transaction)
print("txn1")
print(tx_hash1.hex())


# Sell side
assemble_request_body2 = {
    "userAddr": pubaddress, # the checksummed address used to generate the quote
    "pathId": sell_quote["pathId"], # Replace with the pathId from quote response in step 1
    "simulate": False, # this can be set to true if the user isn't doing their own estimate gas call for the transaction
}

response2 = requests.post(
  assemble_url,
  headers={"Content-Type": "application/json"},
  json=assemble_request_body2
)

if response2.status_code == 200:
  assembled_transaction2 = response2.json()
  # handle Transaction Assembly response data
else:
  print(f"Error in Transaction Assembly: {response2.json()}")
  # handle Transaction Assembly failure cases

# Send TXN 2
transaction2 = assembled_transaction2["transaction"]
# web3py requires the value to be an integer
transaction2["value"] = int(transaction2["value"])
transaction2["nonce"] = int(transaction2["nonce"])+1
signed_tx = poly_web3.eth.account.sign_transaction(transaction2, prikey)
# 4. Send the signed transaction
tx_hash2 = poly_web3.eth.send_raw_transaction(signed_tx.raw_transaction)
print("txn2")
print(tx_hash2.hex())
