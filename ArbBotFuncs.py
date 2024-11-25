import requests
import math
from time import sleep
from hexbytes import HexBytes
from web3 import Web3 
from eth_account import Account
from dotenv import load_dotenv
import os
from web3.middleware import PythonicMiddleware
from datetime import datetime
import pandas as pd
from datetime import datetime



# Old code to get ABI and Decimals, not maintained
def polyGetTokenOrProxyAbi(contractAddress: str, RPCURL: str):
    beginTemplate = 'https://api.polygonscan.com/api?module=contract&action=getabi&address='
    templateWithContract = beginTemplate+contractAddress
    apiKeyTemplate = '&apikey='
    apiKeyToken = '6PRAUJAV1RVRP6Q78GVR9RUG6CINCFQFYT'
    endTemplate = apiKeyTemplate+apiKeyToken
    fullAPICallURL = templateWithContract+endTemplate
    fullAPIResponse = requests.get(fullAPICallURL)
    temp=fullAPIResponse.json()
    response = temp["result"]
    if 'decimals' in response:
        return response
    elif 'proxy' in response:
        web3Poly = Web3(Web3.HTTPProvider(RPCURL))
        proxyContract = web3Poly.eth.contract(contractAddress, abi=response)
        proxyOwnerAddress = proxyContract.functions.implementation().call()      
        beginTemplate = 'https://api.polygonscan.com/api?module=contract&action=getabi&address='
        templateWithContract = beginTemplate+proxyOwnerAddress
        apiKeyTemplate = '&apikey='
        apiKeyToken = '6PRAUJAV1RVRP6Q78GVR9RUG6CINCFQFYT'
        endTemplate = apiKeyTemplate+apiKeyToken
        fullAPICallURL = templateWithContract+endTemplate
        fullAPIResponse = requests.get(fullAPICallURL)
        temp=fullAPIResponse.json()
        response = temp["result"]
        return response
    else: return response

def polyGetDecimals(Address: str, RPCURL: str):
    web3Poly = Web3(Web3.HTTPProvider(RPCURL))
    web3Poly.middleware_onion.inject(PythonicMiddleware, layer=0)
    contractAddr = Web3.to_checksum_address(Address)
    Abi = polyGetTokenOrProxyAbi(contractAddr,RPCURL)
    tokenContract = web3Poly.eth.contract(contractAddr, abi=Abi)
    return tokenContract.functions.decimals().call()


# Standard Order Through Odos
def SOR_Quote(tokenInAdress: str, tknInDecimals: int, ammount: str, tokenOutAddress: str, tknOutDecimals: int, chainID: int, pubaddress: str):
    quote_url = "https://api.odos.xyz/sor/quote/v2"

    quote_request_body = {
        "chainId": chainID, # Replace with desired chainId
        "inputTokens": [
            {
                "tokenAddress": tokenInAdress, # checksummed input token address
                "amount": str(int(ammount * (10 ** tknInDecimals))) , # input amount as a string in fixed integer precision
            }
        ],
    "outputTokens": [
            {
                "tokenAddress": tokenOutAddress, # checksummed output token address
                "proportion": 1
            }
        ],
        "slippageLimitPercent": 0.6, # set your slippage limit percentage (1 = 1%)
        "userAddr": pubaddress, # checksummed user address
        "referralCode": 0, # referral code (recommended)
        "disableRFQs": True,
        "compact": True,
    }

    response = requests.post(
    quote_url,
    headers={"Content-Type": "application/json"},
    json=quote_request_body
    )

    if response.status_code == 200:
        odos_quote = response.json()
        human_quote = str(int(odos_quote['outAmounts'][0]) / (10 ** tknOutDecimals))
        return [odos_quote, human_quote]
        # handle quote response data
    else:
        print(f"Error in Quote: {response.json()}")
        # handle quote failure cases

def SOR_Assemble(odos_quote: str, pubaddress: str):
  assemble_url = "https://api.odos.xyz/sor/assemble"
  
  assemble_request_body = {
     "userAddr": pubaddress, # the checksummed address used to generate the quote
     "pathId": odos_quote["pathId"], # Replace with the pathId from quote response in step 1
     "simulate": False, # this can be set to true if the user isn't doing their own estimate gas call for the transaction
     }
  
  response = requests.post(
     assemble_url,
     headers={"Content-Type": "application/json"},
     json=assemble_request_body
     )
  
  if response.status_code == 200:
     assembled_transaction = response.json()
     return assembled_transaction
     # handle Transaction Assembly response data
  else:
    print(f"Error in Transaction Assembly: {response.json()}")
    # handle Transaction Assembly failure cases

def SOR_SendTxn(assembled_transaction: str, RPCURL: str, prikey: str):
  web3 = Web3(Web3.HTTPProvider(RPCURL))

  # Send TXN 
  transaction = assembled_transaction["transaction"]
  # web3py requires the value to be an integer
  transaction["value"] = int(transaction["value"])
  try:
    transaction["gas"] = int(web3.eth.estimate_gas(transaction)  * 12 // 10)
    transaction["gasPrice"] = int(web3.eth.gas_price * 12 // 10)
  except:
     print("Gas calcs failed")
  
  signed_tx = web3.eth.account.sign_transaction(transaction, prikey)
  # 4. Send the signed transaction
  tx_hash1 = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
  return tx_hash1.hex()

def SOR_GetGas4Quote(odos_quote: str, RPCURL: str, pubaddress: str):
    assembled = SOR_Assemble(odos_quote, pubaddress)
    web3 = Web3(Web3.HTTPProvider(RPCURL))
    # Send TXN 1
    transaction = assembled["transaction"]
    # web3py requires the value to be an integer
    transaction["value"] = int(transaction["value"])
    return web3.eth.estimate_gas(transaction)


# Trade Logging Functions
def get_trade_log_file_name(Log_Dir):
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(Log_Dir, f"trades_{today}.csv")

def init_trade_log_file(Log_Dir):
   current_filename = get_trade_log_file_name(Log_Dir)
   if os.path.isfile(current_filename) == False:
   # Initialize a DataFrame with predefined headers
    columns = ["Timestamp", "Event", "Trade ID", "Symbolin", "Quantity", "SymbolOut", "Price", "GasPrice", "GasEstimate"]
    log_df = pd.DataFrame(columns=columns)
    row = {
         "Timestamp": "init",
         "Event": "init",
         "Trade ID": "init",
         "SymbolIn": "init",
         "Quantity": "init",
         "SymbolOut": "init",
         "Price": "init",
         "GasPrice": "init",
         "GasEstimate": "init",
     }
   # Append the row to the DataFrame
    log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    log_df.to_csv(current_filename, index=False)
    save_trade_log(Log_Dir, log_df)
    return log_df
   else:
       log_df = pd.read_csv(current_filename)
       save_trade_log(Log_Dir, log_df)
       return log_df
   
def log_trade_event(log_df, event, trade_id=None, symbolin=None, quantity=None, symbolout=None, price=None, GasPrice=None, GasEstimate=None):
    
    # Create a new row with the data
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Event": event,
        "Trade ID": trade_id,
        "SymbolIn": symbolin,
        "Quantity": quantity,
        "SymbolOut": symbolout,
        "Price": price,
        "GasPrice": GasPrice,
        "GasEstimate": GasEstimate,
    }
    
    # Append the row to the DataFrame
    log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    return log_df

def save_trade_log(Log_Dir, log_df):
    filename = get_trade_log_file_name(Log_Dir)
    log_df.to_csv(filename, index=False)
    print(f"Log saved to {log_df}")



# Portfolio Logging Functions
def get_port_log_file_name(Port_Dir):
   today = datetime.now().strftime("%Y-%m-%d")
   return os.path.join(Port_Dir, f"Portfolio_{today}.csv")

def init_port_log_file(Port_Dir):
   current_filename = get_port_log_file_name(Port_Dir)
   if os.path.isfile(current_filename) == False:
   # Initialize a DataFrame with predefined headers
    columns = ["Timestamp", "Event", "Coin1", "Price1", "Amount1", "Coin2", "Price2", "Amount2", "Coin3", "Price3", "Amount3", "Coin4", "Price4", "Amount4"]
    port_df = pd.DataFrame(columns=columns)
    row = {
         "Timestamp": "init",
         "Event": "init",
         "Coin1": "WETH",
         "Price1": "init",
         "Amount1": "init",
         "Coin2": "WBTC",
         "Price2": "init",
         "Amount2": "init",
         "Coin3": "USDCe",
         "Price3": "init",
         "Amount3": "init",
         "Coin4": "POL",
         "Price4": "init",
         "Amount4": "init",
     }
   # Append the row to the DataFrame
    port_df = pd.concat([port_df, pd.DataFrame([row])], ignore_index=True)
    port_df.to_csv(current_filename, index=False)
    return port_df
   else:
       port_df = pd.read_csv(current_filename)
       return port_df
   
def log_port_event(Port_Dir, port_df, pubaddress, RPCURL, chainID, event="Just Checking"):
    web3 = Web3(Web3.HTTPProvider(RPCURL))

    WETH_contract = web3.eth.contract(address=os.getenv("poly_WETH_ca"), abi=os.getenv("poly_WETH_abi"))
    WBTC_contract = web3.eth.contract(address=os.getenv("poly_WBTC_ca"), abi=os.getenv("poly_WBTC_abi"))
    USDCe_contract = web3.eth.contract(address=os.getenv("poly_USDCe_ca"), abi=os.getenv("poly_USDCe_abi"))
    POL_contract = web3.eth.contract(address=os.getenv("poly_POL_ca"), abi=os.getenv("poly_POL_abi"))


    # Create a new row with the data
    row = {
         "Timestamp": "init",
         "ChainID": chainID,
         "Event": event,
         "Coin1": "WETH",
         "Price1": SOR_Quote("0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619", 18, 1, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", 6, chainID, pubaddress)[1],
         "Amount1": WETH_contract.functions.balanceOf(pubaddress).call() / (10 ** int(os.getenv("poly_WETH_dec"))),
         "Coin2": "WBTC",
         "Price2": SOR_Quote("0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6", 18, 1, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", 6, chainID, pubaddress)[1],
         "Amount2": WBTC_contract.functions.balanceOf(pubaddress).call() / (10 ** int(os.getenv("poly_WBTC_dec"))),
         "Coin3": "USDCe",
         "Price3": "1",
         "Amount3": USDCe_contract.functions.balanceOf(pubaddress).call() / (10 ** int(os.getenv("poly_USDCe_dec"))),
         "Coin4": "POL",
         "Amount": POL_contract.functions.balanceOf(pubaddress).call() / (10 ** int(os.getenv("poly_POL_dec"))),
     }
    
    # Append the row to the DataFrame
    port_df = pd.concat([port_df, pd.DataFrame([row])], ignore_index=True)
    filename = get_port_log_file_name(Port_Dir)
    port_df.to_csv(filename, index=False)
    return port_df



