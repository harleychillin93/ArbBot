import requests
import math
from time import sleep
from hexbytes import HexBytes
from web3 import Web3 
from eth_account import Account
from dotenv import load_dotenv
import os
from web3.middleware import PythonicMiddleware



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

def SOR_Quote(tokenInAdress: str, tknInDecimals: int, ammount: str, tokenOutAddress: str, tknOutDecimals: int, pubaddress: str):
    quote_url = "https://api.odos.xyz/sor/quote/v2"

    quote_request_body = {
        "chainId": 137, # Replace with desired chainId
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
        "slippageLimitPercent": 0.7, # set your slippage limit percentage (1 = 1%)
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

  
def SOR_Assemble(odos_quote, pubaddress: str):
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

def SOR_SendTxn(assembled_transaction, RPCURL, prikey):
  web3 = Web3(Web3.HTTPProvider(RPCURL))

  # Send TXN 1
  transaction1 = assembled_transaction["transaction"]
  # web3py requires the value to be an integer
  transaction1["value"] = int(transaction1["value"])
  signed_tx = web3.eth.account.sign_transaction(transaction1, prikey)
  # 4. Send the signed transaction
  tx_hash1 = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
  return tx_hash1.hex()
