from web3 import Web3 
from eth_account import Account
from dotenv import load_dotenv
import os
from ArbBotFuncs import polyGetTokenOrProxyAbi
from ArbBotFuncs import polyGetDecimals


load_dotenv()
polyRPCURL = os.getenv("polyRPCURL")

#Change this to get abi and decimals for any token
ca = os.getenv("WETH_ca")

abi = polyGetTokenOrProxyAbi(ca, polyRPCURL)
print(abi)

print('yes')

polyGetDecimals(ca, polyRPCURL)