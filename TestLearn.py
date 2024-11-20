from ArbBotFuncs import SOR_Quote
from ArbBotFuncs import SOR_Assemble
from ArbBotFuncs import SOR_SendTxn
from ArbBotFuncs import SOR_GetGas4Quote
from ArbBotFuncs import polyGetDecimals
from ArbBotFuncs import polyGetTokenOrProxyAbi
from web3 import Web3 
from eth_account import Account
from dotenv import load_dotenv
import os
from time import sleep
import requests

load_dotenv()

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
web3 = Web3(Web3.HTTPProvider(polyRPCURL))

ammount = 0.0015

abi = polyGetTokenOrProxyAbi(tkn1_ca_poly, polyRPCURL)
print(abi)

dec = polyGetDecimals(tkn1_ca_poly, polyRPCURL)
print(dec)


quote = SOR_Quote(tkn1_ca_poly, tkn1_dec_poly, ammount, tkn2_ca_poly, tkn2_dec_poly, pubaddress)
gas_estimate = SOR_GetGas4Quote(quote[0], polyRPCURL, pubaddress)
gas = web3.eth.gas_price
print(gas)
print(gas_estimate)
total_cost = (gas*gas_estimate) / (10 ** 18)
print(total_cost)

network = 'poly'; # could be any supported network
key = os.getenv("Owlracle_Key"); # fill your api key here
res = requests.get('https://api.owlracle.info/v4/{}/gas?apikey={}'.format(network, key))
data = res.json()
print(data)



txntest = 0

if txntest == 1:
    quote = SOR_Quote(tkn1_ca_poly, tkn1_dec_poly, ammount, tkn2_ca_poly, tkn2_dec_poly, pubaddress)
    odos_quote = quote[0]
    human_quote = quote[1]
    print(human_quote)
    assembledTxn = SOR_Assemble(odos_quote, pubaddress)
    nonce = assembledTxn['transaction']['nonce']
    print("hello")
    txn_hash = SOR_SendTxn(assembledTxn, polyRPCURL, prikey)
    print(txn_hash)

    quote = SOR_Quote(tkn2_ca_poly, tkn2_dec_poly, 5, tkn1_ca_poly, tkn1_dec_poly, pubaddress)
    odos_quote = quote[0]
    assembledTxn2 = SOR_Assemble(odos_quote, pubaddress)
    assembledTxn2['transaction']['nonce'] = nonce+1
    txn_hash2 = SOR_SendTxn(assembledTxn2, polyRPCURL, prikey)
    print(txn_hash2)
