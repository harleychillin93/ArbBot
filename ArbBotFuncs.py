import requests
from web3 import Web3 
from eth_account import Account
from dotenv import load_dotenv
import os



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
    web3Poly.middleware_onion.inject(geth_poa_middleware, layer=0)
    contractAddr = Web3.to_checksum_address(Address)
    Abi = polyGetTokenOrProxyAbi(contractAddr,RPCURL)
    tokenContract = web3Poly.eth.contract(contractAddr, abi=Abi)
    return tokenContract.functions.decimals().call()
