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
print(tx_hash2)
