![image](https://github.com/sr2echa/dirty-moni-detector/assets/65058816/f2d86ce3-9d19-499a-bc6e-7ef83f8f7083)

---

Here's the properly formatted content:

## Frontend
### STREAMLIT: [https://dirtymoneydetector.streamlit.app](https://dirtymoneydetector.streamlit.app)
🔗 [Source Code](./web)
---
### API LINK: [https://dirtyapi.replit.app/](https://dirtyapi.replit.app)

## API Reference
### Path: `/api/<wallet_add>`
### Output:
```json
{
  "Address": "string",
  "Avg min between received tnx": "float",
  "Avg min between sent tnx": "float",
  "AvgValSent": "float",
  "AvgValueReceived5Average": "float",
  "AvgValueSentToContract": "int",
  "ERC20AvgTimeBetweenRec_Tnx": "float",
  "ERC20AvgTimeBetweenSent_Tnx": "float",
  "ERC20AvgVal_Rec": "float",
  "ERC20AvgVal_Sent": "float",
  "ERC20MaxVal_Rec": "float",
  "ERC20MaxVal_Sent": "float",
  "ERC20MinVal_Rec": "float",
  "ERC20MinVal_Sent": "float",
  "ERC20MostRecTokenType": "string",
  "ERC20MostSentTokenType": "string",
  "ERC20TotalEtherSentContract": "float",
  "ERC20TotalEther_Received": "float",
  "ERC20TotalEther_Sent": "float",
  "ERC20UniqRecContractAddr": "int",
  "ERC20UniqRecTokenName": "int",
  "ERC20UniqRec_Addr": "int",
  "ERC20UniqSentTokenName": "int",
  "ERC20UniqSent_Addr": "int",
  "MaxValSent": "float",
  "MaxValueReceived": "float",
  "MaxValueSentToContract": "float",
  "MinValSent": "float",
  "MinValueReceived": "float",
  "MinValueSentToContract": "float",
  "NumberofCreated_Contracts": "int",
  "Received_tnx": "int",
  "Sent_tnx": "int",
  "Time Diff between first and_last (Mins)": "float",
  "TotalERC20Tnxs": "int",
  "TotalEtherBalance": "float",
  "TotalEtherReceived": "float",
  "TotalEtherSent": "float",
  "TotalEtherSent_Contracts": "float",
  "TotalTransactions": "int",
  "UniqueReceivedFrom_Addresses": "int",
  "UniqueSentTo_Addresses20": "int"
}
```
---
## ML Model
[sourcecode](./ML)

## How to Use 

1. Clone the Repository 
```bash 
 git clone https://github.com/sr2echa/dirty-moni-detector.git
```

2. Install all requirments and dependencies
```bash
pip install -r requirments.txt
```
3. Run the application:
```bash
streamlit run website.py
```

## Tech Stack 
<figure style="text-align: right;">
  <img src="web/Group 323.png" alt="Tech Stack" style="width:300px;height:300px;">
</figure>

- [x] Streamlit
- [x] Tensorflow
- [x] Github
- [x] replit
- [x] Flask
- [x] Seaborn

 
