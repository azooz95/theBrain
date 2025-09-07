# C:\Users\Omneya\theBrain\db\test_milvus.py
import os
from pymilvus import connections, utility

# clear proxies just in case
for k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"]:
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

print("Connecting to 127.0.0.1:19530 …")
connections.connect(alias="default", host="127.0.0.1", port="19530", secure=False, timeout=60.0)
print("Connected OK:", connections.has_connection("default"))
print("Milvus version:", utility.get_server_version())
print("Collections:", utility.list_collections())
