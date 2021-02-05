import base64
from pathlib import Path
import os
from seldon_core.seldon_client import SeldonClient

def test_data():
    root_path=Path(__file__).parent.parent
    print(root_path)
    with open(os.path.join(root_path,"images/pic1.jpeg"), "rb") as image_file:
        base64str = base64.b64encode(image_file.read())
    data = base64str
    return data

if __name__ == '__main__':

    sc = SeldonClient(deployment_name="DetectFaceClassifyMask", gateway_endpoint="localhost:5000")
    r = sc.predict(transport="grpc",str_data=test_data())
    print(r)
    #requests.post(' http://localhost:6000')
